# scripts/ingest_data.py
import json
import asyncio
import sys
import logging
import re
from pathlib import Path
from typing import List, Dict, Any
import argparse
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter

from bs4 import BeautifulSoup

from app.config import DATA_DIR
from app.db.pinecone_db import initialize_pinecone, is_index_populated, upsert_documents
from app.services.logging_service import setup_logger
from app.utils.logging_utils import logger

def clean_text(text: str) -> str:
    """
    Clean the input text by:
      - Stripping HTML tags.
      - Lowercasing.
      - Removing extra whitespace.
      - Standardizing punctuation and numbers (basic example).
      
    Args:
        text: The raw text.
    
    Returns:
        Cleaned text.
    """
    if not isinstance(text, str):
        text = str(text)
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ")
    # Lowercase and normalize whitespace
    cleaned = cleaned.lower()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Remove extra punctuation (optional; adjust regex as needed)
    # Here we remove characters that are not alphanumerics, punctuation, or whitespace.
    # You might choose to preserve punctuation if needed.
    # cleaned = re.sub(r'[^\w\s\.\,\%\$\-]', '', cleaned)
    return cleaned.strip()

def enrich_metadata(text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich metadata with additional properties based on the cleaned text.
    
    Args:
        text: Cleaned text.
        metadata: Existing metadata dictionary.
    
    Returns:
        Enriched metadata dictionary.
    """
    word_count = len(text.split())
    text_length = len(text)
    enriched = metadata.copy()
    enriched.update({
        "cleaned": True,
        "word_count": word_count,
        "text_length": text_length
    })
    return enriched

def load_convfinqa_dataset(file_name: str) -> List[Dict[str, Any]]:
    """
    Load the ConvFinQA dataset from the local data directory.
    
    Args:
        file_name: Name of the JSON file (e.g., "train.json" or "dev.json")
                  Can be a relative path to DATA_DIR or an absolute path.
        
    Returns:
        A list of dictionaries representing financial documents.
    """
    # Check if file_name is an absolute path or contains a path separator
    if Path(file_name).is_absolute() or '/' in file_name:
        file_path = Path(file_name)
    else:
        file_path = Path(DATA_DIR) / file_name
        
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {len(data)} financial documents from {file_name}")
        return data
    except Exception as e:
        logger.error(f"Failed to load dataset {file_name}: {str(e)}")
        return []

def determine_question_type(question: str) -> str:
    """
    Determine the type of question based on its content.
    
    Args:
        question: The question text
        
    Returns:
        Question type as a string
    """
    question = question.lower()
    
    # Check for calculation questions
    if any(term in question for term in ["calculate", "compute", "what is the", "how much", 
                                        "difference", "increase", "decrease", "percentage", 
                                        "ratio", "total", "sum", "average"]):
        return "calculation"
    
    # Check for comparison questions
    elif any(term in question for term in ["compare", "more than", "less than", "higher", 
                                          "lower", "better", "worse", "versus", "vs"]):
        return "comparison"
    
    # Check for explanation questions
    elif any(term in question for term in ["explain", "why", "how does", "what is the reason"]):
        return "explanation"
    
    # Check for extraction questions
    elif any(term in question for term in ["what is", "what are", "list", "show me", "tell me"]):
        return "extraction"
    
    # Check for yes/no questions
    elif question.startswith(("is ", "are ", "does ", "do ", "can ", "will ", "has ", "have ")):
        return "yes_no"
    
    return "other"

def prepare_documents_for_embedding(financial_docs: List[Dict[str, Any]], mark_test_docs: bool = False, test_doc_ids: List[str] = None) -> List[Dict[str, Any]]:
    """
    Prepare ConvFinQA documents for embedding with special handling for test documents.
    
    Args:
        financial_docs: List of financial documents from the dataset
        mark_test_docs: Whether to mark documents as test documents
        test_doc_ids: List of document IDs to mark as test documents
        
    Returns:
        A list of document dictionaries with keys "id", "text", and enriched "metadata".
    """
    documents = []
    test_docs_count = 0
    question_types_count = {}
    
    for doc in financial_docs:
        doc_id = doc.get("id", f"doc_{len(documents)}")
        is_test_doc = mark_test_docs and test_doc_ids and doc_id in test_doc_ids
        
        # Clean and combine text components
        pre_text = " ".join(clean_text(t) for t in doc.get("pre_text", []))
        post_text = " ".join(clean_text(t) for t in doc.get("post_text", []))
        
        # Format table as text (clean each cell)
        table_rows = doc.get("table", [])
        table_text = ""
        if table_rows:
            table_text = "Table:\n"
            for row in table_rows:
                table_text += " | ".join(clean_text(str(cell)) for cell in row) + "\n"
        
        # Create the full document text WITHOUT including QA pairs for test documents
        full_text = f"Pre-text: {pre_text}\n\n{table_text}\n\nPost-text: {post_text}"
        full_text = clean_text(full_text)
        
        # Base metadata for the document
        base_metadata = {
            "source_id": doc_id,
            "document_type": "financial_document",
            "filename": doc.get("filename", ""),
        }
        
        # Add test document metadata if applicable
        if is_test_doc:
            test_docs_count += 1
            base_metadata["is_test"] = True
            
            # Store QA pairs in metadata but NOT in the indexed text
            if "annotation" in doc:
                annotation = doc["annotation"]
                dialogue_turns = annotation.get("dialogue_break", [])
                turn_programs = annotation.get("turn_program", [])
                exe_answers = annotation.get("exe_ans_list", [])
                
                # Store the conversation in metadata as a JSON string
                qa_pairs = [
                    {
                        "question": q,
                        "program": p,
                        "answer": a,
                        "question_type": determine_question_type(q)
                    }
                    for q, p, a in zip(dialogue_turns, turn_programs, exe_answers)
                ]
                base_metadata["qa_pairs_json"] = json.dumps(qa_pairs)
                
                # Track question types for statistics
                for q in dialogue_turns:
                    q_type = determine_question_type(q)
                    question_types_count[q_type] = question_types_count.get(q_type, 0) + 1
        
        # Create the document
        full_doc = {
            "id": f"{doc_id}_full",
            "text": full_text,
            "metadata": enrich_metadata(full_text, base_metadata)
        }
        documents.append(full_doc)
        
        # Process conversation turns if available and NOT a test document
        # For test documents, we don't want to include QA pairs in the indexed text
        if "annotation" in doc and not is_test_doc:
            annotation = doc["annotation"]
            dialogue_turns = annotation.get("dialogue_break", [])
            turn_programs = annotation.get("turn_program", [])
            exe_answers = annotation.get("exe_ans_list", [])
            
            for i, (question, program, answer) in enumerate(zip(dialogue_turns, turn_programs, exe_answers)):
                # Clean the current question
                clean_question = clean_text(question)
                context = ""
                for j in range(i):
                    clean_prev_q = clean_text(dialogue_turns[j])
                    clean_prev_a = clean_text(str(exe_answers[j]))
                    context += f"Q{j+1}: {clean_prev_q}\nA{j+1}: {clean_prev_a}\n\n"
                turn_text = f"Context:\n{context}\nDocument:\n{full_text}\n\nQuestion: {clean_question}"
                turn_doc = {
                    "id": f"{doc_id}_turn_{i+1}",
                    "text": turn_text,
                    "metadata": enrich_metadata(turn_text, {
                        "source_id": doc_id,
                        "document_type": "conversation_turn",
                        "turn_index": i,
                        "question": clean_question,
                        "program": program,
                        "answer": answer,
                        "filename": doc.get("filename", ""),
                    })
                }
                documents.append(turn_doc)
    
    if mark_test_docs:
        logger.info(f"Marked {test_docs_count} documents as test documents")
        logger.info(f"Question types in test documents: {question_types_count}")
    
    logger.info(f"Prepared {len(documents)} documents for embedding")
    return documents

def split_documents_if_needed(documents: List[Dict[str, Any]], max_tokens: int = 1024, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split documents into smaller chunks if they exceed the maximum token limit.
    We estimate token count as character count divided by 4.
    
    Args:
        documents: List of document dictionaries.
        max_tokens: Maximum allowed tokens per document.
        chunk_overlap: Overlap in tokens between chunks.
        
    Returns:
        A list of document dictionaries, possibly split into smaller chunks.
    """
    chunk_size_chars = max_tokens * 4
    overlap_chars = chunk_overlap * 4
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    result_docs = []
    for doc in documents:
        estimated_tokens = len(doc["text"]) / 4
        if estimated_tokens <= max_tokens:
            result_docs.append(doc)
        else:
            logger.info(f"Splitting document {doc['id']} (approx. {estimated_tokens:.0f} tokens)")
            splits = splitter.split_text(doc["text"])
            for i, split_text in enumerate(splits):
                split_doc = {
                    "id": f"{doc['id']}_part{i+1}",
                    "text": split_text,
                    "metadata": enrich_metadata(split_text, {
                        **doc["metadata"],
                        "is_split": True,
                        "part": i+1,
                        "total_parts": len(splits)
                    })
                }
                result_docs.append(split_doc)
    logger.info(f"After splitting: {len(result_docs)} documents")
    return result_docs

def clean_text_for_sparse(text: str) -> str:
    """
    Clean and preprocess text for sparse retrieval.
    Preserves financial symbols, numbers, and important punctuation.
    
    Args:
        text: The text to clean.
        
    Returns:
        Cleaned text.
    """
    # Store original text for reference
    original_text = text
    
    # Define financial symbols and abbreviations to preserve
    financial_symbols = ['$', '€', '£', '¥', '%']
    financial_abbr = ['USD', 'EUR', 'GBP', 'JPY', 'Q1', 'Q2', 'Q3', 'Q4', 'YOY', 'MOM', 'ROI', 'EPS', 'P/E']
    
    # Preserve financial abbreviations by replacing them with placeholders
    abbr_placeholders = {}
    for abbr in financial_abbr:
        if abbr in text:
            placeholder = f"__ABBR_{len(abbr_placeholders)}__"
            text = text.replace(abbr, placeholder)
            abbr_placeholders[placeholder] = abbr
    
    # Handle decimal numbers and percentages
    # This regex finds numbers with decimal points and optional percentage signs
    import re
    number_pattern = r'(\d+\.\d+%?|\d+%)'
    
    # Find all matches and create placeholders
    number_matches = re.findall(number_pattern, text)
    number_placeholders = {}
    
    for i, match in enumerate(number_matches):
        placeholder = f"__NUM_{i}__"
        text = text.replace(match, placeholder, 1)  # Replace only first occurrence
        number_placeholders[placeholder] = match
    
    # Normalize case but preserve acronyms
    text = text.lower()
    
    # Remove non-alphanumeric characters except important financial symbols
    # and punctuation that might be semantically important
    preserved_chars = financial_symbols + ['.', ',', ':', '-', '/', '(', ')']
    char_mapping = {ord(c): ' ' for c in set(c for c in '!"#&\'*+;<=>?@[\\]^_`{|}~') - set(preserved_chars)}
    text = text.translate(char_mapping)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Restore financial abbreviations
    for placeholder, abbr in abbr_placeholders.items():
        text = text.replace(placeholder, abbr)
    
    # Restore numbers with their original format
    for placeholder, number in number_placeholders.items():
        text = text.replace(placeholder, number)
    
    # Remove stopwords, but keep financial terms
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    
    # Financial terms that might be in stopwords but should be kept
    financial_stopwords_to_keep = ['up', 'down', 'over', 'under', 'above', 'below', 'between', 'against', 'for', 'at']
    
    # Remove these words from stopwords
    for word in financial_stopwords_to_keep:
        if word in stop_words:
            stop_words.remove(word)
    
    # Apply stopword removal
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    text = ' '.join(filtered_words)
    
    # Trim whitespace
    text = text.strip()
    
    return text

def prepare_documents_for_sparse(financial_docs: List[Dict[str, Any]], mark_test_docs: bool = False, test_doc_ids: List[str] = None) -> List[Dict[str, Any]]:
    """
    Prepare ConvFinQA documents specifically for sparse indexing with special handling for test documents.
    
    Args:
        financial_docs: List of financial documents from the dataset
        mark_test_docs: Whether to mark documents as test documents
        test_doc_ids: List of document IDs to mark as test documents
        
    Returns:
        A list of document dictionaries with keys "id", "text", and enriched "metadata".
    """
    documents = []
    test_docs_count = 0
    question_types_count = {}
    
    for doc in financial_docs:
        doc_id = doc.get("id", f"doc_{len(documents)}")
        is_test_doc = mark_test_docs and test_doc_ids and doc_id in test_doc_ids
        
        # Clean and combine text components with sparse-specific cleaning
        pre_text = " ".join(clean_text_for_sparse(t) for t in doc.get("pre_text", []))
        post_text = " ".join(clean_text_for_sparse(t) for t in doc.get("post_text", []))
        
        # Format table as text (clean each cell)
        table_rows = doc.get("table", [])
        table_text = ""
        if table_rows:
            table_text = "Table:\n"
            for row in table_rows:
                table_text += " | ".join(clean_text_for_sparse(str(cell)) for cell in row) + "\n"
        
        # Create the full document text WITHOUT including QA pairs for test documents
        full_text = f"Pre-text: {pre_text}\n\n{table_text}\n\nPost-text: {post_text}"
        full_text = clean_text_for_sparse(full_text)
        
        # Base metadata for the document
        base_metadata = {
            "source_id": doc_id,
            "document_type": "financial_document",
            "filename": doc.get("filename", ""),
            "sparse_optimized": True
        }
        
        # Add test document metadata if applicable
        if is_test_doc:
            test_docs_count += 1
            base_metadata["is_test"] = True
            
            # Store QA pairs in metadata but NOT in the indexed text
            if "annotation" in doc:
                annotation = doc["annotation"]
                dialogue_turns = annotation.get("dialogue_break", [])
                turn_programs = annotation.get("turn_program", [])
                exe_answers = annotation.get("exe_ans_list", [])
                
                # Store the conversation in metadata as a JSON string
                qa_pairs = [
                    {
                        "question": q,
                        "program": p,
                        "answer": a,
                        "question_type": determine_question_type(q)
                    }
                    for q, p, a in zip(dialogue_turns, turn_programs, exe_answers)
                ]
                base_metadata["qa_pairs_json"] = json.dumps(qa_pairs)
                
                # Track question types for statistics
                for q in dialogue_turns:
                    q_type = determine_question_type(q)
                    question_types_count[q_type] = question_types_count.get(q_type, 0) + 1
        
        # Create the document
        full_doc = {
            "id": f"{doc_id}_full_sparse",
            "text": full_text,
            "metadata": enrich_metadata(full_text, base_metadata)
        }
        documents.append(full_doc)
        
        # Process conversation turns if available and NOT a test document
        # For test documents, we don't want to include QA pairs in the indexed text
        if "annotation" in doc and not is_test_doc:
            annotation = doc["annotation"]
            dialogue_turns = annotation.get("dialogue_break", [])
            turn_programs = annotation.get("turn_program", [])
            exe_answers = annotation.get("exe_ans_list", [])
            
            for i, (question, program, answer) in enumerate(zip(dialogue_turns, turn_programs, exe_answers)):
                # Clean the current question
                clean_question = clean_text_for_sparse(question)
                context = ""
                for j in range(i):
                    clean_prev_q = clean_text_for_sparse(dialogue_turns[j])
                    clean_prev_a = clean_text_for_sparse(str(exe_answers[j]))
                    context += f"Q{j+1}: {clean_prev_q}\nA{j+1}: {clean_prev_a}\n\n"
                turn_text = f"Context:\n{context}\nDocument:\n{full_text}\n\nQuestion: {clean_question}"
                turn_doc = {
                    "id": f"{doc_id}_turn_{i+1}_sparse",
                    "text": turn_text,
                    "metadata": enrich_metadata(turn_text, {
                        "source_id": doc_id,
                        "document_type": "conversation_turn",
                        "turn_index": i,
                        "question": clean_question,
                        "program": program,
                        "answer": answer,
                        "filename": doc.get("filename", ""),
                        "sparse_optimized": True
                    })
                }
                documents.append(turn_doc)
    
    if mark_test_docs:
        logger.info(f"Marked {test_docs_count} documents as test documents for sparse indexing")
        logger.info(f"Question types in test documents (sparse): {question_types_count}")
    
    logger.info(f"Prepared {len(documents)} documents for sparse indexing")
    return documents

async def ingest_dataset(
    file_name: str = "train.json", 
    namespace: str = "",
    index_name: str = None,
    is_sparse: bool = False,
    mark_test_docs: bool = False,
    test_doc_ids: List[str] = None
) -> None:
    """
    Ingest and index a ConvFinQA dataset from the local DATA_DIR into Pinecone.
    
    Steps:
      1. Initialize Pinecone and the embedding model.
      2. Check if the index/namespace is already populated (to prevent duplicates).
      3. Load and prepare the dataset.
      4. Split long documents into overlapping chunks.
      5. Upsert the processed documents into Pinecone.
    
    Args:
        file_name: The JSON file to ingest (default "train.json").
        namespace: Optional namespace for the Pinecone index.
        index_name: Optional name of the Pinecone index to use.
        is_sparse: Whether to use sparse vectors for this ingestion.
        mark_test_docs: Whether to mark documents as test documents.
        test_doc_ids: List of document IDs to mark as test documents.
    """
    setup_logger()
    initialize_pinecone(index_name=index_name)

    if is_index_populated(namespace, index_name=index_name):
        logger.info(f"Index already contains data in namespace '{namespace}'. Skipping ingestion.")
        return

    financial_docs = load_convfinqa_dataset(file_name)
    if not financial_docs:
        logger.error(f"No data loaded from {file_name}. Aborting ingestion.")
        return

    # Use different document preparation based on whether we're using sparse or dense vectors
    if is_sparse:
        logger.info("Preparing documents for sparse indexing")
        documents = prepare_documents_for_sparse(financial_docs, mark_test_docs, test_doc_ids)
    else:
        logger.info("Preparing documents for dense indexing")
        documents = prepare_documents_for_embedding(financial_docs, mark_test_docs, test_doc_ids)
    
    processed_docs = split_documents_if_needed(documents, max_tokens=1024, chunk_overlap=200)
    
    success = await upsert_documents(
        processed_docs, 
        namespace=namespace, 
        index_name=index_name,
        is_sparse=is_sparse
    )
    if success:
        logger.info(f"Successfully ingested {len(processed_docs)} documents from '{file_name}' into namespace '{namespace}'")
    else:
        logger.error(f"Failed to ingest dataset '{file_name}' into Pinecone")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data into Pinecone index")
    parser.add_argument("file_name", nargs="?", default="train.json", help="JSON file to ingest (default: train.json)")
    parser.add_argument("--namespace", default="", help="Namespace for the Pinecone index")
    parser.add_argument("--index-name", help="Name of the Pinecone index to use")
    parser.add_argument("--sparse", action="store_true", help="Use sparse vectors for ingestion")
    parser.add_argument("--mark-test-docs", action="store_true", help="Mark documents as test documents")
    parser.add_argument("--test-doc-ids", help="Path to a JSON file containing test document IDs")
    
    args = parser.parse_args()
    
    # Load test document IDs if specified
    test_doc_ids = None
    if args.mark_test_docs and args.test_doc_ids:
        try:
            with open(args.test_doc_ids, 'r') as f:
                test_doc_ids = json.load(f)
            logger.info(f"Loaded {len(test_doc_ids)} test document IDs from {args.test_doc_ids}")
        except Exception as e:
            logger.error(f"Failed to load test document IDs: {str(e)}")
    
    asyncio.run(ingest_dataset(
        file_name=args.file_name,
        namespace=args.namespace,
        index_name=args.index_name,
        is_sparse=args.sparse,
        mark_test_docs=args.mark_test_docs,
        test_doc_ids=test_doc_ids
    ))