# app/db/pinecone_db.py
from typing import List, Dict, Any, Optional
import pinecone
import logging
from app.utils.logging_utils import logger
from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, OPENAI_API_KEY, EMBEDDING_MODEL, SPARSE_INDEX_NAME
from langchain_openai.embeddings import OpenAIEmbeddings
import asyncio

# Global variables for Pinecone and embedding model
pc = None
index = None
sparse_index = None
dense_index = None
embedding_model = None
_is_initialized = False
_sparse_initialized = False

def initialize_pinecone(index_name: str = None):
    """
    Initialize Pinecone and the embedding model using LangChain.
    This function can be called multiple times but will only initialize once.
    
    Args:
        index_name: Optional name of the Pinecone index to use.
                   If not provided, uses PINECONE_INDEX_NAME from config.
    """
    global pc, index, sparse_index, dense_index, embedding_model, _is_initialized, _sparse_initialized
    
    # If index_name is provided, we're initializing a specific index
    if index_name is not None:
        # For sparse index
        if index_name == SPARSE_INDEX_NAME and not _sparse_initialized:
            try:
                if pc is None:
                    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
                sparse_index = pc.Index(SPARSE_INDEX_NAME)
                logger.info(f"Successfully connected to Pinecone sparse index: {SPARSE_INDEX_NAME}")
                _sparse_initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone sparse index: {str(e)}")
                sparse_index = None
            # Don't return here, continue to initialize the embedding model if needed
        # For the dense index specifically
        elif index_name == PINECONE_INDEX_NAME:
            try:
                if pc is None:
                    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
                index = pc.Index(PINECONE_INDEX_NAME)
                dense_index = index  # Set dense_index to the same as index
                logger.info(f"Successfully connected to Pinecone dense index: {PINECONE_INDEX_NAME}")
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone dense index: {str(e)}")
                index = None
                dense_index = None
            # Don't return here, continue to initialize the embedding model if needed
        # For any other specific index
        else:
            try:
                if pc is None:
                    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
                specific_index = pc.Index(index_name)
                logger.info(f"Successfully connected to Pinecone index: {index_name}")
                return specific_index
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone index {index_name}: {str(e)}")
                return None
    
    # Default initialization for the main index and embedding model
    if _is_initialized:
        logger.info("Pinecone already initialized, skipping")
        return
    
    pinecone_initialized = False
    sparse_pinecone_initialized = False
    embedding_initialized = False
    
    try:
        if pc is None:
            pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes]
        
        # Initialize the main dense index if not already initialized
        if index is None and dense_index is None:
            if PINECONE_INDEX_NAME not in index_names:
                logger.warning(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist. It will need to be created before ingestion.")
            else:
                index = pc.Index(PINECONE_INDEX_NAME)
                dense_index = index  # Set dense_index to the same as index
                logger.info(f"Successfully connected to Pinecone index: {PINECONE_INDEX_NAME}")
                pinecone_initialized = True
        else:
            # Index already initialized
            pinecone_initialized = True
        
        # Also initialize the sparse index if it exists and not already initialized
        if sparse_index is None and not _sparse_initialized:
            if SPARSE_INDEX_NAME not in index_names:
                logger.warning(f"Pinecone sparse index '{SPARSE_INDEX_NAME}' does not exist. It will need to be created before ingestion.")
            else:
                sparse_index = pc.Index(SPARSE_INDEX_NAME)
                logger.info(f"Successfully connected to Pinecone sparse index: {SPARSE_INDEX_NAME}")
                sparse_pinecone_initialized = True
                _sparse_initialized = True
            
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        if index is None:  # Only set to None if not already initialized
            index = None
            dense_index = None
        if sparse_index is None:  # Only set to None if not already initialized
            sparse_index = None

    # Initialize embedding model if not already initialized
    if embedding_model is None:
        try:
            embedding_model = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY,
                chunk_size=1000
            )
            logger.info(f"Successfully initialized LangChain embedding model: {EMBEDDING_MODEL}")
            embedding_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize LangChain embedding model: {str(e)}")
            embedding_model = None
    else:
        # Embedding model already initialized
        embedding_initialized = True
    
    # Only set _is_initialized to True if both components are successfully initialized
    _is_initialized = pinecone_initialized and embedding_initialized
    if not _is_initialized:
        logger.warning("Pinecone initialization incomplete. Some components failed to initialize.")

def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding for the given text using LangChain's OpenAIEmbeddings.
    
    Args:
        text: The text to embed.
        
    Returns:
        List of floats representing the embedding.
    """
    if embedding_model is None:
        raise Exception("Embedding model not initialized")
    try:
        embedding = embedding_model.embed_query(text)
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts using LangChain's OpenAIEmbeddings.
    
    Args:
        texts: List of texts to embed.
        
    Returns:
        List of embeddings.
    """
    if embedding_model is None:
        raise Exception("Embedding model not initialized")
    try:
        embeddings = embedding_model.embed_documents(texts)
        return embeddings
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {str(e)}")
        raise

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using LangChain's OpenAIEmbeddings.
    This is a synchronous version of get_embeddings_batch.
    
    Args:
        texts: List of texts to embed.
        
    Returns:
        List of embeddings.
    """
    if embedding_model is None:
        raise Exception("Embedding model not initialized")
    try:
        embeddings = embedding_model.embed_documents(texts)
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def is_index_populated(namespace: str = "", index_name: str = None) -> bool:
    """
    Check if the Pinecone index is already populated with vectors for the given namespace.
    
    Args:
        namespace: The namespace to check.
        index_name: Optional name of the Pinecone index to check.
                   If not provided, uses the current index.
    
    Returns:
        True if there are vectors in the index; False otherwise.
    """
    # If index_name is provided, we need to get that specific index
    target_index = index
    if index_name:
        if index_name == SPARSE_INDEX_NAME:
            target_index = sparse_index
            if target_index is None:
                # Try to initialize the sparse index
                initialize_pinecone(index_name=SPARSE_INDEX_NAME)
                from app.db.pinecone_db import sparse_index as current_sparse_index
                target_index = current_sparse_index
        elif index_name == PINECONE_INDEX_NAME:
            target_index = dense_index
            if target_index is None:
                # Try to initialize the dense index
                initialize_pinecone()
                from app.db.pinecone_db import dense_index as current_dense_index
                target_index = current_dense_index
        else:
            try:
                target_index = pc.Index(index_name)
            except Exception as e:
                logger.error(f"Error getting index {index_name}: {str(e)}")
                return False
    
    if target_index is None:
        logger.error("Pinecone index not initialized")
        return False
    try:
        stats = target_index.describe_index_stats(namespace=namespace)
        vector_count = stats.get("total_vector_count", 0)
        logger.info(f"Index has {vector_count} vectors in namespace '{namespace}'")
        return vector_count > 0
    except Exception as e:
        logger.error(f"Error describing index stats: {str(e)}")
        return False

async def query_pinecone(
    query: str,
    top_k: int = 5,
    namespace: str = "",
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Query Pinecone for relevant documents.
    
    Args:
        query: The query text.
        top_k: Number of results to return.
        namespace: Pinecone namespace.
        filter_metadata: Optional filters.
        
    Returns:
        List of documents with text and metadata.
    """
    if index is None:
        logger.error("Pinecone index not initialized")
        return []
    try:
        query_embedding = get_embedding(query)
        query_params = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": namespace
        }
        if filter_metadata:
            query_params["filter"] = filter_metadata
        results = index.query(**query_params)
        documents = []
        for match in results.matches:
            documents.append({
                "id": match.id,
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata,
                "score": match.score
            })
        return documents
    except Exception as e:
        logger.error(f"Error querying Pinecone: {str(e)}")
        return []

async def upsert_documents(
    documents: List[Dict[str, Any]],
    namespace: str = "",
    batch_size: int = 100,
    index_name: str = None,
    is_sparse: bool = False
) -> bool:
    """
    Upsert a list of documents into Pinecone.
    
    Args:
        documents: List of document dicts (each must have a "text" field).
        namespace: Pinecone namespace.
        batch_size: Batch size for upserting.
        index_name: Optional name of the Pinecone index to use.
                   If not provided, uses the current index.
        is_sparse: Whether to use sparse vectors for this ingestion.
        
    Returns:
        True if successful, False otherwise.
    """
    # If index_name is provided, we need to get that specific index
    target_index = index
    if index_name:
        if index_name == SPARSE_INDEX_NAME:
            target_index = sparse_index
            if target_index is None:
                # Try to initialize the sparse index
                initialize_pinecone(index_name=SPARSE_INDEX_NAME)
                from app.db.pinecone_db import sparse_index as current_sparse_index
                target_index = current_sparse_index
        elif index_name != PINECONE_INDEX_NAME:
            try:
                target_index = pc.Index(index_name)
            except Exception as e:
                logger.error(f"Error getting index {index_name}: {str(e)}")
                return False
    
    if target_index is None:
        logger.error("Pinecone index not initialized")
        return False
    try:
        total_docs = len(documents)
        logger.info(f"Upserting {total_docs} documents in batches of {batch_size}")
        
        # For sparse index, create a consistent TF-IDF vectorizer for all batches
        if is_sparse:
            from sklearn.feature_extraction.text import TfidfVectorizer
            logger.info("Preparing TF-IDF vectorizer on all documents...")
            all_texts = [doc["text"] for doc in documents]
            vectorizer = TfidfVectorizer(max_features=1000)
            vectorizer.fit(all_texts)
            logger.info(f"Fitted TF-IDF vectorizer with {len(vectorizer.get_feature_names_out())} features")
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            texts = [doc["text"] for doc in batch]
            ids = [doc.get("id", f"doc_{i+j}") for j, doc in enumerate(batch)]
            
            # Generate embeddings
            if is_sparse:
                # For sparse index, use the pre-fitted vectorizer
                sparse_matrix = vectorizer.transform(texts)
                
                vectors = []
                for j, (doc, sparse_vector) in enumerate(zip(batch, sparse_matrix)):
                    # Convert sparse vector to indices and values format
                    indices = sparse_vector.indices.tolist()
                    values = sparse_vector.data.tolist()
                    
                    # Create sparse values dict
                    sparse_values = {
                        "indices": indices,
                        "values": values
                    }
                    
                    vector = {
                        "id": ids[j],
                        "sparse_values": sparse_values,
                        "metadata": {"text": doc["text"], **doc.get("metadata", {})}
                    }
                    vectors.append(vector)
            else:
                # For dense index, use the embedding model
                embeddings = await get_embeddings_batch(texts)
                vectors = []
                for j, (doc, embedding) in enumerate(zip(batch, embeddings)):
                    vector = {
                        "id": ids[j],
                        "values": embedding,
                        "metadata": {"text": doc["text"], **doc.get("metadata", {})}
                    }
                    vectors.append(vector)
            
            # Add retry logic for rate limiting
            max_retries = 5
            retry_count = 0
            retry_delay = 1.0
            
            while retry_count <= max_retries:
                try:
                    target_index.upsert(vectors=vectors, namespace=namespace)
                    logger.info(f"Upserted batch {i//batch_size + 1} of {((total_docs-1)//batch_size) + 1}")
                    break
                except Exception as e:
                    retry_count += 1
                    error_message = str(e)
                    
                    if "429" in error_message or "TOO_MANY_REQUESTS" in error_message:
                        logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Attempt {retry_count}/{max_retries})")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    elif retry_count < max_retries:
                        logger.warning(f"Error upserting batch: {error_message}. Retrying in {retry_delay} seconds... (Attempt {retry_count}/{max_retries})")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Failed to upsert batch after {max_retries} retries: {error_message}")
                        raise
            
            # Add a small delay between batches to avoid rate limiting
            await asyncio.sleep(0.5)
            
        logger.info(f"Successfully upserted {total_docs} documents")
        return True
    except Exception as e:
        logger.error(f"Error upserting documents: {str(e)}")
        return False

def list_namespaces() -> List[str]:
    """
    List all available namespaces in the Pinecone index.
    
    Returns:
        List of namespace names.
    """
    if index is None:
        logger.error("Pinecone index not initialized")
        return []
    try:
        stats = index.describe_index_stats()
        namespaces = list(stats.get("namespaces", {}).keys())
        logger.info(f"Found {len(namespaces)} namespaces in Pinecone index")
        for ns in namespaces:
            vector_count = stats["namespaces"].get(ns, {}).get("vector_count", 0)
            logger.info(f"  - {ns}: {vector_count} vectors")
        return namespaces
    except Exception as e:
        logger.error(f"Error listing namespaces: {str(e)}")
        return []