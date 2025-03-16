#!/usr/bin/env python
# scripts/generate_ground_truth.py
"""
Script to generate ground truth QA pairs from train.json for test document IDs.
This extracts the question-answer pairs from the annotation section of each test document.
"""

import json
import os
import sys
from pathlib import Path
import logging

# Add the project root to the Python path if running from scripts directory
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.config import DATA_DIR
from app.utils.logging_utils import logger

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

def generate_ground_truth():
    """
    Generate ground truth QA pairs from train.json for test document IDs.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load test document IDs
    test_doc_ids_path = Path(DATA_DIR) / "test_document_ids.json"
    try:
        with open(test_doc_ids_path, "r", encoding="utf-8") as f:
            test_doc_ids = json.load(f)
        logger.info(f"Loaded {len(test_doc_ids)} test document IDs from {test_doc_ids_path}")
    except Exception as e:
        logger.error(f"Failed to load test document IDs: {str(e)}")
        return
    
    # Create a set for faster lookups
    test_doc_ids_set = set(test_doc_ids)
    
    # Load train.json
    train_json_path = Path(DATA_DIR) / "train.json"
    if not train_json_path.exists():
        logger.error(f"File not found: {train_json_path}")
        return
    
    # Initialize ground truth dictionary
    ground_truth = {}
    
    # Process train.json line by line to avoid loading the entire file into memory
    logger.info(f"Processing {train_json_path}...")
    
    # Read the file as a JSON array
    try:
        with open(train_json_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        
        logger.info(f"Loaded {len(train_data)} documents from train.json")
        
        # Process each document
        found_docs = 0
        total_qa_pairs = 0
        
        for doc in train_data:
            doc_id = doc.get("id")
            
            # Check if this is a test document
            if doc_id in test_doc_ids_set:
                found_docs += 1
                logger.info(f"Processing test document {doc_id} ({found_docs}/{len(test_doc_ids_set)})")
                
                # Extract QA pairs from annotation
                if "annotation" in doc:
                    annotation = doc["annotation"]
                    dialogue_turns = annotation.get("dialogue_break", [])
                    turn_programs = annotation.get("turn_program", [])
                    exe_answers = annotation.get("exe_ans_list", [])
                    
                    # Create QA pairs
                    qa_pairs = []
                    for i, (question, program, answer) in enumerate(zip(dialogue_turns, turn_programs, exe_answers)):
                        qa_pair = {
                            "question": question,
                            "program": program,
                            "answer": answer,
                            "question_type": determine_question_type(question)
                        }
                        qa_pairs.append(qa_pair)
                    
                    # Add to ground truth
                    if qa_pairs:
                        ground_truth[doc_id] = qa_pairs
                        total_qa_pairs += len(qa_pairs)
                else:
                    logger.warning(f"Document {doc_id} has no annotation section")
        
        logger.info(f"Found {found_docs} test documents out of {len(test_doc_ids_set)}")
        logger.info(f"Extracted {total_qa_pairs} QA pairs in total")
        
        # Save ground truth to file
        ground_truth_path = Path(DATA_DIR) / "ground_truth.json"
        with open(ground_truth_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2)
        logger.info(f"Saved ground truth to {ground_truth_path}")
        
        # Create a flattened version for easier evaluation
        flattened_ground_truth = []
        for doc_id, qa_pairs in ground_truth.items():
            for qa_pair in qa_pairs:
                flattened_qa = {
                    "document_id": doc_id,
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "question_type": qa_pair["question_type"]
                }
                flattened_ground_truth.append(flattened_qa)
        
        # Save flattened ground truth to file
        flattened_path = Path(DATA_DIR) / "ground_truth_flattened.json"
        with open(flattened_path, "w", encoding="utf-8") as f:
            json.dump(flattened_ground_truth, f, indent=2)
        logger.info(f"Saved flattened ground truth to {flattened_path}")
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing train.json: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    generate_ground_truth() 