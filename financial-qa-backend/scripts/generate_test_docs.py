#!/usr/bin/env python
# scripts/generate_test_docs.py
"""
Script to generate test document IDs from the ConvFinQA dataset.
This script selects a percentage of documents as test documents,
ensuring a balanced distribution of question types.
"""

import json
import random
import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the Python path if running from scripts directory
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.config import DATA_DIR
from app.services.logging_service import setup_logger
from app.utils.logging_utils import logger
from scripts.ingest_data import determine_question_type

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load the ConvFinQA dataset.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        List of documents from the dataset
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load dataset from {file_path}: {str(e)}")
        return []

def select_test_documents(dataset: List[Dict[str, Any]], percentage: float = 0.2) -> List[str]:
    """
    Select a percentage of documents as test documents.
    Ensures a balanced distribution of question types.
    
    Args:
        dataset: List of documents from the dataset
        percentage: Percentage of documents to select as test documents
        
    Returns:
        List of document IDs selected as test documents
    """
    # Group documents by question types
    question_type_docs = {}
    
    for doc in dataset:
        if "annotation" in doc and "dialogue_break" in doc["annotation"]:
            # Use the first question to determine the document's primary question type
            first_question = doc["annotation"]["dialogue_break"][0] if doc["annotation"]["dialogue_break"] else ""
            question_type = determine_question_type(first_question)
            
            if question_type not in question_type_docs:
                question_type_docs[question_type] = []
            
            question_type_docs[question_type].append(doc["id"])
    
    # Select documents from each question type
    selected_docs = []
    for q_type, docs in question_type_docs.items():
        num_to_select = max(1, int(len(docs) * percentage))
        selected = random.sample(docs, num_to_select)
        selected_docs.extend(selected)
        logger.info(f"Selected {len(selected)} documents of type '{q_type}' (out of {len(docs)})")
    
    return selected_docs

def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description="Generate test document IDs from ConvFinQA dataset")
    parser.add_argument("--input", default=str(DATA_DIR / "train.json"), 
                        help="Path to the ConvFinQA dataset")
    parser.add_argument("--output", default=None, 
                        help="Output file for test document IDs (defaults to test_document_ids.json in the same directory as input)")
    parser.add_argument("--percentage", type=float, default=0.2, 
                        help="Percentage of documents to select as test documents")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logger()
    
    # Load the dataset
    dataset = load_dataset(args.input)
    logger.info(f"Loaded {len(dataset)} documents from {args.input}")
    
    if not dataset:
        logger.error("No documents loaded. Exiting.")
        sys.exit(1)
    
    # Select test documents
    test_docs = select_test_documents(dataset, args.percentage)
    logger.info(f"Selected {len(test_docs)} test documents ({args.percentage * 100:.1f}% of dataset)")
    
    # Determine output path - if not specified, use the same directory as input
    if args.output is None:
        input_path = Path(args.input)
        output_path = input_path.parent / "test_document_ids.json"
    else:
        output_path = Path(args.output)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save test document IDs
    try:
        with open(output_path, 'w') as f:
            json.dump(test_docs, f, indent=2)
        logger.info(f"Saved {len(test_docs)} test document IDs to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save test document IDs: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 