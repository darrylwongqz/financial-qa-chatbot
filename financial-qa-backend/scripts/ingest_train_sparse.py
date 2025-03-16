#!/usr/bin/env python
# scripts/ingest_train_sparse.py
"""
Script to ingest the full train.json dataset into the Pinecone sparse index.
This script uses the sparse namespace defined in the config.
Uses the shared functionality from ingest_data.py and pinecone_db.py to avoid duplication.
"""

import os
import sys
import asyncio
import time
import json
import traceback
from pathlib import Path
import argparse

# Add the project root to the Python path if running from scripts directory
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.db.pinecone_db import initialize_pinecone, is_index_populated, upsert_documents
from app.services.logging_service import setup_logger
from app.utils.logging_utils import logger
from scripts.ingest_data import load_convfinqa_dataset, prepare_documents_for_sparse, split_documents_if_needed
from app.config import SPARSE_INDEX_NAME, SPARSE_NAMESPACE, DATA_DIR

async def ingest_train_dataset_sparse(mark_test_docs: bool = False, test_doc_ids_file: str = None):
    """
    Ingest the full train.json dataset into the Pinecone sparse index.
    Uses the shared upsert_documents function from pinecone_db.py.
    
    Args:
        mark_test_docs: Whether to mark documents as test documents
        test_doc_ids_file: Path to a JSON file containing test document IDs
    """
    logger.info(f"Starting ingestion of train.json into sparse index '{SPARSE_INDEX_NAME}' with namespace '{SPARSE_NAMESPACE}'")
    
    # Load test document IDs if specified
    test_doc_ids = None
    if mark_test_docs and test_doc_ids_file:
        try:
            with open(test_doc_ids_file, 'r') as f:
                test_doc_ids = json.load(f)
            logger.info(f"Loaded {len(test_doc_ids)} test document IDs from {test_doc_ids_file}")
        except Exception as e:
            logger.error(f"Failed to load test document IDs: {str(e)}")
            test_doc_ids = []
    
    # Check if the namespace already has data
    initialize_pinecone(index_name=SPARSE_INDEX_NAME)
    if is_index_populated(SPARSE_NAMESPACE, index_name=SPARSE_INDEX_NAME):
        logger.info(f"Namespace '{SPARSE_NAMESPACE}' already contains data. Skipping ingestion.")
        return
    
    # Record start time for performance measurement
    start_time = time.time()
    
    # Load and prepare the dataset
    file_name = str(DATA_DIR / "train.json")
    logger.info(f"Loading dataset from {file_name}...")
    financial_docs = load_convfinqa_dataset(file_name)
    if not financial_docs:
        logger.error(f"No data loaded from {file_name}. Aborting ingestion.")
        return
    
    # Prepare documents for sparse indexing
    logger.info("Preparing documents for sparse indexing...")
    documents = prepare_documents_for_sparse(financial_docs, mark_test_docs=mark_test_docs, test_doc_ids=test_doc_ids)
    
    # Split documents if needed
    logger.info("Splitting documents if needed...")
    processed_docs = split_documents_if_needed(documents, max_tokens=1024, chunk_overlap=200)
    
    # Upsert documents using the shared function from pinecone_db.py
    success = await upsert_documents(
        documents=processed_docs,
        namespace=SPARSE_NAMESPACE,
        batch_size=50,  # Smaller batch size for better handling
        index_name=SPARSE_INDEX_NAME,
        is_sparse=True
    )
    
    if not success:
        logger.error("Failed to ingest dataset into Pinecone")
        return
    
    # Calculate and log total time
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    logger.info(f"Total ingestion time: {int(minutes)} minutes and {seconds:.2f} seconds")
    logger.info(f"Data is now available in Pinecone sparse index: '{SPARSE_INDEX_NAME}' with namespace: '{SPARSE_NAMESPACE}'")

def main():
    """
    Main entry point for the script.
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Ingest train.json into Pinecone sparse index")
    parser.add_argument("--mark-test-docs", action="store_true", help="Mark documents as test documents")
    parser.add_argument("--test-doc-ids", default=str(DATA_DIR / "test_document_ids.json"), 
                        help="Path to a JSON file containing test document IDs")
    args = parser.parse_args()
    
    # Set up logging
    setup_logger()
    
    # Initialize Pinecone
    initialize_pinecone()
    
    try:
        # Run the ingestion process
        asyncio.run(ingest_train_dataset_sparse(
            mark_test_docs=args.mark_test_docs,
            test_doc_ids_file=args.test_doc_ids
        ))
    except KeyboardInterrupt:
        logger.info("Ingestion process interrupted by user")
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 