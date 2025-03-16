#!/usr/bin/env python
# scripts/ingest_train.py
"""
Script to ingest the full train.json dataset into Pinecone.
This script creates a timestamped namespace to avoid conflicts with previous ingestions.
"""

import os
import sys
import asyncio
import time
import datetime
import json
from pathlib import Path
import argparse

# Add the project root to the Python path if running from scripts directory
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.db.pinecone_db import initialize_pinecone
from app.services.logging_service import setup_logger
from app.utils.logging_utils import logger
from scripts.ingest_data import ingest_dataset
from app.config import DATA_DIR

async def ingest_train_dataset(mark_test_docs: bool = False, test_doc_ids_file: str = None):
    """
    Ingest the full train.json dataset into Pinecone with a timestamped namespace.
    
    Args:
        mark_test_docs: Whether to mark documents as test documents
        test_doc_ids_file: Path to a JSON file containing test document IDs
    """
    # Create a timestamped namespace to avoid conflicts with previous ingestions
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    namespace = f"train_{timestamp}"
    
    logger.info(f"Starting ingestion of train.json into namespace '{namespace}'")
    
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
    
    # Record start time for performance measurement
    start_time = time.time()
    
    # Ingest the dataset with test document marking
    await ingest_dataset(
        file_name="train.json", 
        namespace=namespace,
        mark_test_docs=mark_test_docs,
        test_doc_ids=test_doc_ids
    )
    
    # Calculate and log total time
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    logger.info(f"Total ingestion time: {int(minutes)} minutes and {seconds:.2f} seconds")
    logger.info(f"Data is now available in Pinecone namespace: '{namespace}'")

def main():
    """
    Main entry point for the script.
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Ingest train.json into Pinecone")
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
        asyncio.run(ingest_train_dataset(
            mark_test_docs=args.mark_test_docs,
            test_doc_ids_file=args.test_doc_ids
        ))
    except KeyboardInterrupt:
        logger.info("Ingestion process interrupted by user")
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 