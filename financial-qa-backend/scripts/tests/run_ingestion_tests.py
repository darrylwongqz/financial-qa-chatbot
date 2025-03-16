#!/usr/bin/env python
# scripts/tests/run_ingestion_tests.py
"""
Main test runner script for ingestion tests.
This script runs both dense and sparse ingestion tests.
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Add the project root to the Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.services.logging_service import setup_logger
from app.utils.logging_utils import logger

def setup_nltk():
    """
    Set up NLTK resources if needed.
    This is a no-op in tests since we mock the NLTK functions.
    """
    try:
        import nltk
        # We don't actually need to download anything since we're mocking the functions
        logger.info("NLTK setup complete")
    except ImportError:
        logger.warning("NLTK not installed, but tests will still run with mocks")
    except Exception as e:
        logger.warning(f"Error setting up NLTK: {str(e)}, but tests will still run with mocks")

def run_dense_tests():
    """Run dense ingestion tests."""
    logger.info("Running dense ingestion tests...")
    try:
        from scripts.tests.test_dense_ingestion import main as dense_main
        dense_main()
    except Exception as e:
        logger.error(f"Error running dense ingestion tests: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    return True

def run_sparse_tests():
    """Run sparse ingestion tests."""
    logger.info("Running sparse ingestion tests...")
    try:
        from scripts.tests.test_sparse_ingestion import main as sparse_main
        sparse_main()
    except Exception as e:
        logger.error(f"Error running sparse ingestion tests: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    return True

def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run ingestion tests")
    parser.add_argument("--dense", action="store_true", help="Run dense ingestion tests")
    parser.add_argument("--sparse", action="store_true", help="Run sparse ingestion tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()
    
    # Set up logging
    setup_logger()
    
    # Set up NLTK
    setup_nltk()
    
    # Determine which tests to run
    run_dense = args.dense or args.all or (not args.dense and not args.sparse)
    run_sparse = args.sparse or args.all or (not args.dense and not args.sparse)
    
    # Track success
    success = True
    
    # Run the tests
    if run_dense:
        dense_success = run_dense_tests()
        success = success and dense_success
    
    if run_sparse:
        sparse_success = run_sparse_tests()
        success = success and sparse_success
    
    if success:
        logger.info("All tests completed successfully!")
    else:
        logger.error("Some tests failed. See logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 