#!/usr/bin/env python
# scripts/run_evaluation.py
"""
Script to run an evaluation of the Financial QA chatbot using the ground truth data.
"""

import os
import sys
import argparse
import asyncio
import uuid
import logging
from pathlib import Path

# Add the project root to the Python path if running from scripts directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from app.services.evaluation_service import run_evaluation
from app.db.pinecone_db import initialize_pinecone
from app.db.firestore import initialize_firestore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_evaluation_and_wait(retrieval_profile, model, limit=None):
    """
    Run evaluation and wait for it to complete.
    
    Args:
        retrieval_profile: The retrieval profile to use
        model: The model to use
        limit: Optional limit on the number of QA pairs to evaluate
    """
    try:
        # Initialize Pinecone
        try:
            initialize_pinecone()
            logger.info("Pinecone initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            sys.exit(1)
        
        # Initialize Firestore
        try:
            # Use the correct path to the service key file
            initialize_firestore()
            logger.info("Firestore initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore: {str(e)}")
            # Continue even if Firestore initialization fails
        
        # Generate a unique ID for the evaluation
        evaluation_id = str(uuid.uuid4())
        
        # Set user ID
        user_id = "admin@email.com"
        
        logger.info(f"Starting evaluation {evaluation_id} with retrieval profile {retrieval_profile}, model {model}, limit {limit}, user_id {user_id}")
        
        # Run evaluation
        evaluation = await run_evaluation(evaluation_id, retrieval_profile, model, limit, user_id)
        
        # Get the project root directory
        project_root = Path(__file__).resolve().parent.parent
        results_dir = project_root / "app" / "data" / "evaluation_results"
        results_file = results_dir / f"evaluation_{evaluation_id}.json"
        
        logger.info(f"Evaluation completed with ID: {evaluation_id}")
        logger.info(f"Results saved to: {results_file}")
        
        # Return the evaluation ID
        return evaluation_id
    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")
        raise

def main():
    """Run the evaluation script."""
    parser = argparse.ArgumentParser(description='Run evaluation for the Financial QA chatbot')
    parser.add_argument(
        '--retrieval-profile',
        type=str,
        choices=['fast', 'balanced', 'accurate'],
        required=True,
        help='The retrieval profile to use'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['gpt-3.5-turbo', 'gpt-4'],
        required=True,
        help='The model to use'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit the number of QA pairs to evaluate'
    )
    
    args = parser.parse_args()
    
    # Ensure the evaluation results directory exists
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "app" / "data" / "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run the evaluation
    asyncio.run(run_evaluation_and_wait(
        retrieval_profile=args.retrieval_profile,
        model=args.model,
        limit=args.limit
    ))

if __name__ == '__main__':
    main() 