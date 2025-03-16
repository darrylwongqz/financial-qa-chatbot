"""
Real-world test script to measure and compare the response times of different retrieval profiles.
This script tests the actual performance of the Fast, Balanced, and Accurate retrieval profiles
using the real retrieval service with actual data.
"""

import os
import time
import asyncio
import statistics
from typing import List, Dict, Any, Optional
import json

# Import after setting environment variables
from app.config import RETRIEVAL_PROFILES, DEFAULT_RETRIEVAL_PROFILE
from app.services.retrieval_service import get_relevant_context_with_profile
from app.db.pinecone_db import initialize_pinecone
from app.utils.logging_utils import logger

# Test queries representing different types of financial questions
TEST_QUERIES = [
    "What is the difference between stocks and bonds?",
    "How do interest rates affect inflation?",
    "What are the key financial ratios for company valuation?",
    "Explain the concept of market capitalization.",
    "What is the price-to-earnings ratio?"
]

async def test_real_retrieval_profile_performance():
    """Test the performance of different retrieval profiles with real data"""
    # Dictionary to store results
    results = {}
    
    # Initialize Pinecone
    logger.info("Initializing Pinecone...")
    try:
        initialize_pinecone()
        logger.info("Pinecone initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        logger.warning("Continuing with test, but results may be affected")
    
    # Test each profile
    for profile_name, profile_data in RETRIEVAL_PROFILES.items():
        logger.info(f"\nTesting {profile_name} profile ({profile_data['name']})...")
        
        # List to store response times
        response_times = []
        doc_counts = []
        
        # Run multiple queries
        for query in TEST_QUERIES:
            try:
                # Measure response time
                start_time = time.time()
                docs = get_relevant_context_with_profile(
                    query=query,
                    profile=profile_name
                )
                end_time = time.time()
                
                # Calculate response time
                response_time = end_time - start_time
                response_times.append(response_time)
                doc_counts.append(len(docs))
                
                logger.info(f"  Query: '{query}'")
                logger.info(f"  Response time: {response_time:.4f}s")
                logger.info(f"  Retrieved {len(docs)} documents")
                
                # Print first document snippet for verification
                if docs:
                    doc_text = docs[0].get("text", "")[:100] + "..." if len(docs[0].get("text", "")) > 100 else docs[0].get("text", "")
                    logger.info(f"  First doc: {doc_text}")
            except Exception as e:
                logger.error(f"Error processing query '{query}' with profile '{profile_name}': {e}")
        
        if response_times:
            # Calculate statistics
            avg_time = statistics.mean(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            avg_docs = statistics.mean(doc_counts) if doc_counts else 0
            
            # Store results
            results[profile_name] = {
                "name": profile_data["name"],
                "description": profile_data["description"],
                "config": profile_data["config"],
                "avg_response_time": avg_time,
                "min_response_time": min_time,
                "max_response_time": max_time,
                "avg_docs_retrieved": avg_docs,
                "response_times": response_times
            }
    
    # Print summary
    logger.info("\n=== Performance Summary ===")
    for profile_name, result in results.items():
        logger.info(f"\n{result['name']} Profile:")
        logger.info(f"  Description: {result['description']}")
        logger.info(f"  Average response time: {result['avg_response_time']:.4f}s")
        logger.info(f"  Min response time: {result['min_response_time']:.4f}s")
        logger.info(f"  Max response time: {result['max_response_time']:.4f}s")
        logger.info(f"  Average docs retrieved: {result['avg_docs_retrieved']:.1f}")
    
    # Compare profiles
    if len(results) > 1:
        logger.info("\n=== Profile Comparison ===")
        fastest = min(results.items(), key=lambda x: x[1]["avg_response_time"])
        slowest = max(results.items(), key=lambda x: x[1]["avg_response_time"])
        
        logger.info(f"Fastest profile: {fastest[1]['name']} ({fastest[1]['avg_response_time']:.4f}s)")
        logger.info(f"Slowest profile: {slowest[1]['name']} ({slowest[1]['avg_response_time']:.4f}s)")
        logger.info(f"Speed difference: {slowest[1]['avg_response_time'] / fastest[1]['avg_response_time']:.2f}x")
        
        # Check if profiles match their descriptions
        if "fast" in results and "balanced" in results:
            if results["fast"]["avg_response_time"] < results["balanced"]["avg_response_time"]:
                logger.info("✅ Fast profile is faster than Balanced")
            else:
                logger.warning("❌ Fast profile is NOT faster than Balanced")
                
        if "balanced" in results and "accurate" in results:
            if results["balanced"]["avg_response_time"] < results["accurate"]["avg_response_time"]:
                logger.info("✅ Balanced profile is faster than Accurate")
            else:
                logger.warning("❌ Balanced profile is NOT faster than Accurate")
    
    # Return results for further analysis
    return results

# Run the test if executed directly
if __name__ == "__main__":
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run the test
    asyncio.run(test_real_retrieval_profile_performance()) 