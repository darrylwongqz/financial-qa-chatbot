"""
Comprehensive test script to measure and compare the response times of different retrieval profiles.
This script tests the performance of the Fast, Balanced, and Accurate retrieval profiles
with a larger set of queries and more realistic settings.
"""

import os
import time
import asyncio
import statistics
from typing import List, Dict, Any, Optional
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import after setting environment variables
from app.config import RETRIEVAL_PROFILES, DEFAULT_RETRIEVAL_PROFILE
from app.services.retrieval_service import get_relevant_context_with_profile
from app.db.pinecone_db import initialize_pinecone
from app.utils.logging_utils import logger

# A larger set of test queries representing different types of financial questions
TEST_QUERIES = [
    # Basic financial concepts
    "What is the difference between stocks and bonds?",
    "How do interest rates affect inflation?",
    "What are the key financial ratios for company valuation?",
    "Explain the concept of market capitalization.",
    "What is the price-to-earnings ratio?",
    
    # Investment strategies
    "What is dollar-cost averaging?",
    "How does diversification reduce investment risk?",
    "What is the difference between active and passive investing?",
    "Explain the concept of asset allocation.",
    "What are ETFs and how do they work?",
    
    # Economic concepts
    "What is GDP and how is it calculated?",
    "Explain the relationship between unemployment and inflation.",
    "What causes economic recessions?",
    "How do central banks influence the economy?",
    "What is fiscal policy versus monetary policy?",
    
    # Personal finance
    "How should I prioritize paying off debt?",
    "What is the difference between a traditional IRA and a Roth IRA?",
    "How do I calculate my net worth?",
    "What is a good debt-to-income ratio?",
    "How much should I save for retirement?"
]

async def test_comprehensive_retrieval_profile_performance():
    """Test the performance of different retrieval profiles with a comprehensive set of queries"""
    # Dictionary to store results
    results = {}
    all_query_results = []
    
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
                
                # Store individual query result
                all_query_results.append({
                    "profile": profile_name,
                    "profile_name": profile_data["name"],
                    "query": query,
                    "response_time": response_time,
                    "doc_count": len(docs)
                })
                
                # Print first document snippet for verification if available
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
            median_time = statistics.median(response_times)
            stdev_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
            avg_docs = statistics.mean(doc_counts) if doc_counts else 0
            
            # Store results
            results[profile_name] = {
                "name": profile_data["name"],
                "description": profile_data["description"],
                "config": profile_data["config"],
                "avg_response_time": avg_time,
                "median_response_time": median_time,
                "min_response_time": min_time,
                "max_response_time": max_time,
                "stdev_response_time": stdev_time,
                "avg_docs_retrieved": avg_docs,
                "response_times": response_times
            }
    
    # Print summary
    logger.info("\n=== Performance Summary ===")
    for profile_name, result in results.items():
        logger.info(f"\n{result['name']} Profile:")
        logger.info(f"  Description: {result['description']}")
        logger.info(f"  Average response time: {result['avg_response_time']:.4f}s")
        logger.info(f"  Median response time: {result['median_response_time']:.4f}s")
        logger.info(f"  Min response time: {result['min_response_time']:.4f}s")
        logger.info(f"  Max response time: {result['max_response_time']:.4f}s")
        logger.info(f"  Standard deviation: {result['stdev_response_time']:.4f}s")
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
    
    # Create visualizations
    try:
        # Convert results to DataFrame for easier visualization
        df = pd.DataFrame(all_query_results)
        
        # Create output directory if it doesn't exist
        os.makedirs("app/tests/results", exist_ok=True)
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"app/tests/results/retrieval_profile_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
        
        # Create boxplot of response times by profile
        plt.figure(figsize=(10, 6))
        boxplot = df.boxplot(column='response_time', by='profile_name', grid=False)
        plt.title('Response Time by Retrieval Profile')
        plt.suptitle('')  # Remove default title
        plt.ylabel('Response Time (seconds)')
        plt.xlabel('Retrieval Profile')
        
        # Save plot
        plot_path = f"app/tests/results/retrieval_profile_boxplot_{timestamp}.png"
        plt.savefig(plot_path)
        logger.info(f"Boxplot saved to {plot_path}")
        
        # Create bar chart of average response times
        avg_times = df.groupby('profile_name')['response_time'].mean().reset_index()
        plt.figure(figsize=(10, 6))
        bars = plt.bar(avg_times['profile_name'], avg_times['response_time'])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}s', ha='center', va='bottom')
        
        plt.title('Average Response Time by Retrieval Profile')
        plt.ylabel('Average Response Time (seconds)')
        plt.xlabel('Retrieval Profile')
        
        # Save plot
        bar_path = f"app/tests/results/retrieval_profile_barchart_{timestamp}.png"
        plt.savefig(bar_path)
        logger.info(f"Bar chart saved to {bar_path}")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
    
    # Return results for further analysis
    return results

# Run the test if executed directly
if __name__ == "__main__":
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run the test
    asyncio.run(test_comprehensive_retrieval_profile_performance()) 