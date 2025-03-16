"""
Test script to measure and compare the response times of different retrieval profiles
using mocks to simulate the actual behavior of the retrieval service.
"""

import os
import time
import asyncio
import statistics
from typing import List, Dict, Any, Optional
import json
from unittest.mock import patch, MagicMock

# Import after setting environment variables
from app.config import RETRIEVAL_PROFILES, DEFAULT_RETRIEVAL_PROFILE
from app.models.dto import RetrievalRequestDTO
from app.utils.logging_utils import logger

# Test queries representing different types of financial questions
TEST_QUERIES = [
    "What is the difference between stocks and bonds?",
    "How do interest rates affect inflation?",
    "What are the key financial ratios for company valuation?",
    "Explain the concept of market capitalization.",
    "What is the price-to-earnings ratio?"
]

# Mock documents to return
MOCK_DOCS = [
    {
        "id": "doc1",
        "text": "Stocks represent ownership in a company, while bonds are debt instruments issued by companies or governments.",
        "metadata": {"source": "finance_101.txt", "category": "investments"}
    },
    {
        "id": "doc2",
        "text": "Interest rates and inflation have an inverse relationship in most economic scenarios.",
        "metadata": {"source": "economics_basics.txt", "category": "economics"}
    },
    {
        "id": "doc3",
        "text": "Key financial ratios include P/E ratio, EPS, ROI, and debt-to-equity ratio.",
        "metadata": {"source": "valuation_guide.txt", "category": "analysis"}
    },
    {
        "id": "doc4",
        "text": "Market capitalization is calculated by multiplying a company's share price by its total number of outstanding shares.",
        "metadata": {"source": "stock_market.txt", "category": "equities"}
    },
    {
        "id": "doc5",
        "text": "The price-to-earnings ratio (P/E ratio) is a valuation metric that measures a company's current share price relative to its earnings per share (EPS).",
        "metadata": {"source": "valuation_metrics.txt", "category": "analysis"}
    }
]

def get_relevant_context_with_profile_mock(query: str, profile: str = DEFAULT_RETRIEVAL_PROFILE, **kwargs):
    """Mock implementation of get_relevant_context_with_profile"""
    # Get profile configuration
    profile_config = RETRIEVAL_PROFILES[profile]["config"]
    
    # Extract parameters from the profile
    top_k = profile_config["top_k"]
    rerank = profile_config["rerank"]
    use_hybrid_search = profile_config["use_hybrid_search"]
    
    # Simulate processing time based on parameters
    # More documents (higher top_k) = more time
    # Re-ranking = more time
    # Hybrid search = more time
    base_time = 0.05  # Base processing time in seconds
    
    # Add time for each document processed
    doc_time = top_k * 0.01
    
    # Add time for re-ranking if enabled
    rerank_time = 0.1 if rerank else 0
    
    # Add time for hybrid search if enabled
    hybrid_time = 0.05 if use_hybrid_search else 0
    
    # Calculate total simulated time
    total_time = base_time + doc_time + rerank_time + hybrid_time
    
    # Sleep to simulate processing time
    time.sleep(total_time)
    
    # Return mock documents
    return MOCK_DOCS[:top_k]

async def test_retrieval_profile_performance_with_mocks():
    """Test the performance of different retrieval profiles using mocks"""
    # Dictionary to store results
    results = {}
    
    # Test each profile
    for profile_name, profile_data in RETRIEVAL_PROFILES.items():
        print(f"\nTesting {profile_name} profile ({profile_data['name']})...")
        
        # List to store response times
        response_times = []
        doc_counts = []
        
        # Run multiple queries
        for query in TEST_QUERIES:
            try:
                # Measure response time
                start_time = time.time()
                docs = get_relevant_context_with_profile_mock(
                    query=query,
                    profile=profile_name
                )
                end_time = time.time()
                
                # Calculate response time
                response_time = end_time - start_time
                response_times.append(response_time)
                doc_counts.append(len(docs))
                
                print(f"  Query: '{query}'")
                print(f"  Response time: {response_time:.4f}s")
                print(f"  Retrieved {len(docs)} documents")
                
                # Print first document snippet for verification
                if docs:
                    doc_text = docs[0].get("text", "")[:50] + "..." if len(docs[0].get("text", "")) > 50 else docs[0].get("text", "")
                    print(f"  First doc: {doc_text}")
            except Exception as e:
                print(f"Error processing query '{query}' with profile '{profile_name}': {e}")
        
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
    print("\n=== Performance Summary ===")
    for profile_name, result in results.items():
        print(f"\n{result['name']} Profile:")
        print(f"  Description: {result['description']}")
        print(f"  Average response time: {result['avg_response_time']:.4f}s")
        print(f"  Min response time: {result['min_response_time']:.4f}s")
        print(f"  Max response time: {result['max_response_time']:.4f}s")
        print(f"  Average docs retrieved: {result['avg_docs_retrieved']:.1f}")
    
    # Compare profiles
    if len(results) > 1:
        print("\n=== Profile Comparison ===")
        fastest = min(results.items(), key=lambda x: x[1]["avg_response_time"])
        slowest = max(results.items(), key=lambda x: x[1]["avg_response_time"])
        
        print(f"Fastest profile: {fastest[1]['name']} ({fastest[1]['avg_response_time']:.4f}s)")
        print(f"Slowest profile: {slowest[1]['name']} ({slowest[1]['avg_response_time']:.4f}s)")
        print(f"Speed difference: {slowest[1]['avg_response_time'] / fastest[1]['avg_response_time']:.2f}x")
        
        # Check if profiles match their descriptions
        if "fast" in results and "balanced" in results:
            if results["fast"]["avg_response_time"] < results["balanced"]["avg_response_time"]:
                print("✅ Fast profile is faster than Balanced")
            else:
                print("❌ Fast profile is NOT faster than Balanced")
                
        if "balanced" in results and "accurate" in results:
            if results["balanced"]["avg_response_time"] < results["accurate"]["avg_response_time"]:
                print("✅ Balanced profile is faster than Accurate")
            else:
                print("❌ Balanced profile is NOT faster than Accurate")
    
    # Return results for further analysis
    return results

# Run the test if executed directly
if __name__ == "__main__":
    asyncio.run(test_retrieval_profile_performance_with_mocks()) 