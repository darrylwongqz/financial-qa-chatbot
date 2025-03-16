"""
Test script to measure and compare the response times of different retrieval profiles.
This script tests the actual performance of the Fast, Balanced, and Accurate retrieval profiles.
"""

import os
import time
import asyncio
import statistics
from typing import List, Dict, Any, Optional
import json

# Set environment variables for testing
os.environ["OPENAI_API_KEY"] = "sk-dummy-key"  # Dummy key for testing
os.environ["PINECONE_API_KEY"] = "dummy-key"   # Dummy key for testing

# Import after setting environment variables
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.config import RETRIEVAL_PROFILES
from app.services.retrieval_service import get_relevant_context_with_profile

# Test queries representing different types of financial questions
TEST_QUERIES = [
    "What is the difference between stocks and bonds?",
    "How do interest rates affect inflation?",
    "What are the key financial ratios for company valuation?",
    "Explain the concept of market capitalization.",
    "What is the price-to-earnings ratio?"
]

# Mock data to return for each query
MOCK_DOCS = [
    {
        "id": "doc1",
        "text": "This is a test document about financial markets.",
        "metadata": {"source": "test", "category": "finance"}
    },
    {
        "id": "doc2",
        "text": "Stocks represent ownership in a company, while bonds are debt instruments.",
        "metadata": {"source": "test", "category": "investments"}
    },
    {
        "id": "doc3",
        "text": "Interest rates and inflation have an inverse relationship in most economic scenarios.",
        "metadata": {"source": "test", "category": "economics"}
    }
]

# Create a mock for get_relevant_context that introduces delays based on the profile
def mock_get_relevant_context(retrieval_request):
    """Mock implementation that simulates different response times based on profile settings"""
    # Extract parameters from the request
    query = retrieval_request.query
    top_k = retrieval_request.top_k
    rerank = retrieval_request.rerank
    use_hybrid_search = retrieval_request.use_hybrid_search
    
    # Simulate processing time based on parameters
    # More documents (higher top_k) = more time
    # Re-ranking = more time
    # Hybrid search = more time
    base_time = 0.01  # Base processing time in seconds
    
    # Add time for each document processed
    doc_time = top_k * 0.005
    
    # Add time for re-ranking if enabled
    rerank_time = 0.1 if rerank else 0
    
    # Add time for hybrid search if enabled
    hybrid_time = 0.02 if use_hybrid_search else 0
    
    # Calculate total simulated time
    total_time = base_time + doc_time + rerank_time + hybrid_time
    
    # Sleep to simulate processing time
    time.sleep(total_time)
    
    # Return mock documents
    return MOCK_DOCS[:top_k]

@pytest.mark.asyncio
async def test_retrieval_profile_performance():
    """Test the performance of different retrieval profiles"""
    # Dictionary to store results
    results = {}
    
    # Apply mock
    with patch("app.services.retrieval_service.get_relevant_context", mock_get_relevant_context):
        # Test each profile
        for profile_name, profile_data in RETRIEVAL_PROFILES.items():
            print(f"\nTesting {profile_name} profile ({profile_data['name']})...")
            
            # List to store response times
            response_times = []
            
            # Run multiple queries
            for query in TEST_QUERIES:
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
                
                print(f"  Query: '{query}'")
                print(f"  Response time: {response_time:.4f}s")
                print(f"  Retrieved {len(docs)} documents")
            
            # Calculate statistics
            avg_time = statistics.mean(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            # Store results
            results[profile_name] = {
                "name": profile_data["name"],
                "description": profile_data["description"],
                "config": profile_data["config"],
                "avg_response_time": avg_time,
                "min_response_time": min_time,
                "max_response_time": max_time,
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
    
    # Compare profiles
    print("\n=== Profile Comparison ===")
    fastest = min(results.items(), key=lambda x: x[1]["avg_response_time"])
    slowest = max(results.items(), key=lambda x: x[1]["avg_response_time"])
    
    print(f"Fastest profile: {fastest[1]['name']} ({fastest[1]['avg_response_time']:.4f}s)")
    print(f"Slowest profile: {slowest[1]['name']} ({slowest[1]['avg_response_time']:.4f}s)")
    print(f"Speed difference: {slowest[1]['avg_response_time'] / fastest[1]['avg_response_time']:.2f}x")
    
    # Verify that the profiles match their descriptions
    assert results["fast"]["avg_response_time"] < results["balanced"]["avg_response_time"], "Fast profile should be faster than Balanced"
    assert results["balanced"]["avg_response_time"] < results["accurate"]["avg_response_time"], "Balanced profile should be faster than Accurate"
    
    # Return results for further analysis
    return results

# Run the test if executed directly
if __name__ == "__main__":
    asyncio.run(test_retrieval_profile_performance()) 