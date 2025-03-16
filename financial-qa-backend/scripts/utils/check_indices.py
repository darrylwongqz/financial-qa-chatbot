#!/usr/bin/env python
"""
Script to check the status of both dense and sparse indices in Pinecone.
This helps verify that both indices are properly populated and accessible.
"""

import os
import sys
import json
import pprint
from pathlib import Path
from dotenv import load_dotenv
import pinecone

# Add the project root to the Python path if running from scripts directory
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables
dotenv_path = project_root / "financial-qa-backend" / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Check for Pinecone API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    print("Error: PINECONE_API_KEY not found in environment variables")
    sys.exit(1)

# Index configuration
DENSE_INDEX_NAME = "financial-qa-chatbot"
SPARSE_INDEX_NAME = "financial-qa-sparse-3"
SPARSE_NAMESPACE = "train_sparse_20250316"

def check_indices():
    """Check the status of both dense and sparse indices in Pinecone."""
    print("=" * 80)
    print(f"Checking Pinecone indices status")
    print("=" * 80)
    
    # Initialize Pinecone
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    
    # List all indices
    indices = pc.list_indexes()
    print(f"Available indices: {indices}")
    
    # Check dense index
    print("\n" + "=" * 40)
    print(f"DENSE INDEX: {DENSE_INDEX_NAME}")
    print("=" * 40)
    
    try:
        dense_index = pc.Index(DENSE_INDEX_NAME)
        stats = dense_index.describe_index_stats()
        print("Dense index stats:")
        pprint.pprint(stats)
        
        # Check namespaces
        namespaces = stats.get("namespaces", {})
        print("\nNamespaces in dense index:")
        if not namespaces:
            print("No namespaces found in dense index")
        else:
            for ns, ns_stats in namespaces.items():
                print(f"  - {ns}: {ns_stats.get('vector_count', 0)} vectors")
        
        # Try a simple query
        print("\nTrying a simple query to the dense index...")
        try:
            # Create a simple vector for querying (1536 dimensions for OpenAI embeddings)
            dummy_vector = [0.0] * 1536
            
            query_result = dense_index.query(
                vector=dummy_vector,
                top_k=1,
                include_metadata=True
            )
            print(f"Query result: {query_result}")
        except Exception as e:
            print(f"Error with dense query: {str(e)}")
    
    except Exception as e:
        print(f"Error connecting to dense index: {str(e)}")
    
    # Check sparse index
    print("\n" + "=" * 40)
    print(f"SPARSE INDEX: {SPARSE_INDEX_NAME}")
    print("=" * 40)
    
    try:
        sparse_index = pc.Index(SPARSE_INDEX_NAME)
        stats = sparse_index.describe_index_stats()
        print("Sparse index stats:")
        pprint.pprint(stats)
        
        # Check namespaces
        namespaces = stats.get("namespaces", {})
        print("\nNamespaces in sparse index:")
        if not namespaces:
            print("No namespaces found in sparse index")
        else:
            for ns, ns_stats in namespaces.items():
                print(f"  - {ns}: {ns_stats.get('vector_count', 0)} vectors")
        
        # Check specific namespace
        print(f"\nChecking specific namespace: {SPARSE_NAMESPACE}")
        namespace_stats = namespaces.get(SPARSE_NAMESPACE, {})
        if namespace_stats:
            print(f"Found {namespace_stats.get('vector_count', 0)} vectors in namespace '{SPARSE_NAMESPACE}'")
        else:
            print(f"Namespace '{SPARSE_NAMESPACE}' not found or empty")
        
        # Try a simple query
        print("\nTrying a simple query to the sparse index...")
        try:
            # Create a simple sparse vector for querying
            sparse_vector = {
                "indices": [1, 2, 3],
                "values": [0.5, 0.3, 0.2]
            }
            
            query_result = sparse_index.query(
                sparse_vector=sparse_vector,
                namespace=SPARSE_NAMESPACE,
                top_k=1,
                include_metadata=True
            )
            print(f"Query result: {query_result}")
        except Exception as e:
            print(f"Error with sparse query: {str(e)}")
    
    except Exception as e:
        print(f"Error connecting to sparse index: {str(e)}")

if __name__ == "__main__":
    check_indices() 