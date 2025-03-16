"""
Test script to verify the end-to-end flow of the chat router.
This script mocks external services (OpenAI, Pinecone, Firestore) to test the complete flow.
"""

import os
import asyncio
import json
import sys
from unittest.mock import patch, MagicMock, AsyncMock

# Set environment variables for testing
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["PINECONE_API_KEY"] = "test-key"

# Import app after setting environment variables
from fastapi.testclient import TestClient
from app.main import app
from app.models.dto import ChatRequestDTO, RetrievalRequestDTO, LLMRequestDTO

# Create test client
client = TestClient(app)

# Mock data
mock_context_docs = [
    {
        "id": "doc1",
        "text": "This is a test document about financial markets.",
        "metadata": {"source": "test", "category": "finance"}
    }
]

mock_llm_response = {
    "answer": "This is a test answer about financial markets.",
    "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    "processing_time": 0.5
}

# Test function
async def test_chat_flow():
    print("\n=== Testing Chat Flow ===")
    
    # Create test data
    user_id = "test-user"
    question = "What are financial markets?"
    
    # Create mocks
    mock_get_relevant_context = MagicMock(return_value=mock_context_docs)
    mock_generate_answer = AsyncMock(return_value=mock_llm_response)
    mock_add_document = AsyncMock(return_value="mock-doc-id")
    mock_get_documents = AsyncMock(return_value=[])
    
    # Apply mocks - we need to patch at the module level where the functions are imported
    patches = [
        patch("app.services.chat_service.get_relevant_context", mock_get_relevant_context),
        patch("app.services.chat_service.generate_answer", mock_generate_answer),
        patch("app.services.chat_service.add_document", mock_add_document),
        patch("app.services.chat_service.get_documents", mock_get_documents)
    ]
    
    # Start all patches
    for p in patches:
        p.start()
    
    try:
        # Call the API endpoint
        print("\n--- Testing API Endpoint ---")
        api_response = client.post(
            f"/users/{user_id}/chat/",
            json={"question": question, "model": "gpt-4"}
        )
        print(f"API Status Code: {api_response.status_code}")
        print(f"API Response: {json.dumps(api_response.json(), indent=2)}")
        
        # Verify mocks were called
        print("\n--- Verifying Mock Calls ---")
        print(f"get_relevant_context called: {mock_get_relevant_context.called}")
        print(f"generate_answer called: {mock_generate_answer.called}")
        print(f"add_document called: {mock_add_document.called}")
        print(f"get_documents called: {mock_get_documents.called}")
        
        # Verify add_document was called twice (user message and bot response)
        print(f"add_document call count: {mock_add_document.call_count}")
        
        # Verify the response structure
        response_data = api_response.json()
        print("\n--- Verifying Response Structure ---")
        print(f"Response contains question: {'question' in response_data}")
        print(f"Response contains answer: {'answer' in response_data}")
        print(f"Response contains model: {'model' in response_data}")
        print(f"Response contains conversation_id: {'conversation_id' in response_data}")
        print(f"Response contains context: {'context' in response_data}")
        print(f"Response contains token_usage: {'token_usage' in response_data}")
        print(f"Response contains processing_time: {'processing_time' in response_data}")
    
    finally:
        # Stop all patches
        for p in patches:
            p.stop()
    
    print("\n=== Test Completed ===")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_chat_flow()) 