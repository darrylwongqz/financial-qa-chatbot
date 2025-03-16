import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Financial QA API"}

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

# Add more tests as you implement more functionality
# def test_chat_endpoint():
#     """Test the chat endpoint."""
#     response = client.post(
#         "/chat/",
#         json={"question": "What is inflation?", "model": "gpt-3.5-turbo"}
#     )
#     assert response.status_code == 200
#     assert "answer" in response.json()
#     assert "question" in response.json()
#     assert response.json()["question"] == "What is inflation?" 