"""
Tests for the chat service functionality.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock
import uuid

from app.models.dto import ChatRequestDTO, LLMRequestDTO
from app.services.chat_service import (
    add_chat_message,
    get_recent_conversation_history,
    get_chat_history,
    clear_chat_history
)
from app.config import DEFAULT_MODEL, DEFAULT_RETRIEVAL_PROFILE, DENSE_NAMESPACE

@pytest.fixture
def mock_firestore():
    """Mock Firestore operations."""
    with patch('app.services.chat_service.add_document') as mock_add, \
         patch('app.services.chat_service.get_documents') as mock_get, \
         patch('app.services.chat_service.delete_document') as mock_del, \
         patch('app.services.chat_service.delete_documents') as mock_del_all:
        mock_add.return_value = "test_doc_id"
        mock_get.return_value = []
        yield {
            'add': mock_add,
            'get': mock_get,
            'delete': mock_del,
            'delete_all': mock_del_all
        }

@pytest.fixture
def mock_services():
    """Mock various services used in chat flow."""
    with patch('app.services.chat_service.QuestionAnalysisService') as mock_qa, \
         patch('app.services.chat_service.NumericalReasoningService') as mock_nr, \
         patch('app.services.chat_service.get_relevant_context_with_profile') as mock_retrieve, \
         patch('app.services.chat_service.generate_answer') as mock_llm:
        
        # Setup QuestionAnalysisService mock
        mock_qa_instance = Mock()
        mock_qa_instance.analyze_question.return_value = Mock(
            requires_calculation=False,
            question_type="GENERAL",
            calculation_type=None
        )
        mock_qa.return_value = mock_qa_instance
        
        # Setup NumericalReasoningService mock
        mock_nr_instance = Mock()
        mock_nr_instance.process_question = AsyncMock(
            return_value=("42", "Step 1: Calculate\nStep 2: Verify")
        )
        mock_nr.return_value = mock_nr_instance
        
        # Setup context retrieval mock
        mock_retrieve.return_value = [
            {
                "text": "Sample context",
                "metadata": {"source": "test"},
                "score": 0.95
            }
        ]
        
        # Setup LLM mock
        mock_llm.return_value = {
            "answer": "Test answer",
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            },
            "processing_time": 0.5
        }
        
        yield {
            'question_analysis': mock_qa_instance,
            'numerical_reasoning': mock_nr_instance,
            'retrieve_context': mock_retrieve,
            'llm': mock_llm
        }

@pytest.mark.asyncio
async def test_basic_chat_flow(mock_firestore, mock_services):
    """Test basic chat flow without numerical reasoning."""
    # Prepare test request
    chat_request = ChatRequestDTO(
        question="What is compound interest?",
        user_id="test_user",
        enable_numerical_reasoning=False
    )
    
    # Execute chat flow
    response = await add_chat_message(chat_request)
    
    # Verify question analysis was called
    mock_services['question_analysis'].analyze_question.assert_called_once_with(
        "What is compound interest?"
    )
    
    # Verify context retrieval
    mock_services['retrieve_context'].assert_called_once()
    
    # Verify LLM generation
    mock_services['llm'].assert_called_once()
    
    # Verify Firestore operations
    assert mock_firestore['add'].call_count == 2  # User message and bot response
    
    # Verify response structure
    assert response["question"] == "What is compound interest?"
    assert response["answer"] == "Test answer"
    assert response["model"] == DEFAULT_MODEL
    assert "conversation_id" in response
    assert response["numerical_analysis"] is None

@pytest.mark.asyncio
async def test_numerical_reasoning_flow(mock_firestore, mock_services):
    """Test chat flow with numerical reasoning enabled."""
    # Setup question analysis to require calculation
    mock_services['question_analysis'].analyze_question.return_value = Mock(
        requires_calculation=True,
        question_type="FINANCIAL_CALCULATION",
        calculation_type="COMPOUND_INTEREST"
    )
    
    # Prepare test request
    chat_request = ChatRequestDTO(
        question="Calculate compound interest for $1000 at 5% for 3 years",
        user_id="test_user",
        enable_numerical_reasoning=True,
        calculation_precision=6,
        show_calculation_steps=True
    )
    
    # Execute chat flow
    response = await add_chat_message(chat_request)
    
    # Verify numerical reasoning was called
    mock_services['numerical_reasoning'].process_question.assert_called_once()
    
    # Verify calculation results in response
    assert response["numerical_analysis"]["performed_calculation"] is True
    assert response["numerical_analysis"]["results"] == "42"
    assert response["numerical_analysis"]["steps"] == "Step 1: Calculate\nStep 2: Verify"

@pytest.mark.asyncio
async def test_conversation_history(mock_firestore):
    """Test conversation history retrieval and management."""
    # Setup mock messages
    mock_messages = [
        {
            "sender": "user",
            "message": "Question 1",
            "conversation_id": "conv1",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        {
            "sender": "bot",
            "message": "Answer 1",
            "conversation_id": "conv1",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    ]
    mock_firestore['get'].return_value = mock_messages
    
    # Test history retrieval
    history = await get_recent_conversation_history(
        user_id="test_user",
        conversation_id="conv1",
        limit=5
    )
    assert len(history) == 2
    # Messages should be in reverse chronological order (newest first)
    assert history[0]["message"] == "Answer 1"  # Bot's message is newer
    assert history[1]["message"] == "Question 1"  # User's message is older

@pytest.mark.asyncio
async def test_error_handling(mock_firestore, mock_services):
    """Test error handling in chat flow."""
    # Setup numerical reasoning to raise an exception
    mock_services['numerical_reasoning'].process_question.side_effect = Exception("Calculation error")
    
    # Prepare test request
    chat_request = ChatRequestDTO(
        question="Calculate something",
        user_id="test_user",
        enable_numerical_reasoning=True
    )
    
    # Execute chat flow - should continue despite calculation error
    response = await add_chat_message(chat_request)
    
    # Verify response still generated
    assert response["answer"] == "Test answer"
    assert "numerical_analysis" in response

@pytest.mark.asyncio
async def test_clear_chat_history(mock_firestore):
    """Test chat history clearing functionality."""
    # Setup mock messages
    mock_messages = [
        {"id": "msg1", "conversation_id": "conv1"},
        {"id": "msg2", "conversation_id": "conv1"},
        {"id": "msg3", "conversation_id": "conv2"}
    ]
    mock_firestore['get'].return_value = mock_messages
    
    # Test clearing specific conversation
    await clear_chat_history("test_user", "conv1")
    assert mock_firestore['delete'].call_count == 2
    
    # Test clearing all conversations
    await clear_chat_history("test_user")
    mock_firestore['delete_all'].assert_called_once() 