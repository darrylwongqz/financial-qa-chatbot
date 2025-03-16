# app/routers/chat.py
from fastapi import APIRouter, HTTPException, Path, Body, Query
from typing import Optional, List, Dict, Any
from app.models.response_dto import ChatMessage, ChatResponse, DeleteResponse, RetrievalProfilesResponse
from app.services.chat_service import (
    add_chat_message, 
    get_chat_history, 
    clear_chat_history, 
)
from app.config import RETRIEVAL_PROFILES, DEFAULT_RETRIEVAL_PROFILE
from app.utils.logging_utils import logger, log_endpoint
from app.models.dto import ChatRequestDTO


router = APIRouter(
    prefix="/users/{user_id}/chat",
    tags=["chat"]
)

@router.post("/", 
             summary="Post a chat message", 
             description="""Send a chat message and receive a response using the RAG pipeline.
             
             The response includes relevant context from the knowledge base and provides accurate answers to financial questions.
             For questions involving calculations, the LLM will show its work and provide step-by-step explanations.""",
             response_model=ChatResponse,
             response_description="The generated answer along with context and metadata")
@log_endpoint
async def post_chat_message(
    user_id: str = Path(..., description="User's unique identifier"),
    chat_request: ChatRequestDTO = Body(..., description="Chat request parameters", 
                                      example={
                                          "question": "What was the percentage change in the net cash from operating activities from 2008-2009?",
                                          "model": "gpt-4",
                                          "temperature": 0.7,
                                          "max_tokens": 1000,
                                          "retrieval_profile": "balanced"
                                      })
) -> Dict[str, Any]:
    """
    Process a new chat message for the user:
    1. Retrieve relevant context using the retrieval pipeline
    2. Generate an answer using the LLM service
    3. Store the conversation in the database
    
    Returns the answer along with the retrieved context and metadata.
    """
    try:
        # Set the user_id in the DTO
        chat_request.user_id = user_id
        
        # Process the chat message
        response = await add_chat_message(chat_request)
        return response
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", 
            summary="Get chat history", 
            description="Retrieve the chat history for a user with optional filtering and pagination.",
            response_model=List[ChatMessage],
            response_description="List of chat messages in chronological order")
@log_endpoint
async def get_chat_messages(
    user_id: str = Path(..., description="User's unique identifier"),
    conversation_id: Optional[str] = Query(None, description="Filter by conversation ID"),
    limit: Optional[int] = Query(None, description="Maximum number of messages to return"),
    offset: int = Query(0, description="Number of messages to skip")
) -> List[Dict[str, Any]]:
    """
    Return the chat history for the specified user.
    Optionally filter by conversation_id and apply pagination.
    
    The messages are returned in chronological order (oldest first).
    """
    try:
        logger.info(f"Retrieving chat history for user {user_id}")
        history = await get_chat_history(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=limit,
            offset=offset
        )
        return history
    except Exception as e:
        logger.error(f"Error in get_chat_messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profiles", 
            summary="Get retrieval profiles", 
            description="Get available retrieval profiles for the chat.",
            response_model=RetrievalProfilesResponse,
            response_description="Available retrieval profiles and the default profile")
@log_endpoint
async def get_retrieval_profiles() -> Dict[str, Any]:
    """
    Return the available retrieval profiles that can be used for chat.
    These profiles control the balance between speed and accuracy of the retrieval system.
    
    - Fast: Optimized for speed with improved accuracy (< 0.2s response time)
    - Balanced: Balanced performance and quality (< 1s response time)
    - Accurate: Optimized for accuracy (< 2s response time)
    """
    try:
        # Return the profiles with their configurations
        return {
            "profiles": {
                key: {
                    "name": profile["name"],
                    "description": profile["description"]
                } for key, profile in RETRIEVAL_PROFILES.items()
            },
            "default_profile": DEFAULT_RETRIEVAL_PROFILE
        }
    except Exception as e:
        logger.error(f"Error getting retrieval profiles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/", 
               summary="Clear chat history", 
               description="Clear all chat history for a user (this cannot be undone).",
               response_model=DeleteResponse,
               response_description="Confirmation message")
@log_endpoint
async def delete_chat_messages(
    user_id: str = Path(..., description="User's unique identifier"),
    conversation_id: Optional[str] = Query(None, description="Specific conversation to delete")
) -> Dict[str, str]:
    """
    Delete chat messages for the specified user.
    If conversation_id is provided, only delete messages from that conversation.
    Otherwise, all messages for the user will be deleted.
    
    This operation cannot be undone.
    """
    try:
        if conversation_id:
            logger.info(f"Clearing conversation {conversation_id} for user {user_id}")
            await clear_chat_history(user_id, conversation_id)
            return {"message": f"Conversation {conversation_id} cleared successfully."}
        else:
            logger.info(f"Clearing all chat history for user {user_id}")
            await clear_chat_history(user_id)
            return {"message": "All chat history cleared successfully. Note: The chatbot will not remember previous context."}
    except Exception as e:
        logger.error(f"Error in delete_chat_messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
