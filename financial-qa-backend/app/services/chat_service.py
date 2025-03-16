# app/services/chat_service.py
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import uuid
from decimal import Decimal, getcontext

from app.config import DEFAULT_MODEL, AVAILABLE_MODELS, DEFAULT_RETRIEVAL_PROFILE, DENSE_NAMESPACE, DEFAULT_MEMORY_LIMIT
from app.db.firestore import add_document, get_documents, delete_documents, delete_document, query_documents
from app.services.retrieval_service import get_relevant_context_with_profile
from app.services.llm_service import generate_answer, validate_model
from app.models.dto import ChatRequestDTO, LLMRequestDTO
from app.utils.logging_utils import logger

def get_user_chat_collection(user_id: str) -> str:
    """Return the Firestore collection path for a user's chat history."""
    return f"users/{user_id}/chat"

async def get_recent_conversation_history(
    user_id: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve recent chat messages for a user.
    Returns messages in chronological order (oldest to newest).
    """
    collection = get_user_chat_collection(user_id)
    limit = limit or DEFAULT_MEMORY_LIMIT
    
    # Query with descending order so the newest messages come first
    messages = await get_documents(collection=collection, limit=limit, descending=True)
    
    # Reverse the list to show messages in chronological order (oldest to newest)
    messages.reverse()
    return messages

async def add_chat_message(chat_request: ChatRequestDTO) -> Dict[str, Any]:
    """Process a chat message and generate a response."""
    # Generate conversation ID if not provided
    if not chat_request.conversation_id:
        chat_request.conversation_id = str(uuid.uuid4())
        logger.info(f"Generated new conversation ID: {chat_request.conversation_id}")
    else:
        logger.info(f"Using provided conversation ID: {chat_request.conversation_id}")
    
    # Set up retrieval parameters
    model = chat_request.model or DEFAULT_MODEL
    retrieval_profile = chat_request.retrieval_profile or DEFAULT_RETRIEVAL_PROFILE
    
    # Retrieve context
    context_docs = await get_relevant_context_with_profile(
        query=chat_request.question,
        profile=retrieval_profile,
        namespace=DENSE_NAMESPACE,
        filter_condition=None
    )
    
    # First save the user's message to Firestore
    user_message = {
        "sender": "user",
        "message": chat_request.question,
        "conversation_id": chat_request.conversation_id,
        "retrieval_profile": retrieval_profile,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    user_msg_id = await add_document(collection=get_user_chat_collection(chat_request.user_id), data=user_message)
    logger.info(f"Saved user message with ID: {user_msg_id}")
    
    # Retrieve conversation history
    conversation_history = []
    recent_messages = await get_recent_conversation_history(
        user_id=chat_request.user_id,
        limit=DEFAULT_MEMORY_LIMIT
    )
    
    if recent_messages:
        conversation_history = [
            {"role": msg["sender"], "content": msg["message"]}
            for msg in recent_messages if msg["sender"] in ["user", "bot"]
        ]
    
    # Generate answer using the LLM service
    llm_request = LLMRequestDTO(
        question=chat_request.question,
        context_docs=context_docs,
        model=model,
        conversation_history=conversation_history,
        temperature=chat_request.temperature,
        max_tokens=chat_request.max_tokens
    )
    llm_response = await generate_answer(llm_request)
    
    answer = llm_response.get("answer", "")
    processing_time = llm_response.get("processing_time", 0)
    
    # Save the bot's response to Firestore
    bot_message = {
        "sender": "bot",
        "message": answer,
        "conversation_id": chat_request.conversation_id,
        "model": model,
        "processing_time": processing_time,
        "retrieval_profile": retrieval_profile,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    bot_msg_id = await add_document(collection=get_user_chat_collection(chat_request.user_id), data=bot_message)
    logger.info(f"Saved bot message with ID: {bot_msg_id}")
    
    # Return the response
    return {
        "question": chat_request.question,
        "answer": answer,
        "model": model,
        "conversation_id": chat_request.conversation_id,
        "context": context_docs,
        "processing_time": processing_time,
        "retrieval_profile": retrieval_profile
    }

async def get_chat_history(
    user_id: str, 
    conversation_id: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Retrieve chat history for a user."""
    messages = await get_documents(
        collection=get_user_chat_collection(user_id),
        descending=True  # Get newest messages first
    )
    
    if conversation_id:
        messages = [msg for msg in messages if msg.get("conversation_id") == conversation_id]
    
    # Sort by timestamp ascending (oldest first) if needed
    messages.sort(key=lambda x: x.get("timestamp", ""))
    
    if limit is not None:
        messages = messages[offset:offset + limit]
    elif offset > 0:
        messages = messages[offset:]
    
    for message in messages:
        if "timestamp" in message and not isinstance(message["timestamp"], str):
            message["timestamp"] = message["timestamp"].isoformat() if hasattr(message["timestamp"], "isoformat") else str(message["timestamp"])
    
    return messages

async def clear_chat_history(user_id: str, conversation_id: Optional[str] = None) -> None:
    """Delete chat messages for a user."""
    if conversation_id:
        messages = await get_documents(collection=get_user_chat_collection(user_id))
        messages_to_delete = [msg for msg in messages if msg.get("conversation_id") == conversation_id]
        for msg in messages_to_delete:
            await delete_document(collection=get_user_chat_collection(user_id), document_id=msg["id"])
    else:
        await delete_documents(collection=get_user_chat_collection(user_id))