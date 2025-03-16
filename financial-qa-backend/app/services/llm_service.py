# app/services/llm_service.py
import time
import json
import traceback
from typing import List, Dict, Any, Optional
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.config import OPENAI_API_KEY, DEFAULT_MODEL, AVAILABLE_MODELS
from app.utils.logging_utils import logger
from app.models.dto import LLMRequestDTO

# Ensure API key is set
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set. Please provide a valid API key.")
    raise Exception("Missing OpenAI API key.")

def validate_model(model: str) -> str:
    """
    Validate that the requested model is available.
    Returns the validated model name or falls back to the default.
    """
    if not model or model not in AVAILABLE_MODELS:
        logger.warning(f"Requested model '{model}' is not available. Falling back to default: {DEFAULT_MODEL}")
        return DEFAULT_MODEL
    return model

def create_langchain_messages(
    question: str,
    context: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> List[Any]:
    """Create a list of LangChain messages for the chat model."""
    messages = []
    
    # Add system message with context
    system_message = (
        "You are a precise financial expert assistant specialized in numerical analysis and financial document interpretation. "
        "Your primary goal is to provide accurate, concise answers to financial questions using the provided context.\n\n"
        
        "RESPONSE STRUCTURE:\n"
        "1. Start with a DIRECT, CONCISE answer to the question (1-2 sentences maximum)\n"
        "2. Then show your detailed workings and calculations\n"
        "3. Cite sources using [1], [2], [3] etc. in order of appearance\n"
        "4. For calculations, follow this format:\n"
           "   - State the values from sources with citations\n"
           "   - Show the formula\n"
           "   - Show the calculation with actual numbers\n"
           "   - Show the final result\n\n"
        
        "Example Response Structure:\n"
        "The net cash from operating activities increased by 14.13% from 2008 to 2009.\n\n"
        "Workings:\n"
        "- 2008 net cash: $181,001.0 [1]\n"
        "- 2009 net cash: $206,588.0 [1]\n"
        "Formula: (New - Old) / Old × 100%\n"
        "Calculation: (206,588.0 - 181,001.0) / 181,001.0 × 100%\n"
        "           = 25,587.0 / 181,001.0 × 100%\n"
        "           = 14.13%\n\n"
        
        "CONTEXT UTILIZATION INSTRUCTIONS:\n"
        "1. THOROUGHLY examine ALL provided context documents before answering\n"
        "2. Extract EXACT numerical values, percentages, dates, and financial metrics from the context\n"
        "3. When performing calculations, show your work clearly and verify your math is correct\n"
        "4. For numerical answers, maintain consistent precision (e.g., same number of decimal places as in the source)\n"
        "5. Cite sources using [1], [2], etc. when referring to specific information\n\n"
        
        "QUESTION TYPE HANDLING:\n"
        "1. CALCULATION questions: Extract all relevant values from the context, show your calculation steps, and provide a precise numerical answer\n"
        "2. COMPARISON questions: Identify the specific entities being compared, extract relevant metrics for each, and clearly state which is higher/lower/better\n"
        "3. EXPLANATION questions: Provide concise explanations based strictly on information in the context, avoiding speculation\n"
        "4. TEMPORAL questions: Pay special attention to dates, fiscal periods, and time-based trends in the data\n"
        "5. ENTITY questions: Identify specific companies, people, or organizations mentioned in the context and provide relevant details\n"
        "6. LISTING questions: When asked to enumerate items, provide a complete, numbered list based on the context\n\n"
        
        "NUMERICAL ACCURACY GUIDELINES:\n"
        "1. Double-check all calculations before providing an answer\n"
        "2. For percentages, clearly indicate whether you're using percentage points or percent change\n"
        "3. When calculating growth or change, use the formula: (New - Old) / Old × 100% for percentage change\n"
        "4. For financial ratios, use the exact formulas as defined in standard financial practice\n"
        "5. Round only at the final step of calculation, not during intermediate steps\n"
        "6. Express large numbers in appropriate formats (e.g., $1.2 million, $3.5B) consistent with the context\n\n"
        
        "Here is the context to use for answering the question:\n\n"
        f"{context}\n\n"
        "If you cannot find the answer in the context, say so clearly."
    )
    messages.append(SystemMessage(content=system_message))
    
    # Add conversation history if available
    if conversation_history:
        # Add a message explaining the conversation history
        messages.append(SystemMessage(content=(
            "Below is the relevant conversation history. Each exchange provides important context for the current question. "
            "When using this history:\n"
            "1. Review it carefully to maintain consistency in your answers\n"
            "2. If you previously calculated values, ensure new calculations are consistent with past results\n"
            "3. If the current question references previous answers, use those exact values\n"
            "4. For follow-up questions, consider both the new context and previous context\n"
            "5. If you detect any inconsistencies with previous answers, acknowledge and explain the difference\n"
            "6. Pay special attention to any numerical values or calculations mentioned in previous responses"
        )))
        
        for msg in conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "bot":
                messages.append(AIMessage(content=msg["content"]))
        
        # Add a special reminder about the conversation history
        messages.append(SystemMessage(content=(
            "IMPORTANT: The conversation history above provides critical context for your response. Ensure that:\n"
            "1. Any numerical values you use are consistent with previous calculations\n"
            "2. Your answer builds upon, rather than contradicts, previous responses\n"
            "3. If the current question references previous answers, use those exact values\n"
            "4. If you need to revise any previous statements, explicitly acknowledge and explain why"
        )))
    
    # Add the current question
    messages.append(HumanMessage(content=question))
    
    return messages

def format_context(context_docs: List[Dict[str, Any]]) -> str:
    """Format context documents into a single string."""
    formatted_docs = []
    for i, doc in enumerate(context_docs, 1):
        text = doc.get("text", "").strip()
        metadata = doc.get("metadata", {})
        source = metadata.get("source", "Unknown")
        date = metadata.get("date", "Unknown date")
        formatted_docs.append(f"Document {i} (Source: {source}, Date: {date}):\n{text}\n")
    return "\n".join(formatted_docs)

async def generate_answer(
    llm_request: LLMRequestDTO
) -> Dict[str, Any]:
    """
    Generate an answer to a financial question using LangChain and the OpenAI API.
    
    Args:
        llm_request: The LLM request parameters
        
    Returns:
        Dict containing the answer and processing time
    """
    start_time = time.time()
    
    # Extract parameters from the DTO
    question = llm_request.question
    context_docs = llm_request.context_docs
    model = llm_request.model
    conversation_history = llm_request.conversation_history
    temperature = llm_request.temperature
    max_tokens = llm_request.max_tokens
    
    # Log conversation history usage
    if conversation_history:
        logger.info(f"Using conversation history in generate_answer: {len(conversation_history)} messages")
        for i, msg in enumerate(conversation_history[:3]):  # Log first 3 messages as a sample
            logger.info(f"History sample {i+1}: {msg['role']} - {msg['content'][:100]}...")
    else:
        logger.info("No conversation history provided to generate_answer")
    
    # Validate inputs
    if not question:
        logger.error("Empty question provided to generate_answer")
        return {"answer": "I couldn't understand your question. Could you please rephrase it?", "processing_time": 0}
    
    # Process context documents
    if not context_docs:
        logger.warning("No context documents provided for question: " + question)
        context_text = "No relevant information found in the knowledge base. I'll try to answer based on conversation history and general knowledge."
    else:
        # Extract text from context documents
        context_text = format_context(context_docs)
        logger.info(f"Using {len(context_docs)} context documents for question: {question[:50]}...")
    
    try:
        # Create LangChain messages
        messages = create_langchain_messages(question, context_text, conversation_history=conversation_history)
        logger.debug(f"Created {len(messages)} LangChain messages")
        
        # Initialize the chat model
        chat = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Generate response
        response = chat.invoke(messages)
        answer = response.content
        
        processing_time = time.time() - start_time
        
        return {
            "answer": answer,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        logger.error(traceback.format_exc())
        
        return {
            "answer": "I'm sorry, I encountered an error while processing your question. Please try again later.",
            "error": str(e),
            "processing_time": time.time() - start_time
        } 