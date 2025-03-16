"""
Data Transfer Objects (DTOs) for the Financial QA Chatbot.

This module contains Pydantic models that serve as DTOs for passing data between
different components of the application, helping to organize parameters and ensure
type safety.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from app.config import DEFAULT_RETRIEVAL_PROFILE


class ChatRequestDTO(BaseModel):
    """Model representing a chat request."""
    question: str = Field(..., example="What was the percentage change in the S&P 500 in 2008-2009?", 
                         description="The question to ask")
    model: Optional[str] = Field(None, example="gpt-4", 
                                description="The LLM model to use")
    temperature: float = Field(0.7, example=0.7, 
                             description="Temperature for response generation")
    max_tokens: int = Field(1000, example=1000, 
                          description="Maximum tokens in the response")
    retrieval_profile: Optional[str] = Field(None, example="balanced", 
                                           description="Profile for context retrieval")
    conversation_id: Optional[str] = Field(None, example="550e8400-e29b-41d4-a716-446655440000", 
                                         description="Unique identifier for the conversation")
    user_id: Optional[str] = Field(None, example="example@email.com", 
                                  description="Unique identifier for the user. This is automatically set from the path parameter.")


class RetrievalRequestDTO(BaseModel):
    """DTO for retrieval request parameters."""
    query: str = Field(
        ..., 
        example="What is the difference between stocks and bonds?",
        description="The query to search for relevant documents"
    )
    top_k: int = Field(
        5, 
        example=5,
        description="Number of documents to retrieve"
    )
    namespace: Optional[str] = Field(
        None, 
        example="financial_docs",
        description="Namespace to search in (if using namespaces in the vector store)"
    )
    rerank: bool = Field(
        False, 
        example=True,
        description="Whether to apply re-ranking to improve result quality"
    )
    filter_condition: Optional[Dict[str, Any]] = Field(
        None, 
        example={"category": "investments"},
        description="Metadata filter condition for the vector store query"
    )
    use_hybrid_search: bool = Field(
        False, 
        example=True,
        description="Whether to use hybrid search (combining dense and sparse retrieval)"
    )
    expand_query: bool = Field(
        True, 
        example=True,
        description="Whether to expand the query with financial terminology"
    )
    preprocess: bool = Field(
        True, 
        example=True,
        description="Whether to preprocess the query (remove stopwords, etc.)"
    )
    use_cache: bool = Field(
        True, 
        example=True,
        description="Whether to use caching for repeated queries"
    )
    offset: int = Field(
        0, 
        example=0,
        description="Offset for pagination"
    )
    limit: Optional[int] = Field(
        None, 
        example=10,
        description="Limit for pagination"
    )
    fallback_strategy: str = Field(
        "dense", 
        example="dense",
        description="Fallback strategy if the primary retrieval method fails (dense or sparse)"
    )
    profile: str = Field(
        DEFAULT_RETRIEVAL_PROFILE,
        example="balanced",
        description="The retrieval profile to use (fast, balanced, or accurate)"
    )


class LLMRequestDTO(BaseModel):
    """Model representing a request to the LLM service."""
    question: str = Field(..., example="What is the difference between stocks and bonds?", 
                         description="The question to answer")
    context_docs: List[Dict[str, Any]] = Field(..., description="List of relevant context documents")
    model: str = Field(..., example="gpt-4", description="The LLM model to use")
    conversation_history: List[Dict[str, str]] = Field([], description="Previous conversation messages")
    temperature: float = Field(0.7, example=0.7, description="Temperature for response generation")
    max_tokens: int = Field(1000, example=1000, description="Maximum tokens in the response") 