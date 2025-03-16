# app/services/retrieval_service.py
import time
import json
import re
import traceback
import sys
from typing import List, Dict, Any, Optional

# Add the project root to the Python path if needed
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.config import (
    TOP_K_RETRIEVAL, 
    RETRIEVAL_PROFILES, 
    DEFAULT_RETRIEVAL_PROFILE,
    SPARSE_INDEX_NAME,
    SPARSE_NAMESPACE,
    DENSE_NAMESPACE
)
from app.utils.logging_utils import logger
from app.db.pinecone_db import initialize_pinecone, get_embeddings
from app.models.dto import RetrievalRequestDTO

import numpy as np

# Define financial stopwords - words that are common in English but not important for financial retrieval
# This list excludes financial terms that might be in standard stopword lists
FINANCIAL_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when', 'where', 'how', 'who', 'which',
    'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'into',
    'during', 'before', 'after', 'while', 'of', 'to', 'in', 'for', 'on', 'by', 'with', 'is', 'am', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'can', 'could',
    'should', 'would', 'might', 'will', 'shall', 'may', 'must', 'ought'
}

# Financial terms that should never be treated as stopwords
FINANCIAL_TERMS_TO_PRESERVE = {
    # Market direction terms
    'up', 'down', 'high', 'low', 'higher', 'lower', 'increase', 'decrease', 'gain', 'loss',
    
    # Time periods
    'year', 'quarter', 'month', 'week', 'day', 'annual', 'quarterly', 'monthly', 'weekly', 'daily',
    
    # Financial metrics
    'revenue', 'profit', 'earnings', 'income', 'expense', 'cost', 'price', 'value', 'rate', 'ratio',
    'margin', 'growth', 'return', 'yield', 'dividend', 'interest', 'debt', 'equity', 'asset', 'liability',
    
    # Financial instruments
    'stock', 'bond', 'share', 'option', 'future', 'etf', 'fund', 'index', 'commodity',
    
    # Market terms
    'market', 'trade', 'buy', 'sell', 'bid', 'ask', 'bull', 'bear', 'volatility',
    
    # Comparative terms
    'over', 'under', 'above', 'below', 'between', 'versus', 'vs', 'against', 'compared',
    
    # Financial reporting
    'report', 'statement', 'balance', 'sheet', 'cash', 'flow', 'fiscal', 'financial',
    
    # Numeric indicators
    'first', 'second', 'third', 'fourth', 'q1', 'q2', 'q3', 'q4', 'fy', 'ytd',
    
    # Financial abbreviations
    'eps', 'p/e', 'roi', 'roe', 'roa', 'ebitda', 'cagr', 'yoy', 'mom', 'ttm'
}

# Remove any financial terms from the stopwords list
FINANCIAL_STOPWORDS = FINANCIAL_STOPWORDS - FINANCIAL_TERMS_TO_PRESERVE

def preprocess_query(query: str) -> str:
    """
    Preprocess the query by removing stopwords and normalizing text.
    
    Args:
        query (str): The original query.
        
    Returns:
        str: The preprocessed query.
    """
    # Convert to lowercase
    query = query.lower()
    
    # Remove special characters but keep important financial symbols
    query = re.sub(r'[^\w\s\$\%\.\,\-]', ' ', query)
    
    # Remove stopwords
    tokens = query.split()
    filtered_tokens = [token for token in tokens if token not in FINANCIAL_STOPWORDS]
    
    # If all tokens were stopwords, return the original query
    if not filtered_tokens:
        return query
    
    # Reconstruct the query
    return ' '.join(filtered_tokens)

def expand_financial_query(query: str) -> str:
    """
    Expand the query with financial terminology to improve retrieval.
    
    Args:
        query (str): The original query.
        
    Returns:
        str: The expanded query.
    """
    # Financial terms that might be abbreviated or have synonyms
    financial_terms = {
        "revenue": ["rev", "sales", "income", "earnings", "top line"],
        "profit": ["earnings", "net income", "bottom line", "gain", "surplus"],
        "expense": ["cost", "expenditure", "spending", "outlay"],
        "assets": ["property", "holdings", "resources"],
        "liabilities": ["debt", "obligations", "payables"],
        "equity": ["shareholder equity", "stockholder equity", "net worth"],
        "dividend": ["payout", "distribution", "disbursement"],
        "stock": ["share", "equity", "security"],
        "balance sheet": ["statement of financial position", "financial position"],
        "income statement": ["profit and loss statement", "P&L", "statement of operations"],
        "cash flow": ["cash movement", "liquidity", "cash position"],
        "market share": ["market position", "industry share"],
        "growth": ["expansion", "increase", "appreciation"],
        "decline": ["decrease", "reduction", "depreciation"],
        "investment": ["capital expenditure", "capex", "allocation"],
        "acquisition": ["takeover", "purchase", "buyout"],
        "merger": ["consolidation", "combination", "integration"],
        "quarterly": ["Q1", "Q2", "Q3", "Q4"],
        "annual": ["yearly", "fiscal year", "FY"],
        "forecast": ["projection", "outlook", "guidance"]
    }
    
    expanded_query = query
    
    # Check if any financial terms are in the query and expand with synonyms
    for term, synonyms in financial_terms.items():
        if term.lower() in query.lower():
            # Add relevant synonyms to the query
            expanded_terms = " OR ".join(synonyms[:3])  # Limit to 3 synonyms to avoid query explosion
            if expanded_terms:
                expanded_query = f"{expanded_query} ({expanded_terms})"
    
    return expanded_query

def get_sparse_results_independent(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Get sparse retrieval results using Pinecone's native sparse vector support.
    This uses TF-IDF to create sparse vectors for the query.
    
    Args:
        query: The query string.
        top_k: Number of results to return.
        
    Returns:
        List of documents with scores.
    """
    try:
        # Clean and preprocess the query for sparse retrieval
        from scripts.ingest_data import clean_text_for_sparse
        
        # Store original query for logging
        original_query = query
        
        # Process query with the same function used for documents
        processed_query = clean_text_for_sparse(query)
        
        logger.info(f"Original query: '{original_query}'")
        logger.info(f"Processed query for sparse retrieval: '{processed_query}'")
        
        # Use TF-IDF to create a sparse vector
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Configure TfidfVectorizer with settings optimized for financial text
        vectorizer = TfidfVectorizer(
            max_features=2000,  # Increased from 1000 to capture more terms
            ngram_range=(1, 2),  # Include bigrams for better matching
            min_df=1,  # Include terms that appear at least once
            max_df=1.0,  # 1.0 to include all terms
            use_idf=True,
            sublinear_tf=True  # Apply sublinear tf scaling (1 + log(tf))
        )
        
        # Fit and transform in one step
        vectorizer.fit([processed_query])
        sparse_vector = vectorizer.transform([processed_query])
        
        # Convert to indices and values format
        indices = sparse_vector.indices.tolist()
        values = sparse_vector.data.tolist()
        
        # Log the sparse vector details for debugging
        feature_names = vectorizer.get_feature_names_out()
        sparse_terms = [feature_names[idx] for idx in indices]
        logger.debug(f"Sparse vector terms: {sparse_terms}")
        logger.debug(f"Sparse vector values: {values}")
        
        # Create sparse values dict
        sparse_values = {
            "indices": indices,
            "values": values
        }
        
        # Get the sparse index
        from app.db.pinecone_db import sparse_index
        
        # Initialize the sparse index if needed
        if sparse_index is None:
            logger.info("Initializing sparse index")
            from app.db.pinecone_db import initialize_pinecone
            initialize_pinecone(index_name=SPARSE_INDEX_NAME)
            
            # Get the sparse index again after initialization
            from app.db.pinecone_db import sparse_index as current_sparse_index
            
            if current_sparse_index is None:
                logger.error("Sparse index is still None after initialization")
                return []
                
            # Use the newly initialized sparse index
            current_index = current_sparse_index
        else:
            # Use the existing sparse index
            current_index = sparse_index
        
        # Query the sparse index
        results = current_index.query(
            sparse_vector=sparse_values,
            top_k=top_k,
            include_metadata=True,
            namespace=SPARSE_NAMESPACE
        )
        
        # Process results
        processed_results = []
        for match in results.matches:
            doc = {
                "id": match.id,
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata,
                "score": match.score,
                "source": "sparse"
            }
            processed_results.append(doc)
        
        logger.info(f"Sparse retrieval completed, found {len(processed_results)} results")
        return processed_results
        
    except Exception as e:
        logger.error(f"Error in sparse retrieval: {e}")
        logger.error(traceback.format_exc())
        return []

async def get_relevant_context(
    retrieval_request: RetrievalRequestDTO
) -> List[Dict[str, Any]]:
    """
    Get relevant context for a query using vector search.
    
    Args:
        retrieval_request (RetrievalRequestDTO): The retrieval request parameters.
        
    Returns:
        List[Dict[str, Any]]: List of relevant documents with their metadata.
    """
    query = retrieval_request.query
    top_k = retrieval_request.top_k
    namespace = retrieval_request.namespace
    rerank = retrieval_request.rerank
    filter_condition = retrieval_request.filter_condition
    use_hybrid_search = retrieval_request.use_hybrid_search
    expand_query = retrieval_request.expand_query
    preprocess = retrieval_request.preprocess
    offset = retrieval_request.offset
    limit = retrieval_request.limit
    fallback_strategy = retrieval_request.fallback_strategy
    profile = retrieval_request.profile if hasattr(retrieval_request, 'profile') else DEFAULT_RETRIEVAL_PROFILE
    
    # Ensure Pinecone is initialized
    initialize_pinecone()
    
    # Start timing
    start_time = time.time()
    
    # Preprocess query if enabled
    if preprocess:
        processed_query = preprocess_query(query)
    else:
        processed_query = query
    
    # Expand query with financial terminology if enabled
    if expand_query:
        expanded_query = expand_financial_query(processed_query)
    else:
        expanded_query = processed_query
    
    # Initialize results
    results = []
    
    try:
        # Use hybrid search if enabled, otherwise use dense search
        if use_hybrid_search:
            try:
                results = await get_hybrid_search_results(
                    expanded_query, 
                    top_k=top_k, 
                    namespace=namespace,
                    filter_condition=filter_condition,
                    profile=profile
                )
            except Exception as e:
                logger.error(f"Error in hybrid search: {e}")
                if fallback_strategy == "dense":
                    logger.info("Falling back to dense retrieval")
                    results = await get_dense_results(
                        expanded_query, 
                        top_k=top_k,
                        namespace=namespace,
                        filter_condition=filter_condition
                    )
                elif fallback_strategy == "sparse":
                    logger.info("Falling back to sparse retrieval")
                    results = get_sparse_results_independent(
                        expanded_query, 
                        top_k=top_k
                    )
        else:
            # Use dense retrieval if hybrid search is disabled
            logger.info(f"Using dense-only retrieval for query: '{expanded_query}' (fast profile)")
            results = await get_dense_results(
                expanded_query, 
                top_k=top_k,
                namespace=namespace,
                filter_condition=filter_condition
            )
        
        # Apply re-ranking if enabled
        if rerank and results:
            try:
                from app.services.re_ranking_service import re_rank_documents
                results = re_rank_documents(expanded_query, results)
            except Exception as e:
                logger.error(f"Error in re-ranking: {e}")
                # Continue with the non-reranked results
        
        # Apply pagination if specified
        if offset > 0 or limit is not None:
            end_idx = None if limit is None else offset + limit
            results = results[offset:end_idx]
        
        # Log retrieval time
        retrieval_time = time.time() - start_time
        logger.debug(f"Retrieved {len(results)} documents in {retrieval_time:.2f} seconds")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in retrieval: {e}")
        logger.error(traceback.format_exc())
        return []

async def get_relevant_context_with_profile(
    query: str,
    profile: str = DEFAULT_RETRIEVAL_PROFILE,
    namespace: str = DENSE_NAMESPACE,
    filter_condition: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get relevant context documents using the specified retrieval profile.
    """
    # Get profile configuration
    profile_config = RETRIEVAL_PROFILES.get(profile, RETRIEVAL_PROFILES[DEFAULT_RETRIEVAL_PROFILE])
    config = profile_config.get("config", {})
    
    try:
        # Initialize variables for results
        final_results = []
        top_k = config.get("top_k", TOP_K_RETRIEVAL)
        
        # Get dense results
        dense_results = await get_dense_results(
            query=query,
            top_k=top_k,  # Use the profile's top_k
            namespace=namespace,
            filter_condition=filter_condition
        )
        final_results.extend(dense_results)
        
        # Get sparse results if hybrid search is enabled
        if config.get("use_hybrid_search", False):
            sparse_results = get_sparse_results_independent(
                query=query,
                top_k=top_k  # Use the same top_k for consistency
            )
            final_results.extend(sparse_results)
        
        # Deduplicate results based on document ID
        seen_ids = set()
        unique_results = []
        for doc in final_results:
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                unique_results.append(doc)
        
        # Sort by score and limit to top_k
        unique_results.sort(key=lambda x: x["score"], reverse=True)
        final_results = unique_results[:top_k]
        
        # Apply re-ranking if enabled
        if config.get("rerank", False) and final_results:
            try:
                from app.services.re_ranking_service import re_rank_documents
                final_results = re_rank_documents(query, final_results)
            except Exception as e:
                logger.error(f"Error in re-ranking: {e}")
                # Continue with the non-reranked results
        
        return final_results
        
    except Exception as e:
        logger.error(f"Error in get_relevant_context_with_profile: {str(e)}")
        logger.error(traceback.format_exc())
        return []

async def get_dense_results(
    query: str, 
    top_k: int = TOP_K_RETRIEVAL, 
    namespace: Optional[str] = None,
    filter_condition: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Get dense retrieval results using Pinecone.
    
    Args:
        query: The query string.
        top_k: Number of results to return.
        
    Returns:
        List of documents with scores.
    """
    try:
        # Store original query for logging
        original_query = query
        
        # Preprocess query for dense retrieval
        # We'll use a lighter preprocessing for dense retrieval
        # but log both versions for consistency checking
        processed_query = preprocess_query(query)
        
        logger.info(f"Original query: '{original_query}'")
        logger.info(f"Processed query for dense retrieval: '{processed_query}'")
        
        # For comparison, also log how it would be processed for sparse retrieval
        from scripts.ingest_data import clean_text_for_sparse
        sparse_processed = clean_text_for_sparse(query)
        logger.debug(f"For comparison - sparse processed query would be: '{sparse_processed}'")
        
        # Get embeddings for the query
        embeddings = get_embeddings([processed_query])
        if not embeddings or len(embeddings) == 0:
            logger.error("Failed to get embeddings for query")
            return []
            
        query_embedding = embeddings[0]
        
        # Try to use the dense_index first
        from app.db.pinecone_db import dense_index, index
        current_index = dense_index if dense_index is not None else index
        
        # Initialize Pinecone if needed
        if current_index is None:
            logger.info("Initializing Pinecone")
            initialize_pinecone()
            # Try again after initialization
            from app.db.pinecone_db import dense_index as current_dense_index, index as current_main_index
            current_index = current_dense_index if current_dense_index is not None else current_main_index
        
        if current_index is None:
            logger.error("Pinecone index is still None after initialization")
            return []
        
        # Query Pinecone
        results = current_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace or DENSE_NAMESPACE
        )
        
        # Process results
        processed_results = []
        for match in results.matches:
            doc = {
                "id": match.id,
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata,
                "score": match.score,
                "source": "dense"
            }
            processed_results.append(doc)
        
        logger.info(f"Dense retrieval completed, found {len(processed_results)} results")
        return processed_results
        
    except Exception as e:
        logger.error(f"Error in dense retrieval: {e}")
        logger.error(traceback.format_exc())
        return []

async def get_hybrid_search_results(
    query: str, 
    top_k: int = TOP_K_RETRIEVAL, 
    namespace: Optional[str] = None,
    filter_condition: Optional[Dict[str, Any]] = None,
    profile: str = DEFAULT_RETRIEVAL_PROFILE
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining dense and sparse retrieval.
    This function is only called when use_hybrid_search is True in the retrieval profile.
    The "fast" profile doesn't use this function as it has use_hybrid_search set to False.
    
    Args:
        query: The query string to search for
        top_k: Number of results to return
        namespace: Optional namespace to search in
        filter_condition: Optional filter condition for Pinecone query
        profile: The retrieval profile to use
        
    Returns:
        List of document dictionaries with text and metadata
    """
    from app.services.re_ranking_service import hybrid_rerank
    
    logger.info(f"Performing hybrid search for query: '{query}' (profile={profile})")
    start_time = time.time()
    
    try:
        # Get dense results
        dense_results = await get_dense_results(
            query=query,
            top_k=top_k * 2,  # Get more results for re-ranking
            namespace=namespace,
            filter_condition=filter_condition
        )
        
        # Get sparse results (independent)
        sparse_results = get_sparse_results_independent(
            query=query,
            top_k=top_k * 2  # Get more results for re-ranking
        )
        
        # Combine and re-rank results
        hybrid_results = hybrid_rerank(
            query=query,
            dense_docs=dense_results,
            sparse_docs=sparse_results,
            profile=profile
        )
        
        # Limit to top_k results
        hybrid_results = hybrid_results[:top_k]
        
        end_time = time.time()
        logger.info(f"Hybrid search completed in {end_time - start_time:.4f}s, found {len(hybrid_results)} results")
        
        return hybrid_results
    
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        logger.error(traceback.format_exc())
        return [] 