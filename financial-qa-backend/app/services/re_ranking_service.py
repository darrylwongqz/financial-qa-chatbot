# app/services/re_ranking_service.py
import time
import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
from functools import lru_cache
from app.utils.logging_utils import logger
from app.config import (
    RERANKING_MODEL,
    RERANKING_BATCH_SIZE,
    RERANKING_CACHE_SIZE,
    USE_GPU,
    RETRIEVAL_PROFILES,
    DEFAULT_RETRIEVAL_PROFILE
)

# Configuration for re-ranking
RERANKING_CONFIG = {
    "model_name": RERANKING_MODEL,
    "model_version": "1.0.0",  # Track model version for consistency
    "batch_size": RERANKING_BATCH_SIZE,
    "cache_size": RERANKING_CACHE_SIZE,
    "default_weight_factors": {
        "financial_document": 1.1,
        "conversation_turn": 1.15,
        "split_document": 0.95,
        "long_document": 1.05
    },
    "score_combination_method": "weighted_sum",  # Options: weighted_sum, harmonic_mean, geometric_mean, max
    "hybrid_alpha": 0.6  # Weight for dense scores (1-alpha for sparse)
}

# Lazy initialization for the cross-encoder model
_cross_encoder_model = None
_model_lock = None  # Will be initialized on first use
_model_initialization_time = None
_model_initialization_status = "not_initialized"  # Possible values: not_initialized, initializing, ready, failed

def pre_initialize_model(sample_query: str = "What is revenue?", sample_text: str = "Revenue is the income generated from business operations.") -> bool:
    """
    Pre-initialize the cross-encoder model at application startup.
    This function should be called during application startup to avoid the delay
    when the model is first used.
    
    Args:
        sample_query (str): A sample query to use for initialization
        sample_text (str): A sample text to use for initialization
        
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global _model_initialization_status, _model_initialization_time
    
    if _model_initialization_status == "ready":
        logger.info("Cross-encoder model already initialized")
        return True
    
    _model_initialization_status = "initializing"
    start_time = time.time()
    
    try:
        # Get the model (this will initialize it if not already initialized)
        model = get_cross_encoder_model()
        if model is None:
            _model_initialization_status = "failed"
            return False
        
        # Run a sample prediction to fully initialize the model
        model.predict([[sample_query, sample_text]])
        
        _model_initialization_time = time.time() - start_time
        _model_initialization_status = "ready"
        
        logger.info(f"Cross-encoder model pre-initialized in {_model_initialization_time:.2f} seconds")
        return True
    except Exception as e:
        _model_initialization_status = "failed"
        logger.error(f"Failed to pre-initialize cross-encoder model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def get_cross_encoder_model():
    """
    Lazily initialize the cross-encoder model with thread safety.
    
    Returns:
        CrossEncoder: The initialized cross-encoder model.
    """
    global _cross_encoder_model, _model_lock, _model_initialization_status
    
    # Initialize the lock if needed
    if _model_lock is None:
        import threading
        _model_lock = threading.RLock()
    
    # Check if model is already initialized
    if _cross_encoder_model is not None:
        return _cross_encoder_model
    
    # Thread-safe initialization
    with _model_lock:
        # Double-check to avoid race conditions
        if _cross_encoder_model is not None:
            return _cross_encoder_model
        
        _model_initialization_status = "initializing"
        start_time = time.time()
            
        try:
            from sentence_transformers import CrossEncoder
            model_name = RERANKING_CONFIG["model_name"]
            
            # Configure device placement based on available hardware
            device = None  # Let the library choose the best device
            if USE_GPU:
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
                        logger.info(f"Using CUDA for cross-encoder model")
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        device = "mps"
                        logger.info(f"Using MPS (Apple Silicon) for cross-encoder model")
                except ImportError:
                    logger.warning("Torch not available, using default device")
            
            # Initialize the model with the appropriate device
            _cross_encoder_model = CrossEncoder(model_name, device=device)
            
            # Log initialization success
            _model_initialization_status = "ready"
            initialization_time = time.time() - start_time
            logger.info(f"Successfully loaded cross-encoder model: {model_name} (v{RERANKING_CONFIG['model_version']})")
            
            # Save model info for tracking
            model_info = {
                "name": model_name,
                "version": RERANKING_CONFIG["model_version"],
                "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": device or "default",
                "initialization_time": f"{initialization_time:.2f}s"
            }
            logger.info(f"Model info: {json.dumps(model_info)}")
            
        except Exception as e:
            _model_initialization_status = "failed"
            logger.error(f"Failed to load cross-encoder model: {e}")
            _cross_encoder_model = None
            import traceback
            logger.error(traceback.format_exc())
            
    return _cross_encoder_model

def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize a list of scores to the range [0, 1] using min-max normalization.
    
    Args:
        scores (List[float]): List of raw scores.
        
    Returns:
        List[float]: Normalized scores.
    """
    if not scores:
        return scores
    scores_array = np.array(scores)
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    if max_score == min_score:
        return [1.0] * len(scores)
    normalized_scores = (scores_array - min_score) / (max_score - min_score)
    return normalized_scores.tolist()

def sigmoid_normalize_scores(scores: List[float], temperature: float = 1.0) -> List[float]:
    """
    Normalize scores using a sigmoid function for smoother scaling.
    
    Args:
        scores (List[float]): List of raw scores.
        temperature (float): Temperature parameter to control the steepness of the sigmoid.
        
    Returns:
        List[float]: Normalized scores.
    """
    if not scores:
        return scores
    scores_array = np.array(scores)
    # Center the scores around their mean
    centered_scores = scores_array - np.mean(scores_array)
    # Apply sigmoid normalization
    sigmoid_scores = 1 / (1 + np.exp(-centered_scores / temperature))
    return sigmoid_scores.tolist()

def get_document_weight(
    doc: Dict[str, Any], 
    weight_factors: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate a weight for a document based on its metadata.
    
    Args:
        doc (Dict[str, Any]): The document.
        weight_factors (Optional[Dict[str, float]]): Custom weight factors.
        
    Returns:
        float: The document weight.
    """
    # Use default weight factors if none provided
    factors = weight_factors or RERANKING_CONFIG["default_weight_factors"]
    
    weight = 1.0
    metadata = doc.get("metadata", {})
    
    # Apply document type weighting
    doc_type = metadata.get("document_type", "")
    if doc_type == "financial_document":
        weight *= factors.get("financial_document", 1.1)
    elif doc_type == "conversation_turn":
        weight *= factors.get("conversation_turn", 1.15)
    
    # Apply split document weighting
    if metadata.get("is_split", False):
        weight *= factors.get("split_document", 0.95)
    
    # Apply length-based weighting
    text_length = metadata.get("text_length", 0)
    if text_length > 500:  # Consider documents over 500 chars as long
        weight *= factors.get("long_document", 1.05)
    
    # Apply recency weighting if timestamp is available
    timestamp = metadata.get("timestamp", "")
    if timestamp and timestamp.startswith("2023"):  # Example: boost 2023 documents
        weight *= 1.1
    
    # Apply source weighting
    source_id = metadata.get("source_id", "")
    if "annual_report" in source_id:
        weight *= 1.2
    elif "earnings_call" in source_id:
        weight *= 1.15
    
    return weight

@lru_cache(maxsize=RERANKING_CONFIG["cache_size"])
def get_cached_pair_score(query: str, text: str) -> float:
    """
    Get the cross-encoder score for a query-text pair with caching.
    
    Args:
        query (str): The query string.
        text (str): The document text.
        
    Returns:
        float: The cross-encoder score.
    """
    model = get_cross_encoder_model()
    if model is None:
        return 0.0
    
    try:
        score = model.predict([[query, text]])
        return float(score)
    except Exception as e:
        logger.error(f"Error getting cached pair score: {e}")
        return 0.0

def keyword_fallback_score(query: str, text: str) -> float:
    """
    Calculate a fallback score based on keyword matching.
    Used when the cross-encoder model fails.
    
    Args:
        query (str): The query string.
        text (str): The document text.
        
    Returns:
        float: A score between 0 and 1.
    """
    # Simple keyword matching
    query_terms = query.lower().split()
    text_lower = text.lower()
    
    # Count matches
    matches = sum(1 for term in query_terms if term in text_lower)
    
    # Calculate score
    if not query_terms:
        return 0.0
    
    return matches / len(query_terms)

def combine_scores(
    scores: List[float], 
    weights: List[float] = None, 
    method: str = "weighted_sum"
) -> List[float]:
    """
    Combine multiple score lists using the specified method.
    
    Args:
        scores (List[float]): List of score lists.
        weights (List[float]): Weights for each score list.
        method (str): Combination method.
        
    Returns:
        List[float]: Combined scores.
    """
    if not scores:
        return []
    
    # Default to equal weights if not provided
    if weights is None:
        weights = [1.0] * len(scores)
    
    # Normalize weights
    weights_sum = sum(weights)
    if weights_sum > 0:
        weights = [w / weights_sum for w in weights]
    
    # Combine scores using the specified method
    if method == "weighted_sum":
        return [sum(s * w for s, w in zip(score_set, weights)) for score_set in zip(*scores)]
    
    elif method == "harmonic_mean":
        result = []
        for score_set in zip(*scores):
            # Avoid division by zero
            valid_scores = [(s, w) for s, w in zip(score_set, weights) if s > 0]
            if not valid_scores:
                result.append(0.0)
            else:
                # Weighted harmonic mean
                weighted_sum = sum(w for _, w in valid_scores)
                if weighted_sum == 0:
                    result.append(0.0)
                else:
                    harmonic_mean = weighted_sum / sum(w / s for s, w in valid_scores)
                    result.append(harmonic_mean)
        return result
    
    elif method == "geometric_mean":
        result = []
        for score_set in zip(*scores):
            # Avoid zero scores
            valid_scores = [(s, w) for s, w in zip(score_set, weights) if s > 0]
            if not valid_scores:
                result.append(0.0)
            else:
                # Weighted geometric mean
                weighted_sum = sum(w for _, w in valid_scores)
                if weighted_sum == 0:
                    result.append(0.0)
                else:
                    geometric_mean = np.prod([s ** (w / weighted_sum) for s, w in valid_scores])
                    result.append(geometric_mean)
        return result
    
    elif method == "max":
        return [max(score_set) for score_set in zip(*scores)]
    
    else:
        logger.warning(f"Unknown score combination method: {method}, using weighted_sum")
        return [sum(s * w for s, w in zip(score_set, weights)) for score_set in zip(*scores)]

def re_rank_documents(
    query: str, 
    documents: List[Dict[str, Any]],
    batch_size: int = None,
    use_metadata_weighting: bool = True,
    weight_factors: Optional[Dict[str, float]] = None,
    normalize: bool = True,
    normalization_method: str = "minmax",
    fallback_on_error: bool = True
) -> List[Dict[str, Any]]:
    """
    Re-rank documents using a cross-encoder model.
        
    This function processes the documents in batches, obtains relevance scores
    for each query-document pair, optionally applies metadata-based weighting,
    normalizes the scores, and then sorts the documents accordingly.
    
    Args:
        query (str): The query string.
        documents (List[Dict[str, Any]]): List of documents to re-rank.
        batch_size (int): Batch size for processing.
        use_metadata_weighting (bool): Whether to apply metadata-based weighting.
        weight_factors (Optional[Dict[str, float]]): Custom weight factors.
        normalize (bool): Whether to normalize scores.
        normalization_method (str): Normalization method.
        fallback_on_error (bool): Whether to use fallback scoring on error.
        
    Returns:
        List[Dict[str, Any]]: Re-ranked documents.
    """
    if not documents:
        return []
    
    if not query:
        logger.error("Empty query provided to re_rank_documents")
        return documents
    
    # Use default batch size if not specified
    if batch_size is None:
        batch_size = RERANKING_CONFIG["batch_size"]
    
    # Start timing
    start_time = time.time()
    
    try:
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            text = doc.get("text", "")
            if text:
                pairs.append((query, text))
        
        # Get scores
        scores = []
        
        # Process in batches
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            batch_scores = []
            
            for query, text in batch_pairs:
                try:
                    # Use cached scoring
                    score = get_cached_pair_score(query, text)
                    batch_scores.append(score)
                except Exception as e:
                    if fallback_on_error:
                        # Use fallback scoring
                        logger.warning(f"Error scoring pair, using fallback: {e}")
                        fallback_score = keyword_fallback_score(query, text)
                        batch_scores.append(fallback_score)
                    else:
                        # Propagate the error
                        raise
            
            scores.extend(batch_scores)
            
            # Log progress for large document sets
            if len(pairs) > batch_size * 2:
                logger.info(f"Re-ranking progress: {min(i+batch_size, len(pairs))}/{len(pairs)} documents processed")
        
        # Apply metadata weighting if enabled
        if use_metadata_weighting:
            weighted_scores = []
            for doc, score in zip(documents, scores):
                weight = get_document_weight(doc, weight_factors)
                weighted_scores.append(score * weight)
            scores = weighted_scores
            logger.info("Applied metadata weighting to scores")
        
        # Normalize scores if enabled
        if normalize:
            if normalization_method == "sigmoid":
                scores = sigmoid_normalize_scores(scores)
                logger.info("Normalized scores using sigmoid function")
            else:
                scores = normalize_scores(scores)
                logger.info("Normalized scores to [0, 1] using min-max normalization")
        
        # Attach scores to documents
        for doc, score in zip(documents, scores):
            doc["re_rank_score"] = score
        
        # Sort by score
        sorted_docs = sorted(documents, key=lambda x: x.get("re_rank_score", 0), reverse=True)
        
        # Log performance
        processing_time = time.time() - start_time
        logger.info(f"Re-ranked {len(documents)} documents in {processing_time:.2f} seconds")
        
        # Log score distribution
        if scores:
            logger.info(f"Score range: min={min(scores):.4f}, max={max(scores):.4f}, avg={sum(scores)/len(scores):.4f}")
        
        return sorted_docs
    
    except Exception as e:
        logger.error(f"Error in re-ranking: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        if fallback_on_error:
            logger.info("Using original document order due to re-ranking error")
            return documents
        else:
            raise

def hybrid_rerank(
    query: str,
    dense_docs: List[Dict[str, Any]],
    sparse_docs: List[Dict[str, Any]],
    profile: str = DEFAULT_RETRIEVAL_PROFILE
) -> List[Dict[str, Any]]:
    """
    Combine and re-rank results from dense and sparse retrieval.
    
    Args:
        query: The query string
        dense_docs: Results from dense retrieval
        sparse_docs: Results from sparse retrieval
        profile: The retrieval profile to use
        
    Returns:
        Combined and re-ranked results
    """
    # Get the profile configuration
    if profile not in RETRIEVAL_PROFILES:
        logger.warning(f"Unknown retrieval profile: {profile}. Using default: {DEFAULT_RETRIEVAL_PROFILE}")
        profile = DEFAULT_RETRIEVAL_PROFILE
    
    profile_config = RETRIEVAL_PROFILES[profile]["config"]
    
    # Get the score combination method and hybrid alpha from the profile
    score_combination_method = profile_config.get("score_combination_method", "weighted_sum")
    hybrid_alpha = profile_config.get("hybrid_alpha", 0.6)
    
    # Create a mapping of document IDs to documents and scores
    doc_map = {}
    
    # Process dense results
    for doc in dense_docs:
        doc_id = doc.get("id", "")
        if not doc_id:
            continue
            
        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "doc": doc,
                "dense_score": doc.get("score", 0.0),
                "sparse_score": 0.0,
                "combined_score": 0.0
            }
        else:
            doc_map[doc_id]["dense_score"] = doc.get("score", 0.0)
    
    # Process sparse results
    for doc in sparse_docs:
        doc_id = doc.get("id", "")
        if not doc_id:
            continue
            
        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "doc": doc,
                "dense_score": 0.0,
                "sparse_score": doc.get("score", 0.0),
                "combined_score": 0.0
            }
        else:
            doc_map[doc_id]["sparse_score"] = doc.get("score", 0.0)
    
    # Combine scores
    for doc_id, entry in doc_map.items():
        dense_score = entry["dense_score"]
        sparse_score = entry["sparse_score"]
        
        # Apply the selected score combination method
        if score_combination_method == "weighted_sum":
            combined_score = hybrid_alpha * dense_score + (1 - hybrid_alpha) * sparse_score
        elif score_combination_method == "harmonic_mean":
            if dense_score > 0 and sparse_score > 0:
                combined_score = 2 / ((1 / dense_score) + (1 / sparse_score))
            else:
                combined_score = 0.0
        elif score_combination_method == "geometric_mean":
            combined_score = (dense_score * sparse_score) ** 0.5
        elif score_combination_method == "max":
            combined_score = max(dense_score, sparse_score)
        else:
            # Default to weighted sum
            combined_score = hybrid_alpha * dense_score + (1 - hybrid_alpha) * sparse_score
        
        entry["combined_score"] = combined_score
    
    # Sort by combined score
    sorted_entries = sorted(doc_map.values(), key=lambda x: x["combined_score"], reverse=True)
    
    # Return the documents with combined scores
    results = []
    for entry in sorted_entries:
        doc = entry["doc"].copy()
        doc["dense_score"] = entry["dense_score"]
        doc["sparse_score"] = entry["sparse_score"]
        doc["combined_score"] = entry["combined_score"]
        results.append(doc)
    
    return results

def get_model_stats() -> Dict[str, Any]:
    """
    Get detailed statistics about the cross-encoder model.
    
    Returns:
        Dict[str, Any]: Dictionary with model statistics
    """
    global _cross_encoder_model, _model_initialization_status, _model_initialization_time
    
    stats = {
        "model_name": RERANKING_CONFIG["model_name"],
        "model_version": RERANKING_CONFIG["model_version"],
        "status": _model_initialization_status,
        "initialization_time": f"{_model_initialization_time:.2f}s" if _model_initialization_time else None,
        "is_loaded": _cross_encoder_model is not None
    }
    
    # Add device information if model is loaded
    if _cross_encoder_model is not None:
        try:
            import torch
            device_info = {}
            
            # Check if model is using CUDA
            if hasattr(_cross_encoder_model._target_device, 'type') and _cross_encoder_model._target_device.type == 'cuda':
                device_info["type"] = "cuda"
                device_info["name"] = torch.cuda.get_device_name(_cross_encoder_model._target_device)
                device_info["memory"] = {
                    "allocated": f"{torch.cuda.memory_allocated(_cross_encoder_model._target_device) / 1024**2:.2f} MB",
                    "reserved": f"{torch.cuda.memory_reserved(_cross_encoder_model._target_device) / 1024**2:.2f} MB"
                }
            # Check if model is using MPS (Apple Silicon)
            elif hasattr(_cross_encoder_model._target_device, 'type') and _cross_encoder_model._target_device.type == 'mps':
                device_info["type"] = "mps"
                device_info["name"] = "Apple Silicon"
            else:
                device_info["type"] = "cpu"
            
            stats["device"] = device_info
        except Exception as e:
            stats["device"] = {"error": str(e)}
    
    return stats

def get_reranking_stats() -> Dict[str, Any]:
    """
    Get statistics about the re-ranking system.
    
    Returns:
        Dict[str, Any]: Dictionary with re-ranking statistics
    """
    global _cross_encoder_model
    
    # Get model statistics
    model_stats = get_model_stats()
    
    # Get cache statistics
    cache_info = get_cached_pair_score.cache_info()
    cache_stats = {
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "maxsize": cache_info.maxsize,
        "currsize": cache_info.currsize,
        "hit_rate": f"{cache_info.hits / (cache_info.hits + cache_info.misses) * 100:.2f}%" if (cache_info.hits + cache_info.misses) > 0 else "0%"
    }
    
    # Combine all statistics
    stats = {
        "model": model_stats,
        "cache": cache_stats,
        "config": RERANKING_CONFIG
    }
    
    return stats

def clear_reranking_cache():
    """
    Clear the re-ranking cache.
    """
    get_cached_pair_score.cache_clear()
    logger.info("Re-ranking cache cleared") 