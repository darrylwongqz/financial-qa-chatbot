import os
from dotenv import load_dotenv
from pathlib import Path
import logging

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Environment settings
DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "False").lower() in ("true", "1", "t", "yes", "y")
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# Logging configuration
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}
LOG_LEVEL = LOG_LEVEL_MAP.get(LOG_LEVEL_STR, logging.INFO if DEVELOPMENT_MODE else logging.WARNING)

# API keys and credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "financial-qa-chatbot")
DENSE_NAMESPACE = os.getenv("DENSE_NAMESPACE", "train_20250316_004516")
SPARSE_INDEX_NAME = os.getenv("SPARSE_INDEX_NAME", "financial-qa-sparse-3")
SPARSE_NAMESPACE = os.getenv("SPARSE_NAMESPACE", "train_sparse_20250316")

# Google Cloud Firestore
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
FIRESTORE_COLLECTION_DATASETS = "datasets"
FIRESTORE_COLLECTION_EVALUATIONS = "evaluations"

# LLM Configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4")
AVAILABLE_MODELS = ["gpt-3.5-turbo", "gpt-4"]
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

# Chat Configuration
DEFAULT_MEMORY_LIMIT = int(os.getenv("DEFAULT_MEMORY_LIMIT", "10"))  # Number of previous messages to include in context

# Retrieval Configuration
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

# Re-ranking Configuration
PRE_INIT_MODEL = os.getenv("PRE_INIT_MODEL", "False").lower() in ("true", "1", "t", "yes", "y") or not DEVELOPMENT_MODE
PRE_WARM_CACHE = os.getenv("PRE_WARM_CACHE", "False").lower() in ("true", "1", "t", "yes", "y") or not DEVELOPMENT_MODE
RERANKING_MODEL = os.getenv("RERANKING_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKING_BATCH_SIZE = int(os.getenv("RERANKING_BATCH_SIZE", "32"))
RERANKING_CACHE_SIZE = int(os.getenv("RERANKING_CACHE_SIZE", "1000"))
USE_GPU = os.getenv("USE_GPU", "False").lower() in ("true", "1", "t", "yes", "y")

# Retrieval Profiles
# These profiles can be exposed to the frontend to allow users to choose their preferred retrieval strategy
RETRIEVAL_PROFILES = {
    "fast": {
        "name": "Fast",
        "description": "Optimized for speed with improved accuracy (< 0.2s response time)",
        "config": {
            "top_k": 5,
            "rerank": False,
            "use_hybrid_search": False,
            "expand_query": True,
            "preprocess": True,
            "use_cache": True
        }
    },
    "balanced": {
        "name": "Balanced",
        "description": "Balanced performance and quality (< 1s response time)",
        "config": {
            "top_k": 7,
            "rerank": True,
            "use_hybrid_search": True,
            "expand_query": True,
            "preprocess": True,
            "use_cache": True,
            "score_combination_method": "weighted_sum",
            "hybrid_alpha": 0.6
        }
    },
    "accurate": {
        "name": "Accurate",
        "description": "Optimized for accuracy (< 2s response time)",
        "config": {
            "top_k": 10,
            "rerank": True,
            "use_hybrid_search": True,
            "expand_query": True,
            "preprocess": True,
            "use_cache": True,
            "score_combination_method": "harmonic_mean"
        }
    }
}

# Default retrieval profile
DEFAULT_RETRIEVAL_PROFILE = "balanced"

# API Configuration
API_PREFIX = "/api/v1" 