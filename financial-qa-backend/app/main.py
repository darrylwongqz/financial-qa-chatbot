import os
import sys
import time
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from app.db.firebase_init import initialize_firebase
from app.db.firestore import initialize_firestore
from app.db.pinecone_db import initialize_pinecone

# Load environment variables first
load_dotenv()

# Import directly since we're always running from project root
from app.services.logging_service import setup_logger
from app.config import (
    DEVELOPMENT_MODE, 
    PRE_INIT_MODEL, 
    PRE_WARM_CACHE,
    RERANKING_MODEL,
    RERANKING_BATCH_SIZE,
    RERANKING_CACHE_SIZE,
    USE_GPU
)

# Set up logging with our custom configuration
# This is the central point for all logging configuration
setup_logger()

# Now import the logger after it's been configured
from app.utils.logging_utils import logger, log_endpoint

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info(f"Starting Financial QA API application ({'DEVELOPMENT' if DEVELOPMENT_MODE else 'PRODUCTION'} MODE)")
    # Initialize database modules in the correct order
    logger.info("Initializing database modules...")
    initialize_firebase()
    initialize_firestore()
    initialize_pinecone()
    
    logger.info("All database modules initialized")
    
    # Pre-initialize the cross-encoder model if configured to do so
    if PRE_INIT_MODEL:
        logger.info("Pre-initializing cross-encoder model...")
        try:
            from app.services.re_ranking_service import pre_initialize_model
            success = pre_initialize_model()
            if not success:
                logger.warning("Cross-encoder model pre-initialization was not successful, will initialize on first use")
                # Don't pre-warm cache if model initialization failed
                pre_warm_cache_local = False
            else:
                pre_warm_cache_local = PRE_WARM_CACHE
        except Exception as e:
            logger.error(f"Error pre-initializing cross-encoder model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Don't pre-warm cache if model initialization failed
            pre_warm_cache_local = False
    else:
        logger.info("Skipping cross-encoder model pre-initialization")
        logger.info("Model will be lazy-initialized on first use")
        pre_warm_cache_local = PRE_WARM_CACHE
    
    # Pre-warm the cache if configured to do so
    if pre_warm_cache_local:
        logger.info("Scheduling cache pre-warming...")
        asyncio.create_task(pre_warm_model_cache())
    else:
        logger.info("Skipping cache pre-warming")
    
    yield
    # Shutdown logic
    logger.warning("Application shutting down")

async def pre_warm_model_cache():
    """
    Pre-warm the model cache with common financial queries.
    This runs as a background task after the model is initialized.
    """
    logger.info("Starting to pre-warm model cache with common financial queries...")
    
    # Wait a short time to ensure the server is fully started
    await asyncio.sleep(5)
    
    try:
        # Common financial queries and documents for pre-warming
        common_queries = [
            "What was the revenue last year?",
            "How much profit did the company make?",
            "What is the dividend policy?",
            "How much debt is on the balance sheet?",
            "What are the main expenses?",
            "How has the cash flow changed?",
            "What is the company's market share?",
            "What are the growth projections?",
            "How much was spent on R&D?",
            "What risks does the company face?"
        ]
        
        # Sample documents for each query (simplified for demonstration)
        sample_document = "The company reported strong financial results last year with revenue of $10.5 billion, " \
                         "representing a 5% increase from the previous year. Net income was $2.3 billion. " \
                         "The board approved a quarterly dividend of $0.25 per share."
        
        # Import necessary functions
        from app.services.re_ranking_service import get_cross_encoder_model, get_cached_pair_score
        
        # Check if model is already initialized
        model = get_cross_encoder_model()
        model_was_initialized = model is not None
        
        if not model_was_initialized:
            logger.info("Model not yet initialized, will initialize during cache pre-warming")
        
        # Pre-warm the cache with common queries
        start_time = time.time()
        for i, query in enumerate(common_queries):
            try:
                # Use the cached function to store the result in cache
                score = get_cached_pair_score(query, sample_document)
                logger.debug(f"Pre-warmed cache with query {i+1}/{len(common_queries)}: '{query}', score: {score:.4f}")
                
                # Add a small delay between queries to avoid overwhelming the system
                if i < len(common_queries) - 1:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Failed to pre-warm cache for query '{query}': {e}")
        
        # Log completion
        duration = time.time() - start_time
        logger.info(f"Completed pre-warming model cache with {len(common_queries)} queries in {duration:.2f} seconds")
        
        # Log whether model was initialized during this process
        if not model_was_initialized and get_cross_encoder_model() is not None:
            logger.info("Model was successfully initialized during cache pre-warming")
        
    except Exception as e:
        logger.error(f"Error pre-warming model cache: {e}")
        import traceback
        logger.error(traceback.format_exc())

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Financial QA API",
    description="API for financial question answering system",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Root endpoint
@app.get("/")
@log_endpoint
async def root():
    return {"message": "Welcome to the Financial QA API"}

# Health check endpoint
@app.get("/health")
@log_endpoint
async def health_check():
    return {"status": "healthy"}

# Model status endpoint
@app.get("/model-status")
@log_endpoint
async def model_status():
    try:
        from app.services.re_ranking_service import _model_initialization_status, _model_initialization_time, RERANKING_CONFIG
        
        status = {
            "model_name": RERANKING_CONFIG["model_name"],
            "model_version": RERANKING_CONFIG["model_version"],
            "status": _model_initialization_status,
            "initialization_time": f"{_model_initialization_time:.2f}s" if _model_initialization_time else None,
            "config": {
                "batch_size": RERANKING_CONFIG["batch_size"],
                "cache_size": RERANKING_CONFIG["cache_size"],
                "score_combination_method": RERANKING_CONFIG["score_combination_method"],
                "hybrid_alpha": RERANKING_CONFIG["hybrid_alpha"]
            }
        }
        return status
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting model status: {str(e)}")

# Import and include the chat router
from app.routers import chat, evaluation
app.include_router(chat.router)
app.include_router(evaluation.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 