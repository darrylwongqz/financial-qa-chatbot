"""
API endpoints for evaluation service.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query

from app.services.evaluation_service import (
    run_evaluation,
    get_evaluation,
    get_evaluation_results,
    get_evaluations,
    compare_evaluations
)
from app.config import RETRIEVAL_PROFILES, AVAILABLE_MODELS
from app.utils.logging_utils import logger

router = APIRouter(
    prefix="/api/evaluation",
    tags=["evaluation"],
    responses={404: {"description": "Not found"}},
)

@router.get("/{evaluation_id}/results")
async def get_results_by_evaluation_id(
    evaluation_id: str,
) -> List[Dict[str, Any]]:
    """
    Get the results for an evaluation.
    
    Args:
        evaluation_id: The evaluation ID
        user_id: The user ID (from authentication)
        
    Returns:
        List of evaluation results
    """
    try:
        evaluation = await get_evaluation(evaluation_id)
        
        if not evaluation:
            raise HTTPException(status_code=404, detail=f"Evaluation {evaluation_id} not found")
        
        results = await get_evaluation_results(evaluation_id)
        
        return results
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error getting evaluation results for {evaluation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting evaluation results: {str(e)}")

@router.get("/")
async def list_evaluations(
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Get a list of evaluations.
    
    Args:
        limit: Maximum number of evaluations to return
        user_id: The user ID (from authentication)
        
    Returns:
        List of evaluations
    """
    try:
        evaluations = await get_evaluations(limit=limit)
        
        return evaluations
    
    except Exception as e:
        logger.error(f"Error listing evaluations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing evaluations: {str(e)}")
