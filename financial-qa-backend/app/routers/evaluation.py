"""
API endpoints for evaluation service.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.openapi.models import Example

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

@router.get("/", 
    summary="List evaluations",
    description="Retrieve a list of completed evaluation runs with their metrics and configuration details",
    response_description="Array of evaluation objects with detailed performance metrics",
    responses={
        200: {
            "description": "Evaluations retrieved successfully",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": "330aa128-a24a-49c5-b860-087e614b79fc",
                            "status": "completed",
                            "retrieval_profile": "fast",
                            "model": "gpt-3.5-turbo",
                            "created_at": "2025-03-17T00:00:00.000000",
                            "updated_at": "2025-03-17T00:15:36.000000",
                            "user_id": "admin@email.com",
                            "metrics": {
                                "total_count": 200,
                                "error_count": 60,
                                "error_rate": 0.3,
                                "question_type_counts": {
                                    "extraction": {
                                        "total": 7,
                                        "error_or_no_context": {
                                            "count": 2,
                                            "percentage": 0.2857
                                        },
                                        "successful": {
                                            "count": 5
                                        }
                                    },
                                    "other": {
                                        "total": 75,
                                        "error_or_no_context": {
                                            "count": 25,
                                            "percentage": 0.3333
                                        },
                                        "successful": {
                                            "count": 50
                                        }
                                    },
                                    "calculation": {
                                        "total": 118,
                                        "error_or_no_context": {
                                            "count": 33,
                                            "percentage": 0.2797
                                        },
                                        "successful": {
                                            "count": 85,
                                            "has_calculation_steps": 85
                                        }
                                    }
                                },
                                "non_error_metrics": {
                                    "total_count": 140,
                                    "numerical_accuracy": {
                                        "total": 77,
                                        "count": 140,
                                        "average": 0.55
                                    },
                                    "financial_accuracy": {
                                        "total": 74.2,
                                        "count": 140,
                                        "average": 0.53
                                    },
                                    "answer_relevance": {
                                        "total": 84,
                                        "count": 140,
                                        "average": 0.6
                                    },
                                    "partial_numerical_match": {
                                        "total": 92.4,
                                        "count": 140,
                                        "average": 0.66
                                    },
                                    "has_citations": {
                                        "total": 140,
                                        "count": 140,
                                        "average": 1
                                    },
                                    "has_calculation_steps": {
                                        "total": 85,
                                        "count": 85,
                                        "average": 1
                                    }
                                }
                            }
                        },
                        {
                            "id": "c3d4e5f6-a7b8-4c9d-8e0f-abcdef123456",
                            "status": "completed",
                            "retrieval_profile": "accurate",
                            "model": "gpt-3.5-turbo",
                            "created_at": "2025-03-17T00:00:00.000000",
                            "updated_at": "2025-03-17T00:22:42.000000",
                            "user_id": "admin@email.com",
                            "metrics": {
                                "total_count": 200,
                                "error_count": 13,
                                "error_rate": 0.065,
                                "question_type_counts": {
                                    "extraction": {
                                        "total": 7,
                                        "error_or_no_context": {
                                            "count": 1,
                                            "percentage": 0.1429
                                        },
                                        "successful": {
                                            "count": 6
                                        }
                                    },
                                    "other": {
                                        "total": 75,
                                        "error_or_no_context": {
                                            "count": 3,
                                            "percentage": 0.04
                                        },
                                        "successful": {
                                            "count": 72
                                        }
                                    },
                                    "calculation": {
                                        "total": 118,
                                        "error_or_no_context": {
                                            "count": 9,
                                            "percentage": 0.0763
                                        },
                                        "successful": {
                                            "count": 109,
                                            "has_calculation_steps": 109
                                        }
                                    }
                                },
                                "non_error_metrics": {
                                    "total_count": 187,
                                    "numerical_accuracy": {
                                        "total": 134.64,
                                        "count": 187,
                                        "average": 0.72
                                    },
                                    "financial_accuracy": {
                                        "total": 130.89,
                                        "count": 187,
                                        "average": 0.7
                                    },
                                    "answer_relevance": {
                                        "total": 134.64,
                                        "count": 187,
                                        "average": 0.72
                                    },
                                    "partial_numerical_match": {
                                        "total": 130.9,
                                        "count": 187,
                                        "average": 0.7
                                    },
                                    "has_citations": {
                                        "total": 187,
                                        "count": 187,
                                        "average": 1
                                    },
                                    "has_calculation_steps": {
                                        "total": 109,
                                        "count": 109,
                                        "average": 1
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Error listing evaluations: Database connection failed"}
                }
            }
        }
    }
)
async def list_evaluations(
    limit: int = Query(10, description="Maximum number of evaluations to return (1-100)", ge=1, le=100),
) -> List[Dict[str, Any]]:
    """
    Get a list of evaluation runs with detailed performance metrics.
    
    This endpoint returns a paginated list of evaluation runs, sorted by creation date (newest first).
    Each evaluation contains comprehensive metrics about model performance across different question types
    and evaluation criteria.
    
    The response includes:
    - Basic evaluation metadata (ID, status, model, retrieval profile, timestamps)
    - Error rates and question counts
    - Detailed breakdown by question type (extraction, calculation, other)
    - Performance metrics for non-error responses:
      - numerical_accuracy: Whether numerical values match within tolerance
      - financial_accuracy: Whether financial values match within stricter tolerance
      - answer_relevance: How relevant the answer is to the question
      - partial_numerical_match: Degree of numerical similarity on a continuous scale
      - has_citations: Whether answers include citations to sources
      - has_calculation_steps: For calculation questions, whether steps are shown
    
    Args:
        limit: Maximum number of evaluations to return (1-100)
        
    Returns:
        List of evaluation objects with detailed metrics
    """
    try:
        evaluations = await get_evaluations(limit=limit)
        
        return evaluations
    
    except Exception as e:
        logger.error(f"Error listing evaluations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing evaluations: {str(e)}")
