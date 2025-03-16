import pytest
import json
from unittest.mock import patch, MagicMock
import asyncio

from app.services.evaluation_service import (
    EvaluationMetrics, 
    extract_qa_pairs, 
    get_test_documents,
    _calculate_aggregate_metrics
)

@pytest.mark.asyncio
async def test_get_test_documents():
    """Test that get_test_documents loads documents from the ground truth file."""
    # Mock the json.load function to return test data
    mock_ground_truth = {
        "doc1": [
            {"question": "Q1", "answer": "A1", "question_type": "extraction"},
            {"question": "Q2", "answer": "A2", "question_type": "calculation"}
        ],
        "doc2": [
            {"question": "Q3", "answer": "A3", "question_type": "comparison"}
        ]
    }
    
    with patch("builtins.open", MagicMock()), \
         patch("json.load", MagicMock(return_value=mock_ground_truth)):
        
        # Call the function
        docs = await get_test_documents()
        
        # Check the results
        assert len(docs) == 2
        assert docs[0]["id"] in ["doc1", "doc2"]
        assert docs[1]["id"] in ["doc1", "doc2"]
        assert "qa_pairs_json" in docs[0]["metadata"]
        assert "qa_pairs_json" in docs[1]["metadata"]
        
        # Check that the QA pairs were properly serialized
        qa_pairs1 = json.loads(docs[0]["metadata"]["qa_pairs_json"])
        assert isinstance(qa_pairs1, list)
        assert len(qa_pairs1) in [1, 2]  # Either doc1 or doc2

def test_extract_qa_pairs():
    """Test that extract_qa_pairs correctly extracts QA pairs from a document."""
    # Test document with qa_pairs_json in metadata
    doc1 = {
        "id": "doc1",
        "metadata": {
            "qa_pairs_json": json.dumps([
                {"question": "Q1", "answer": "A1", "question_type": "extraction"},
                {"question": "Q2", "answer": "A2", "question_type": "calculation"}
            ])
        }
    }
    
    # Call the function
    qa_pairs = extract_qa_pairs(doc1)
    
    # Check the results
    assert len(qa_pairs) == 2
    assert qa_pairs[0]["question"] == "Q1"
    assert qa_pairs[0]["answer"] == "A1"
    assert qa_pairs[1]["question"] == "Q2"
    assert qa_pairs[1]["answer"] == "A2"
    
    # Test document without qa_pairs_json in metadata
    doc2 = {
        "id": "doc2",
        "metadata": {}
    }
    
    # Mock the ground truth file loading
    mock_ground_truth = {
        "doc2": [
            {"question": "Q3", "answer": "A3", "question_type": "comparison"}
        ]
    }
    
    with patch("builtins.open", MagicMock()), \
         patch("json.load", MagicMock(return_value=mock_ground_truth)):
        
        # Call the function
        qa_pairs = extract_qa_pairs(doc2)
        
        # Check the results
        assert len(qa_pairs) == 1
        assert qa_pairs[0]["question"] == "Q3"
        assert qa_pairs[0]["answer"] == "A3"

def test_extract_numerical_value():
    """Test that extract_numerical_value correctly extracts numerical values from text."""
    # Test with direct numbers
    assert EvaluationMetrics.extract_numerical_value(42) == 42.0
    assert EvaluationMetrics.extract_numerical_value("42") == 42.0
    assert EvaluationMetrics.extract_numerical_value(3.14) == 3.14
    
    # Test with percentages
    assert EvaluationMetrics.extract_numerical_value("The result is 42%") == 0.42
    
    # Test with numbers after keywords
    assert EvaluationMetrics.extract_numerical_value("The answer is 42") == 42.0
    assert EvaluationMetrics.extract_numerical_value("The result = 42") == 42.0
    assert EvaluationMetrics.extract_numerical_value("The value: 42") == 42.0
    
    # Test with currency
    assert EvaluationMetrics.extract_numerical_value("The cost is $42") == 42.0
    
    # Test with no numbers
    assert EvaluationMetrics.extract_numerical_value("No numbers here") is None

def test_calculate_metrics():
    """Test that calculate_metrics correctly calculates metrics for different question types."""
    # Test for calculation question
    metrics = EvaluationMetrics.calculate_metrics(
        predicted="The answer is 42",
        ground_truth={"question": "What is 6 * 7?", "answer": 42, "question_type": "calculation"},
        question_type="calculation"
    )
    
    assert "exact_match" in metrics
    assert "numerical_accuracy" in metrics
    assert "percentage_error" in metrics
    assert "financial_accuracy" in metrics
    assert metrics["numerical_accuracy"] == 1.0  # Perfect match
    
    # Test for non-calculation question
    metrics = EvaluationMetrics.calculate_metrics(
        predicted="The capital is Paris",
        ground_truth={"question": "What is the capital of France?", "answer": "Paris", "question_type": "extraction"},
        question_type="extraction"
    )
    
    assert "exact_match" in metrics
    assert "numerical_accuracy" not in metrics
    assert metrics["exact_match"] == 0.0  # Not an exact match due to different text

def test_calculate_aggregate_metrics():
    """Test that _calculate_aggregate_metrics correctly calculates aggregate metrics."""
    # Create some test results
    results = [
        {
            "question_type": "calculation",
            "metrics": {
                "exact_match": 1.0,
                "numerical_accuracy": 1.0,
                "percentage_error": 0.0,
                "financial_accuracy": 1.0,
                "has_context": 1.0,
                "context_count": 3,
                "keyword_coverage": 0.8,
                "has_direct_answer": 1.0,
                "has_calculation_steps": 1.0,
                "has_citations": 0.0
            },
            "response_time": 2.5
        },
        {
            "question_type": "calculation",
            "metrics": {
                "exact_match": 0.0,
                "numerical_accuracy": 0.9,
                "percentage_error": 0.1,
                "financial_accuracy": 0.95,
                "has_context": 1.0,
                "context_count": 2,
                "keyword_coverage": 0.6,
                "has_direct_answer": 1.0,
                "has_calculation_steps": 1.0,
                "has_citations": 0.0
            },
            "response_time": 1.5
        },
        {
            "question_type": "extraction",
            "metrics": {
                "exact_match": 1.0,
                "has_context": 1.0,
                "context_count": 4,
                "keyword_coverage": 0.9,
                "has_direct_answer": 1.0,
                "has_calculation_steps": 0.0,
                "has_citations": 1.0
            },
            "response_time": 2.0
        }
    ]
    
    # Calculate aggregate metrics
    metrics = _calculate_aggregate_metrics(results)
    
    # Check basic metrics
    assert metrics["total_count"] == 3
    assert metrics["error_count"] == 0
    assert metrics["average_response_time"] == 2.0  # (2.5 + 1.5 + 2.0) / 3
    
    # Check exact match
    assert metrics["exact_match"]["average"] == 2/3  # (1.0 + 0.0 + 1.0) / 3
    
    # Check numerical accuracy
    assert metrics["numerical_accuracy"]["average"] == 0.95  # (1.0 + 0.9) / 2
    
    # Check by question type
    assert "calculation" in metrics["by_question_type"]
    assert "extraction" in metrics["by_question_type"]
    assert metrics["by_question_type"]["calculation"]["count"] == 2
    assert metrics["by_question_type"]["extraction"]["count"] == 1
    
    # Check calculation metrics
    calc_metrics = metrics["by_question_type"]["calculation"]
    assert calc_metrics["numerical_accuracy"]["average"] == 0.95  # (1.0 + 0.9) / 2
    assert calc_metrics["financial_accuracy"]["average"] == 0.975  # (1.0 + 0.95) / 2
    
    # Check retrieval quality metrics
    assert metrics["retrieval_quality"]["has_context"]["average"] == 1.0  # All have context
    assert metrics["retrieval_quality"]["context_count"]["average"] == 3.0  # (3 + 2 + 4) / 3 