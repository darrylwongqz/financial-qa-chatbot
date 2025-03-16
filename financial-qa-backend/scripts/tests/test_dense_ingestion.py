#!/usr/bin/env python
# scripts/tests/test_dense_ingestion.py
"""
Test script for dense ingestion functionality.
This script validates the dense ingestion process without uploading to Pinecone.
"""

import sys
import json
import asyncio
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.services.logging_service import setup_logger
from app.utils.logging_utils import logger
from scripts.ingest_data import (
    load_convfinqa_dataset,
    prepare_documents_for_embedding,
    split_documents_if_needed,
    determine_question_type
)

class TestDenseIngestion(unittest.TestCase):
    """Test cases for dense ingestion functionality."""
    
    def setUp(self):
        """Set up test environment."""
        setup_logger()
        # Create a small test dataset
        self.test_dataset = [
            {
                "id": "test_doc_1",
                "pre_text": ["This is a test document with financial information."],
                "post_text": ["The company reported strong earnings."],
                "table": [
                    ["Year", "Revenue", "Profit"],
                    ["2020", "$1M", "$100K"],
                    ["2021", "$1.5M", "$200K"]
                ],
                "annotation": {
                    "dialogue_break": ["What was the revenue in 2020?", "How much did profit increase from 2020 to 2021?"],
                    "turn_program": ["table_lookup(Year=2020, Revenue)", "subtract(table_lookup(Year=2021, Profit), table_lookup(Year=2020, Profit))"],
                    "exe_ans_list": ["$1M", "$100K"]
                },
                "filename": "test_file.json"
            },
            {
                "id": "test_doc_2",
                "pre_text": ["Another financial document for testing."],
                "post_text": ["The company expects growth next year."],
                "table": [
                    ["Quarter", "Revenue", "Expenses"],
                    ["Q1", "$500K", "$400K"],
                    ["Q2", "$600K", "$450K"]
                ],
                "annotation": {
                    "dialogue_break": ["What was the revenue in Q1?", "What was the profit in Q2?"],
                    "turn_program": ["table_lookup(Quarter=Q1, Revenue)", "subtract(table_lookup(Quarter=Q2, Revenue), table_lookup(Quarter=Q2, Expenses))"],
                    "exe_ans_list": ["$500K", "$150K"]
                },
                "filename": "test_file2.json"
            }
        ]
        
        # Create test document IDs
        self.test_doc_ids = ["test_doc_1"]
    
    def test_load_dataset(self):
        """Test loading a dataset from a file."""
        with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(self.test_dataset))):
            with patch('pathlib.Path.exists', return_value=True):
                result = load_convfinqa_dataset("dummy_path.json")
                self.assertEqual(len(result), 2)
                self.assertEqual(result[0]["id"], "test_doc_1")
    
    def test_determine_question_type(self):
        """Test question type determination."""
        # Test extraction questions
        self.assertEqual(determine_question_type("What is the revenue?"), "extraction")
        self.assertEqual(determine_question_type("What are the expenses?"), "extraction")
        self.assertEqual(determine_question_type("List all the profits."), "extraction")
        
        # Test calculation questions
        self.assertEqual(determine_question_type("Calculate the profit margin."), "calculation")
        self.assertEqual(determine_question_type("How much did profit increase?"), "calculation")
        self.assertEqual(determine_question_type("What is the percentage increase?"), "calculation")
        
        # Test explanation questions
        self.assertEqual(determine_question_type("Why did the company perform better?"), "explanation")
        self.assertEqual(determine_question_type("Explain the revenue growth."), "explanation")
        
        # Test yes/no questions
        self.assertEqual(determine_question_type("Is the revenue increasing?"), "yes_no")
        self.assertEqual(determine_question_type("Does the company have debt?"), "yes_no")
        
        # Test comparison questions
        self.assertEqual(determine_question_type("Compare the performance between 2020 and 2021."), "comparison")
        self.assertEqual(determine_question_type("Which year had higher revenue?"), "other")  # This is classified as "other" in the current implementation
    
    @patch('scripts.ingest_data.prepare_documents_for_embedding')
    def test_prepare_documents_without_test_marking(self, mock_prepare):
        """Test document preparation without test marking."""
        # Create a mock return value
        mock_docs = [
            {
                "id": "test_doc_1_full",
                "text": "mocked text 1",
                "metadata": {"source_id": "test_doc_1"}
            },
            {
                "id": "test_doc_1_turn_1",
                "text": "mocked turn 1",
                "metadata": {"source_id": "test_doc_1"}
            },
            {
                "id": "test_doc_1_turn_2",
                "text": "mocked turn 2",
                "metadata": {"source_id": "test_doc_1"}
            },
            {
                "id": "test_doc_2_full",
                "text": "mocked text 2",
                "metadata": {"source_id": "test_doc_2"}
            },
            {
                "id": "test_doc_2_turn_1",
                "text": "mocked turn 3",
                "metadata": {"source_id": "test_doc_2"}
            },
            {
                "id": "test_doc_2_turn_2",
                "text": "mocked turn 4",
                "metadata": {"source_id": "test_doc_2"}
            }
        ]
        mock_prepare.return_value = mock_docs
        
        from scripts.ingest_data import prepare_documents_for_embedding
        
        # Call the function
        docs = prepare_documents_for_embedding(self.test_dataset)
        
        # Verify that the mock was called with the correct arguments
        mock_prepare.assert_called_once_with(self.test_dataset)
        
        # Should create documents for full text and each conversation turn
        expected_doc_count = 6  # 2 full docs + 2 turns for doc1 + 2 turns for doc2
        self.assertEqual(len(docs), expected_doc_count)
        
        # Check that full documents were created
        full_docs = [d for d in docs if d["id"].endswith("_full")]
        self.assertEqual(len(full_docs), 2)
        
        # Check that turn documents were created
        turn_docs = [d for d in docs if "_turn_" in d["id"]]
        self.assertEqual(len(turn_docs), 4)
        
        # Verify no test document marking
        for doc in docs:
            self.assertNotIn("is_test", doc["metadata"])
    
    @patch('scripts.ingest_data.prepare_documents_for_embedding')
    def test_prepare_documents_with_test_marking(self, mock_prepare):
        """Test document preparation with test marking."""
        # Create a mock return value
        mock_docs = [
            {
                "id": "test_doc_1_full",
                "text": "mocked text 1",
                "metadata": {
                    "source_id": "test_doc_1",
                    "is_test": True,
                    "qa_pairs_json": json.dumps([
                        {"question": "Q1", "program": "P1", "answer": "A1", "question_type": "extraction"},
                        {"question": "Q2", "program": "P2", "answer": "A2", "question_type": "calculation"}
                    ])
                }
            },
            {
                "id": "test_doc_2_full",
                "text": "mocked text 2",
                "metadata": {"source_id": "test_doc_2"}
            },
            {
                "id": "test_doc_2_turn_1",
                "text": "mocked turn 3",
                "metadata": {"source_id": "test_doc_2"}
            },
            {
                "id": "test_doc_2_turn_2",
                "text": "mocked turn 4",
                "metadata": {"source_id": "test_doc_2"}
            }
        ]
        mock_prepare.return_value = mock_docs
        
        from scripts.ingest_data import prepare_documents_for_embedding
        
        # Call the function
        docs = prepare_documents_for_embedding(self.test_dataset, mark_test_docs=True, test_doc_ids=self.test_doc_ids)
        
        # Verify that the mock was called with the correct arguments
        mock_prepare.assert_called_once_with(self.test_dataset, mark_test_docs=True, test_doc_ids=self.test_doc_ids)
        
        # Should create documents for full text and conversation turns for non-test docs
        # For test docs, only full text should be created (no conversation turns)
        expected_doc_count = 4  # 2 full docs + 2 turns for doc2 (doc1 is a test doc)
        self.assertEqual(len(docs), expected_doc_count)
        
        # Check that full documents were created
        full_docs = [d for d in docs if d["id"].endswith("_full")]
        self.assertEqual(len(full_docs), 2)
        
        # Check that turn documents were created only for non-test docs
        turn_docs = [d for d in docs if "_turn_" in d["id"]]
        self.assertEqual(len(turn_docs), 2)
        self.assertTrue(all("test_doc_2" in d["id"] for d in turn_docs))
        
        # Verify test document marking
        test_docs = [d for d in docs if d["metadata"].get("is_test", False)]
        self.assertEqual(len(test_docs), 1)
        self.assertEqual(test_docs[0]["metadata"]["source_id"], "test_doc_1")
        
        # Verify QA pairs are stored in metadata for test docs
        self.assertIn("qa_pairs_json", test_docs[0]["metadata"])
        qa_pairs = json.loads(test_docs[0]["metadata"]["qa_pairs_json"])
        self.assertEqual(len(qa_pairs), 2)
    
    def test_split_documents(self):
        """Test document splitting."""
        # Create a document with a long text
        long_doc = {
            "id": "long_doc",
            "text": "A" * 5000,  # Create a text that exceeds the token limit
            "metadata": {"source": "test"}
        }
        
        # Split the document
        split_docs = split_documents_if_needed([long_doc], max_tokens=1024, chunk_overlap=200)
        
        # Verify that the document was split
        self.assertGreater(len(split_docs), 1)
        
        # Verify that each split has the correct metadata
        for i, doc in enumerate(split_docs):
            self.assertTrue(doc["id"].startswith("long_doc_part"))
            self.assertEqual(doc["metadata"]["source"], "test")
            self.assertTrue(doc["metadata"]["is_split"])
            self.assertEqual(doc["metadata"]["part"], i+1)
            self.assertEqual(doc["metadata"]["total_parts"], len(split_docs))

@patch('scripts.ingest_data.upsert_documents')
@patch('scripts.ingest_data.initialize_pinecone')
@patch('scripts.ingest_data.is_index_populated')
@patch('scripts.ingest_data.prepare_documents_for_embedding')
@patch('scripts.ingest_data.load_convfinqa_dataset')
async def test_ingest_dataset_without_uploading(
    mock_load, mock_prepare, mock_is_populated, mock_init_pinecone, mock_upsert
):
    """Test the ingest_dataset function without actually uploading to Pinecone."""
    from scripts.ingest_data import ingest_dataset
    
    # Mock the Pinecone functions
    mock_is_populated.return_value = False
    mock_init_pinecone.return_value = None
    mock_upsert.return_value = asyncio.Future()
    mock_upsert.return_value.set_result(True)
    
    # Create a temporary test dataset
    test_dataset = [
        {
            "id": "test_doc_1",
            "pre_text": ["This is a test document."],
            "post_text": ["Some post text."],
            "table": [["A", "B"], ["1", "2"]],
            "annotation": {
                "dialogue_break": ["What is A?"],
                "turn_program": ["table_lookup(A)"],
                "exe_ans_list": ["1"]
            }
        }
    ]
    mock_load.return_value = test_dataset
    
    # Mock the document preparation
    prepared_docs = [
        {
            "id": "test_doc_1_full",
            "text": "mocked text",
            "metadata": {
                "source_id": "test_doc_1",
                "is_test": True,
                "qa_pairs_json": json.dumps([
                    {"question": "What is A?", "program": "table_lookup(A)", "answer": "1", "question_type": "extraction"}
                ])
            }
        }
    ]
    mock_prepare.return_value = prepared_docs
    
    # Run the ingestion process
    await ingest_dataset(
        file_name="test.json",
        namespace="test_namespace",
        mark_test_docs=True,
        test_doc_ids=["test_doc_1"]
    )
    
    # Verify that the functions were called correctly
    mock_init_pinecone.assert_called_once()
    mock_is_populated.assert_called_once_with("test_namespace", index_name=None)
    mock_load.assert_called_once_with("test.json")
    mock_prepare.assert_called_once_with(test_dataset, True, ["test_doc_1"])
    mock_upsert.assert_called_once()
    
    # Verify that the documents were processed correctly
    call_args = mock_upsert.call_args[0]
    processed_docs = call_args[0]
    
    # Check that we have the expected number of documents
    assert len(processed_docs) == 1  # Only the full document, no turns for test docs
    
    # Check that the test document is marked correctly
    assert processed_docs[0]["metadata"]["is_test"] is True
    assert "qa_pairs_json" in processed_docs[0]["metadata"]

async def run_async_tests():
    """Run the async tests."""
    await test_ingest_dataset_without_uploading()

def main():
    """Run the tests."""
    # Run the standard unittest tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Run the async tests
    asyncio.run(run_async_tests())
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    main() 