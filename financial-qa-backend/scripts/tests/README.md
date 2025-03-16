# Ingestion Tests

This directory contains test scripts for validating the ingestion functionality without actually uploading data to Pinecone.

## Available Tests

1. **Dense Ingestion Tests** (`test_dense_ingestion.py`): Tests for the dense vector ingestion process.
2. **Sparse Ingestion Tests** (`test_sparse_ingestion.py`): Tests for the sparse vector ingestion process.
3. **Test Runner** (`run_ingestion_tests.py`): A script to run both dense and sparse ingestion tests.

## Running the Tests

You can run the tests using the following commands:

### Run All Tests

```bash
python financial-qa-backend/scripts/tests/run_ingestion_tests.py --all
```

### Run Only Dense Ingestion Tests

```bash
python financial-qa-backend/scripts/tests/run_ingestion_tests.py --dense
```

### Run Only Sparse Ingestion Tests

```bash
python financial-qa-backend/scripts/tests/run_ingestion_tests.py --sparse
```

### Run Individual Test Files Directly

```bash
python financial-qa-backend/scripts/tests/test_dense_ingestion.py
python financial-qa-backend/scripts/tests/test_sparse_ingestion.py
```

## What the Tests Validate

These tests validate the following aspects of the ingestion process:

1. **Document Loading**: Tests that documents can be loaded from JSON files.
2. **Text Cleaning**: Tests the text cleaning functions for both dense and sparse indexing.
3. **Document Preparation**: Tests the preparation of documents for embedding, including:
   - Creating full document representations
   - Creating conversation turn documents
   - Handling test document marking
   - Storing QA pairs in metadata for test documents
4. **Document Splitting**: Tests the splitting of long documents into smaller chunks.
5. **Ingestion Process**: Tests the full ingestion process without actually uploading to Pinecone.

## How the Tests Work

The tests use Python's unittest framework and mock the Pinecone functions to prevent actual uploads. This allows you to validate the ingestion process without consuming Pinecone resources or requiring a connection to Pinecone.

The tests include:

- Unit tests for individual functions
- Integration tests for the full ingestion process
- Mocking of external dependencies (Pinecone, file operations)
- Validation of test document marking functionality

## Adding New Tests

To add new tests:

1. Create a new test file in this directory
2. Import the necessary functions from the ingestion scripts
3. Create test cases using the unittest framework
4. Update the `run_ingestion_tests.py` script to include your new tests 