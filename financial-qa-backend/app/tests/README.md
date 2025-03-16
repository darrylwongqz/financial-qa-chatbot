# Financial QA Chatbot Tests

This directory contains tests for the Financial QA Chatbot application. The tests are organized by module and service.

## Test Structure

- `services/`: Tests for backend services
  - `test_chat_service.py`: Tests for the chat service
  - `test_evaluation_service.py`: Tests for the evaluation service

## Running Tests

You can run the tests using the provided script in the `scripts` directory:

```bash
# Run all tests
python scripts/run_tests.py

# Run tests with verbose output
python scripts/run_tests.py --verbose

# Run only evaluation service tests
python scripts/run_tests.py --evaluation

# Run a specific test file
python scripts/run_tests.py --path app/tests/services/test_evaluation_service.py
```

## Evaluation Service Tests

The evaluation service tests cover:

1. **Metrics Calculation**:
   - Numerical value extraction from complex financial answers
   - Exact match calculation
   - Numerical accuracy calculation
   - Percentage error calculation
   - Financial accuracy calculation
   - Retrieval quality metrics
   - Format adherence metrics

2. **Document Evaluation**:
   - Extracting QA pairs from documents
   - Evaluating documents with simple and complex financial questions
   - Retry mechanism for document evaluation

3. **Evaluation Flow**:
   - Running evaluations
   - Batch document processing
   - Retrieving evaluation results
   - Comparing evaluations

4. **Complex Financial Scenarios**:
   - Percentage change calculations
   - Compound annual growth rate (CAGR) calculations
   - Financial metrics with detailed workings and citations

## Adding New Tests

When adding new tests:

1. Follow the existing pattern of test classes and methods
2. Use descriptive test method names
3. Add appropriate assertions to verify functionality
4. For async tests, use the `@pytest.mark.asyncio` decorator
5. Mock external dependencies using `unittest.mock`

## Test Data

The tests include realistic financial QA examples with:
- Complex calculations with step-by-step workings
- Percentage calculations
- Currency values
- Citations and references
- Different question types (factoid, calculation) 