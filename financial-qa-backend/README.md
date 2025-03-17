# Financial QA Backend

A FastAPI-based backend for a financial question answering system. This system uses retrieval-augmented generation (RAG) to provide accurate answers to financial questions.

## Overview

The Financial QA Backend powers the Financial QA Chatbot, providing intelligent responses to financial questions using state-of-the-art language models and vector search technology. The system is designed to be fast, accurate, and transparent, with configurable retrieval profiles to balance speed and accuracy based on user needs.

For a comprehensive breakdown of the evaluation methodology and results, please refer to [README_EVALUATION_RESULTS.md](https://github.com/darrylwongqz/financial-qa-chatbot/blob/main/README_EVALUATION_RESULTS.md).

## Features

- **Question Answering**: Generate accurate responses to financial questions using OpenAI's GPT models
- **Contextual Retrieval**: Find relevant financial information using vector search with Pinecone
- **Configurable Retrieval Profiles**: Choose between Fast, Balanced, and Accurate retrieval modes
- **Conversation Memory**: Maintain context across multiple questions in a session
- **Source Citations**: Include references to source documents in responses
- **Calculation Steps**: Show detailed steps for financial calculations
- **Comprehensive Evaluation**: Detailed metrics on model performance across different configurations

## Project Structure

```
financial-qa-backend/
├── app/
│   ├── data/                      # Local storage for datasets and evaluation results
│   │   ├── evaluation_results/    # Stored evaluation results
│   │   └── financial_data/        # Financial datasets
│   ├── main.py                    # FastAPI application entry point
│   ├── config.py                  # Configuration settings
│   ├── routers/                   # API endpoints
│   │   ├── chat.py                # Chat endpoints
│   │   └── evaluation.py          # Evaluation endpoints
│   ├── services/                  # Business logic
│   │   ├── chat_service.py        # Chat processing logic
│   │   ├── retrieval_service.py   # Context retrieval logic
│   │   ├── evaluation_service.py  # Evaluation metrics and processing
│   │   └── ...
│   ├── models/                    # Pydantic models
│   ├── db/                        # Database integrations
│   ├── tests/                     # Unit and integration tests
│   └── requirements.txt           # Python dependencies
└── scripts/                       # Utility scripts
    └── run_evaluation.py          # Script to run evaluations
```

## Setup

### Prerequisites

- Python 3.9+
- Pinecone account
- OpenAI API key
- Firebase project (optional, for query logging)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-qa-chatbot.git
cd financial-qa-backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp app/.env.example app/.env
# Edit .env with your API keys and configuration
```

### Running the Application

Start the FastAPI server:
```bash
# From the project root directory (financial-qa-backend)
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

API documentation will be available at http://localhost:8000/docs

## API Endpoints

### Chat Endpoints

- `POST /users/{user_id}/chat/`: Send a chat message and receive a response
  ```json
  {
    "question": "What was the percentage change in the net cash from operating activities from 2008-2009?",
    "model": "gpt-4",
    "retrieval_profile": "balanced",
    "temperature": 0.7,
    "max_tokens": 1000
  }
  ```

- `GET /users/{user_id}/chat/`: Retrieve chat history for a user
- `GET /users/{user_id}/chat/profiles`: Get available retrieval profiles
- `DELETE /users/{user_id}/chat/`: Clear chat history for a user

## Retrieval Profiles

The system implements three distinct retrieval profiles that allow users to choose their preferred balance between speed and accuracy:

### Profile Characteristics

1. **Fast Profile**
   - Optimized for speed
   - Retrieves 5 documents per query
   - No re-ranking (saves processing time)

2. **Balanced Profile (Default)**
   - Balanced performance and quality
   - Retrieves 7 documents per query
   - Includes re-ranking for better result quality

3. **Accurate Profile**
   - Optimized for accuracy
   - Retrieves 10 documents per query
   - Includes sophisticated re-ranking algorithms

## Evaluation Framework

Our evaluation framework assesses the chatbot using the following key metrics:

- **Error Rate**: Percentage of questions where the model failed to provide a relevant answer
- **Numerical Accuracy**: Whether numerical values match the ground truth within a tolerance of 1%
- **Financial Accuracy**: Similar to numerical accuracy but with a stricter tolerance for financial figures
- **Answer Relevance**: A score (0.0–1.0) that gauges how relevant the answer is to the question
- **Has Citations**: Whether the answer includes proper citations to sources
- **Has Calculation Steps**: For calculation questions, whether the model provides its calculation steps

### Key Findings

- **GPT-4 with Accurate Retrieval**: Highest performance (80% numerical accuracy, 5% error rate)
- **GPT-4 with Balanced Retrieval**: Recommended default configuration (75% numerical accuracy, 7.5% error rate)
- **GPT-3.5-Turbo with Accurate Retrieval**: Budget option (72% numerical accuracy, 6.5% error rate)

For detailed evaluation results and methodology, see [README_EVALUATION_RESULTS.md](https://github.com/darrylwongqz/financial-qa-chatbot/blob/main/README_EVALUATION_RESULTS.md).

## Running Evaluations

To run an evaluation:

```bash
# From the project root directory
python financial-qa-backend/scripts/run_evaluation.py --retrieval-profile accurate --model gpt-4 --limit 200
```

Parameters:
- `--retrieval-profile`: The retrieval profile to use (fast, balanced, accurate)
- `--model`: The model to use (gpt-3.5-turbo, gpt-4)
- `--limit`: The number of questions to evaluate

## Development

### Adding New Features

1. Implement the feature in the appropriate service module
2. Add any necessary API endpoints in the routers
3. Update the configuration if needed
4. Add tests for the new feature

## License

[MIT License](LICENSE)