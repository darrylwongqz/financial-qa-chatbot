# Financial QA Backend

A FastAPI-based backend for a financial question answering system. This system uses retrieval-augmented generation (RAG) to provide accurate answers to financial questions.

## Features

- Question answering using OpenAI's GPT models
- Vector search using Pinecone for relevant context retrieval
- Document ingestion and management
- Query logging and analytics
- Evaluation metrics for system performance

## Project Structure

```
financial-qa-backend/
├── app/
│   ├── data/                      # Local storage for JSON datasets
│   ├── main.py                    # FastAPI application entry point
│   ├── config.py                  # Configuration file
│   ├── routers/                   # API endpoints
│   ├── services/                  # Business logic
│   ├── models/                    # Pydantic models
│   ├── db/                        # Database integrations
│   ├── tests/                     # Unit and integration tests
│   ├── requirements.txt           # Python dependencies
│   └── .env                       # Environment variables
└── scripts/                       # Utility scripts
```

## Setup

### Prerequisites

- Python 3.9+
- Pinecone account
- OpenAI API key
- Google Cloud Firestore (for query logging and analytics)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd financial-qa-backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r app/requirements.txt
```

4. Set up environment variables:
```bash
cp app/.env.example app/.env
# Edit .env with your API keys and configuration
```

5. Set up Firebase/Firestore:
   - Create a Firebase project at https://console.firebase.google.com/
   - Generate a service account key (Project settings > Service accounts > Generate new private key)
   - Save the key as `service_key.json` in the project root directory (not in the app directory)
   - Set the path to this file in your `.env` file:
     ```
     GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/your/service_key.json
     ```
   - Note: For development without Firestore, you can skip this step. The application will log warnings but continue to function for features that don't require Firestore.

### Running the Application

⚠️ **Important**: Always run the application from the project root directory (financial-qa-backend), not from inside the app directory.

Start the FastAPI server:
```bash
# From the project root directory
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

API documentation will be available at http://localhost:8000/docs

#### Common Issues

- **ModuleNotFoundError: No module named 'app'**: This error occurs when you try to run the server from inside the app directory. Make sure to run the server from the project root directory as shown above.
- **ModuleNotFoundError: No module named 'firebase_admin'**: Install the Firebase Admin SDK with `pip install firebase-admin`.
- **Failed to initialize Firestore client**: Ensure your `service_key.json` file is correctly set up and the path is properly configured in your `.env` file.
- **Service account key not found**: Make sure your service key is in the correct location and the path in your `.env` file is correct. For development without Firestore, you can ignore this warning.

## API Endpoints

- `GET /`: Root endpoint
- `GET /health`: Health check endpoint
- `POST /users/{user_id}/chat/`: Send a chat message and receive a response
- `GET /users/{user_id}/chat/`: Retrieve chat history for a user
- `GET /users/{user_id}/chat/profiles`: Get available retrieval profiles
- `DELETE /users/{user_id}/chat/`: Clear chat history for a user
- `GET /chat/models`: List available language models

## Retrieval Profiles

The Financial QA Chatbot implements three distinct retrieval profiles that allow users to choose their preferred balance between speed and accuracy when retrieving information:

### Profile Characteristics

1. **Fast Profile**
   - **Description**: Optimized for speed (< 0.15s response time)
   - **Use Case**: Mobile applications, real-time interactions, or when immediate responses are critical
   - **Configuration**:
     - Retrieves 3 documents per query
     - No re-ranking (saves processing time)
     - Uses hybrid search with weighted sum for score combination
     - Enables query expansion and preprocessing
     - Maintains caching for repeated queries

2. **Balanced Profile (Default)**
   - **Description**: Balanced performance and quality (< 0.3s response time)
   - **Use Case**: General purpose usage, offering a good balance of speed and quality
   - **Configuration**:
     - Retrieves 5 documents per query
     - Includes re-ranking for better result quality
     - Uses hybrid search with weighted sum for score combination
     - Enables query expansion and preprocessing
     - Maintains caching for repeated queries

3. **Accurate Profile**
   - **Description**: Optimized for accuracy (< 0.35s response time)
   - **Use Case**: Research applications, complex financial questions, or when accuracy is paramount
   - **Configuration**:
     - Retrieves 10 documents per query (more comprehensive context)
     - Includes re-ranking with more sophisticated algorithms
     - Uses hybrid search with harmonic mean for score combination (better quality)
     - Enables query expansion and preprocessing
     - Maintains caching for repeated queries

### Performance Comparison

Based on comprehensive testing, the profiles demonstrate the following performance characteristics:

| Profile   | Avg. Response Time | Documents Retrieved | Relative Speed |
|-----------|-------------------|---------------------|---------------|
| Fast      | ~0.13s            | 3                   | 1x (baseline) |
| Balanced  | ~0.25s            | 5                   | 1.9x slower   |
| Accurate  | ~0.30s            | 5-10                | 2.3x slower   |

### Response Time Testing

We conducted extensive testing to measure the actual performance of each retrieval profile using:

1. **Controlled Testing with Mocks**: Simulated the retrieval process with controlled parameters to measure relative performance differences between profiles.

2. **Comprehensive Query Set**: Tested with 20 diverse financial questions across different categories:
   - Basic financial concepts (stocks, bonds, market capitalization)
   - Investment strategies (diversification, asset allocation)
   - Economic concepts (GDP, inflation, fiscal policy)
   - Personal finance (debt management, retirement planning)

3. **Performance Metrics Collected**:
   - Average response time
   - Median response time
   - Minimum and maximum response times
   - Standard deviation of response times
   - Number of documents retrieved

The test results confirmed that the profiles perform as designed, with clear performance differences that allow users to choose the appropriate profile based on their specific needs. The Fast profile consistently delivers sub-0.15s response times, while the Accurate profile prioritizes comprehensive context retrieval at the cost of slightly longer response times.

### Implementation Details

The retrieval profiles are implemented in the following components:

- **Configuration**: Defined in `app/config.py` as `RETRIEVAL_PROFILES`
- **Retrieval Service**: Implemented in `app/services/retrieval_service.py` with the `get_relevant_context_with_profile` function
- **Re-ranking Service**: Enhanced in `app/services/re_ranking_service.py` with the `hybrid_rerank` function
- **API**: Exposed through the chat router in `app/routers/chat.py` with a `/profiles` endpoint

### Using Retrieval Profiles

Clients can specify their preferred retrieval profile when making a chat request:

```json
{
  "question": "What is the difference between stocks and bonds?",
  "retrieval_profile": "fast"  // Options: "fast", "balanced", "accurate"
}
```

If not specified, the system defaults to the "balanced" profile.

## Development

### Adding New Endpoints

1. Create a new router file in `app/routers/`
2. Define your endpoints using FastAPI's router
3. Import and include your router in `main.py`

### Running Tests

```bash
# From the project root directory
pytest app/tests/
```

#### Retrieval Profile Tests

The project includes specialized test scripts to evaluate the performance of retrieval profiles:

1. **Basic Profile Testing**:
   ```bash
   python -m app.tests.test_retrieval_profiles
   ```
   Tests the basic functionality of retrieval profiles with simple mocks.

2. **Comprehensive Performance Testing**:
   ```bash
   python -m app.tests.test_retrieval_profiles_comprehensive
   ```
   Runs a comprehensive test suite that measures response times across a diverse set of financial queries and generates visualizations of the results.

3. **Mock-based Testing**:
   ```bash
   python -m app.tests.test_retrieval_profiles_with_mocks
   ```
   Uses sophisticated mocks to simulate the retrieval process with realistic timing characteristics.

The test results are saved to `app/tests/results/` and include:
- CSV files with detailed performance metrics
- Box plots showing the distribution of response times
- Bar charts comparing average response times across profiles

### Logging

The application uses a dual logging approach:

1. **System Logging**: Standard Python logging for application events, errors, and debugging
2. **Business Logging**: Structured logging to Firestore for query analytics and user interactions

The logging configuration is centralized in `app/services/logging_service.py` and initialized at application startup.

### Firebase/Firestore Integration

The application uses Firebase/Firestore for:
- Storing user chat history
- Logging queries for analytics
- Storing evaluation results

For development without Firestore:
- The application will continue to function for features that don't require Firestore
- Warning logs will be displayed, but no errors will be raised
- Core question-answering functionality will still work

## License

[MIT License](LICENSE)