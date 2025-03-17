# FinancialQA Chatbot

A sophisticated AI-powered financial question answering system with high accuracy and transparency, built using Retrieval Augmented Generation (RAG) technology.

[Try out the live version of the chatbot here](https://financial-qa-frontend-production.up.railway.app/)

If you want to skip to the chatbot evaluation results straight, please refer to [README_EVALUATION_RESULTS.md](README_EVALUATION_RESULTS.md).

A prettified version of the evaluation metrics is available [in the dashboard tab of the app](https://financial-qa-frontend-production.up.railway.app/dashboard)
Note that you need to log in with your google account before viewing the prettified metrics

## Overview

The FinancialQA Chatbot is a specialized AI assistant designed to answer complex financial questions with precision and transparency. It combines advanced vector search technology with state-of-the-art language models to provide accurate, contextually relevant answers with proper citations to source documents.

The system is built as a full-stack application with a FastAPI backend and a Next.js frontend, leveraging vector databases for efficient knowledge retrieval and modern authentication for secure user access.

## Key Features

- **Vector-Powered Knowledge Retrieval**: Advanced RAG with Pinecone vector embeddings delivers precise answers from vast financial datasets
- **Hybrid Search Technology**: Combines dense and sparse embeddings for optimal retrieval performance
- **Blazing Fast Responses**: Configurable retrieval profiles allow users to balance speed and accuracy based on their needs
- **Conversation Memory**: Smart conversations that maintain context across your entire session, creating truly coherent financial discussions
- **Source Transparency**: Displays the exact source documents used to generate answers, with proper citations
- **Calculation Steps**: Shows detailed calculation steps for mathematical financial questions, ensuring transparency and verifiability
- **Cross-Encoder Re-ranking**: Sophisticated re-ranking of retrieved documents for improved relevance
- **Performance Optimization**: Model pre-initialization, cache pre-warming, and GPU acceleration support

## Architecture

The application features a modern tech stack with clear separation of concerns:

### Backend (FastAPI)

The backend is built with FastAPI and implements a sophisticated RAG pipeline:

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
│   │   ├── llm_service.py         # Language model integration
│   │   ├── re_ranking_service.py  # Cross-encoder re-ranking
│   │   ├── question_analysis_service.py # Question analysis for improved retrieval
│   │   ├── evaluation_service.py  # Evaluation metrics and processing
│   │   └── logging_service.py     # Centralized logging configuration
│   ├── models/                    # Pydantic models and DTOs
│   ├── db/                        # Database integrations
│   │   ├── firebase_init.py       # Firebase initialization
│   │   ├── firestore.py           # Firestore database operations
│   │   └── pinecone_db.py         # Pinecone vector database operations
│   ├── utils/                     # Utility functions
│   └── tests/                     # Unit and integration tests
└── scripts/                       # Utility scripts
    └── run_evaluation.py          # Script to run evaluations
```

#### Key Backend Components:

1. **Chat Service**: Manages user conversations, stores messages, and orchestrates the QA process
2. **Retrieval Service**: Implements vector search with both dense and sparse embeddings to find relevant financial documents
3. **LLM Service**: Interfaces with OpenAI models (GPT-3.5 and GPT-4) to generate answers
4. **Re-ranking Service**: Uses a cross-encoder model to improve retrieval quality
5. **Question Analysis Service**: Analyzes financial questions to improve retrieval
6. **Evaluation Service**: Comprehensive framework for evaluating system performance

### Frontend (Next.js)

The frontend is built with Next.js and provides a modern, responsive user interface:

```
financial-qa-frontend/
├── app/                    # Next.js app directory
│   ├── (protected)/        # Protected routes requiring authentication
│   │   ├── chat/           # Chat interface page
│   │   └── dashboard/      # Evaluation dashboard page
│   ├── api/                # API routes
│   ├── sign-in/            # Authentication pages
│   └── sign-up/
├── components/             # React components
│   ├── ui/                 # UI components
│   ├── ChatInterface.tsx   # Main chat interface component
│   ├── InstructionsPanel.tsx # Instructions and context panel
│   └── ...
├── lib/                    # Utility functions and shared code
│   ├── contextStore.ts     # Zustand state management for context
│   ├── firebase.ts         # Firebase configuration
│   └── utils.ts            # Utility functions
├── public/                 # Static assets
└── evaluation-results/     # Evaluation results data
```

#### Key Frontend Components:

1. **Chat Interface**: Real-time chat with message history and context display
2. **Authentication**: User authentication via Clerk
3. **State Management**: Firebase Firestore for persistent storage and Zustand for client-side state
4. **UI Components**: Resizable panels, markdown rendering, and responsive design

## Retrieval Profiles

The system implements three distinct retrieval profiles that allow users to choose their preferred balance between speed and accuracy:
Note that the response times were logged with my local machine, Apple M1 Max Chip with 64GB of RAM, and might differ in the cloud.

### 1. Fast Profile

- **Purpose**: Optimized for speed with improved accuracy
- **Response Time**: < 1.5s
- **Implementation Details**:
  - Retrieves 5 documents per query
  - Uses only dense vector search (no hybrid search)
  - No re-ranking (saves processing time)
  - Smaller batch size for processing
  - Minimal query preprocessing

### 2. Balanced Profile (Default)

- **Purpose**: Balanced performance and quality
- **Response Time**: < 3.2s
- **Implementation Details**:
  - Retrieves 7 documents per query
  - Uses hybrid search (dense + sparse vectors)
  - Includes re-ranking for better result quality
  - Medium batch size for processing
  - Standard query preprocessing

### 3. Accurate Profile

- **Purpose**: Optimized for accuracy
- **Response Time**: < 4.8s
- **Implementation Details**:
  - Retrieves 10 documents per query
  - Uses hybrid search with optimized weights
  - Includes sophisticated re-ranking algorithms
  - Larger batch size for processing
  - Advanced query preprocessing and expansion
  - Higher re-ranking threshold

## Technical Stack

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.9+
- **Vector Database**: Pinecone
- **Document Storage**: Firebase/Firestore
- **LLM Integration**: OpenAI API (GPT-3.5-Turbo, GPT-4)
- **Re-ranking**: Cross-encoder models
- **Containerization**: Docker

### Frontend
- **Framework**: Next.js 15
- **Language**: TypeScript
- **UI**: React, Tailwind CSS, shadcn/ui
- **Authentication**: Clerk
- **State Management**: Zustand
- **Database**: Firebase Firestore
- **Animation**: Framer Motion

## Performance Evaluation

Our evaluation framework assesses the chatbot using the following key metrics:

- **Error Rate**: Percentage of questions where the model failed to provide a relevant answer
- **Numerical Accuracy**: Whether numerical values match the ground truth within a tolerance of 1%
- **Financial Accuracy**: Similar to numerical accuracy but with a stricter tolerance for financial figures
- **Answer Relevance**: A score (0.0–1.0) that gauges how relevant the answer is to the question
- **Has Citations**: Whether the answer includes proper citations to sources
- **Has Calculation Steps**: For calculation questions, whether the model provides its calculation steps
- **Response Time**: Time taken to generate a complete response

### Key Findings

- **GPT-4 with Accurate Retrieval**: Highest performance (80% numerical accuracy, 5% error rate)
- **GPT-4 with Balanced Retrieval**: Recommended default configuration (75% numerical accuracy, 7.5% error rate)
- **GPT-3.5-Turbo with Accurate Retrieval**: Budget option (72% numerical accuracy, 6.5% error rate)

For a comprehensive breakdown of the evaluation methodology and detailed results, please refer to [README_EVALUATION_RESULTS.md](README_EVALUATION_RESULTS.md).

## Setup and Installation

### Backend Setup

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
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000 with documentation at http://localhost:8000/docs

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd financial-qa-frontend
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Set up environment variables:
Create a `.env.local` file in the root directory with the following variables:
```
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key
CLERK_SECRET_KEY=your_clerk_secret_key
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

4. Run the development server:
```bash
npm run dev
# or
yarn dev
```

5. Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

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

## Advanced Features

### 1. Cross-Encoder Re-ranking

The system uses a cross-encoder model to re-rank retrieved documents, significantly improving the relevance of the context provided to the LLM:

- **Model**: Cross-encoder/ms-marco-MiniLM-L-6-v2
- **Caching**: LRU cache for efficient re-use of computed scores
- **Batch Processing**: Configurable batch size for optimal performance
- **Score Combination**: Hybrid scoring combining dense vector similarity and semantic relevance

### 2. Question Analysis

The question analysis service enhances retrieval by:

- **Query Expansion**: Adding relevant financial terms to the query
- **Query Classification**: Identifying question types (extraction, calculation, etc.)
- **Entity Recognition**: Identifying financial entities in the question
- **Temporal Analysis**: Detecting time periods and fiscal quarters

### 3. Performance Optimization

- **Model Pre-initialization**: Pre-loads models at startup
- **Cache Pre-warming**: Pre-computes common financial queries
- **GPU Acceleration**: Optional GPU support for faster inference
- **Hybrid Retrieval**: Combines dense and sparse embeddings for optimal results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Scaling Considerations

For deploying this application at scale to handle thousands of concurrent users, several architectural enhancements would be necessary:

### Backend Scaling

1. **Containerization and Orchestration**:
   - Deploy using Kubernetes for container orchestration
   - Implement horizontal pod autoscaling based on CPU/memory usage
   - Use a service mesh like Istio for advanced traffic management

2. **Database Scaling**:
   - Implement sharding for Pinecone vector database
   - Use Firestore in production mode with appropriate capacity provisioning
   - Add Redis caching layer for frequently accessed data

3. **API Gateway and Load Balancing**:
   - Implement an API gateway for rate limiting and request throttling
   - Use cloud load balancers for distributing traffic
   - Set up global CDN for static assets

4. **Asynchronous Processing**:
   - Move long-running tasks to background workers using Celery or similar
   - Implement message queues (RabbitMQ/Kafka) for handling high throughput
   - Add WebSockets for real-time updates without polling

### LLM Optimization

1. **Model Serving Infrastructure**:
   - Deploy dedicated model serving infrastructure (e.g., TorchServe, TensorRT)
   - Implement model quantization for faster inference
   - Consider fine-tuned smaller models for common question types

2. **Caching Strategy**:
   - Implement multi-level caching (in-memory, distributed, persistent)
   - Cache common queries and their results
   - Use semantic caching to handle similar but not identical queries

3. **Cost Management**:
   - Implement token usage budgeting and monitoring
   - Add fallback to cheaper models during peak loads
   - Optimize prompt templates to reduce token consumption

### Frontend Scaling

1. **Static Site Generation and CDN**:
   - Use Next.js static site generation for non-dynamic pages
   - Deploy to global CDN for low-latency access
   - Implement edge caching for API responses

2. **Progressive Web App Features**:
   - Add offline capabilities for core functionality
   - Implement background sync for offline queries
   - Optimize bundle size with code splitting

3. **Authentication at Scale**:
   - Use Clerk's enterprise features for high-volume authentication
   - Implement JWT token management with proper rotation
   - Add rate limiting for authentication endpoints

### Monitoring and Reliability

1. **Observability Stack**:
   - Implement distributed tracing (Jaeger/Zipkin)
   - Set up comprehensive logging with ELK stack
   - Add real-time monitoring dashboards with Grafana

2. **Reliability Engineering**:
   - Implement circuit breakers for external dependencies
   - Add graceful degradation for non-critical features
   - Set up automated failover and disaster recovery

3. **Performance Testing**:
   - Conduct regular load testing with realistic user patterns
   - Implement continuous performance benchmarking
   - Monitor and optimize response time percentiles (p95, p99)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ConvFinQA dataset for training data
- OpenAI for the underlying language models
- Pinecone for vector database technology
- Next.js team for the frontend framework
