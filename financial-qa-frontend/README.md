# Financial QA Chatbot

A sophisticated AI-powered chatbot application for answering financial questions with high accuracy and transparency.

![Financial QA Chatbot](https://via.placeholder.com/800x400?text=Financial+QA+Chatbot)

## Overview

The Financial QA Chatbot is a specialized AI assistant trained on the ConvFinQA dataset that can answer complex financial questions with precision. It leverages Retrieval Augmented Generation (RAG) to provide accurate, contextually relevant answers with proper citations to source documents.

### Key Features

- **Vector-Powered Knowledge Retrieval**: Utilizes Pinecone vector embeddings for precise answers from financial datasets
- **Configurable Intelligence Modes**: Three retrieval profiles (Fast, Balanced, Accurate) to balance speed and accuracy
- **Conversation Memory**: Maintains context across sessions for coherent discussions
- **Source Transparency**: Displays the source documents used to generate answers
- **Calculation Steps**: Shows detailed calculation steps for mathematical financial questions
- **Performance Metrics**: Comprehensive evaluation framework to measure accuracy and relevance

## Technology Stack

- **Frontend**: Next.js 15, React, TypeScript, Tailwind CSS
- **Authentication**: Clerk
- **Database**: Firebase Firestore
- **Vector Database**: Pinecone
- **AI Models**: OpenAI GPT-3.5-Turbo and GPT-4

## Getting Started

### Prerequisites

- Node.js 18.x or higher
- npm or yarn
- Firebase account
- Clerk account for authentication
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/darrylwongqz/financial-qa-chatbot.git
   cd financial-qa-chatbot/financial-qa-frontend
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
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Run the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

5. Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

## Usage

### Chat Interface

The main chat interface allows users to:
- Ask financial questions in natural language
- Select between different AI models (GPT-3.5-Turbo or GPT-4)
- Choose retrieval profiles (Fast, Balanced, Accurate) based on their needs
- View conversation history
- Clear chat history when needed

### Evaluation Dashboard

The evaluation dashboard provides insights into the performance of different model and retrieval profile combinations:
- Error rates and accuracy metrics
- Performance comparison across different question types
- Recommendations for optimal configurations
- Detailed methodology explanation

## Performance Evaluation

Our evaluation framework assesses the chatbot using the following key metrics:

- **Error Rate**: Percentage of questions where the model failed to provide a relevant answer
- **Numerical Accuracy**: Whether numerical values match the ground truth within a tolerance of 1%
- **Financial Accuracy**: Similar to numerical accuracy but with a stricter tolerance for financial figures
- **Answer Relevance**: A score (0.0–1.0) that gauges how relevant the answer is to the question
- **Has Citations**: Whether the answer includes proper citations to sources
- **Has Calculation Steps**: For calculation questions, whether the model provides its calculation steps

### Key Findings

- GPT-4 consistently outperforms GPT-3.5-Turbo across all metrics
- The Balanced retrieval profile offers performance close to the Accurate profile while being significantly faster
- Different question types (extraction, calculation, other) show varying performance characteristics

## Project Structure

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
├── public/                 # Static assets
└── evaluation-results/     # Evaluation results data
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ConvFinQA dataset for training data
- OpenAI for the underlying language models
- Next.js team for the amazing framework
