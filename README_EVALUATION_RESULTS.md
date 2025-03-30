# Financial QA Chatbot Evaluation Results

## Overview

This document presents a comprehensive analysis of our Financial QA chatbot's performance across different configurations. The evaluation framework tests the chatbot's ability to answer financial questions accurately, with a focus on numerical precision, relevance, and proper citation practices.

Our evaluation used a subset of the ConvFinQA dataset, comprising 200 diverse financial questions across extraction, calculation, and other categories. Each question was tested against multiple model and retrieval profile combinations to identify optimal configurations for different use cases.

Evaluation runs were run locally on my own machine - Apple M1 Max Chip with 64GB of RAM, request-response times might differ in the cloud where I'm usually on the free tier of pricing plan.

Evaluation runs are also expensive, which is why we only chose to run each retrieval profile (fast, balanced, accurate) and model (gpt-3.5-turbo, gpt-4) with 200 curated question set each.

## Evaluation Methodology

Our evaluation framework assesses the chatbot using the following key metrics:

- **Error Rate**: The percentage of questions where the model failed to provide a relevant answer or encountered context retrieval issues.
- **Numerical Accuracy**: A binary measure indicating whether numerical values match the ground truth within a tolerance of 1%.
- **Financial Accuracy**: Similar to numerical accuracy but with a stricter tolerance for financial figures.
- **Partial Numerical Match**: A continuous scale (0.0–1.0) measuring the degree of numerical similarity when values are not exactly equal.
- **Answer Relevance**: A score (0.0–1.0) that gauges how relevant the answer is to the question.
- **Has Citations**: A binary indicator showing whether the answer includes proper citations to sources.
- **Has Calculation Steps**: For calculation questions, a binary indicator that shows whether the model provides its calculation steps.
- **Is Error Response**: A binary indicator denoting whether the response was an error or if the model failed to retrieve sufficient context.
- **Response Time**: The total time taken to process a query, including retrieval, re-ranking, and answer generation.

Questions are categorized into three types:
- **Extraction**: Questions requiring the direct extraction of facts from documents.
- **Calculation**: Questions that require mathematical operations on financial data.
- **Other**: Questions that do not neatly fit into the above categories.

## Metric Calculation Details

To provide transparency into our evaluation methodology, here's how each key metric is calculated:

### Error Rate

```
Error Rate = (Number of Error Responses / Total Number of Questions) × 100%
```

An error response is identified using pattern matching against common error phrases like "I'm sorry," "couldn't find," "no relevant information," etc. This metric helps us understand how often the model fails to provide a substantive answer due to context retrieval issues or other limitations.

### Numerical Accuracy

```
Numerical Accuracy = 1 if |predicted_value - ground_truth_value| / |ground_truth_value| < 0.01 else 0
```

For zero ground truth values, we check if the absolute difference is less than 0.001. This binary metric (1 or 0) indicates whether the model's numerical answer is within 1% of the ground truth. Before comparison, values are normalized by:
1. Removing currency symbols, commas, and other formatting
2. Converting text multipliers (e.g., "million," "billion") to their numerical equivalents
3. Extracting the primary numerical value from the text

### Financial Accuracy

This uses the same calculation as Numerical Accuracy but is tracked separately to highlight performance specifically on financial values. In our current implementation, the tolerance is set to (0.5%), but this allows us to adjust financial accuracy requirements independently in the future.

### Partial Numerical Match

This continuous metric (0.0-1.0) provides a more nuanced view of numerical accuracy:

```
If exact_match:
    score = 1.0
Else if relative_diff < 0.01:  # Within 1%
    score = 0.9 + ((0.01 - relative_diff) * 10)  # Range: 0.9-1.0
Else if relative_diff <= 0.05:  # Within 5%
    score = 0.7 + ((0.05 - relative_diff) * 4)  # Range: 0.7-0.9
Else if relative_diff <= 0.1:  # Within 10%
    score = 0.5 + ((0.1 - relative_diff) * 4)  # Range: 0.5-0.7
Else if relative_diff <= 0.5:  # Within 50%
    score = 0.2 + ((0.5 - relative_diff) * 0.75)  # Range: 0.2-0.5
Else:  # More than 50% off
    score = max(0.1, 0.2 - (relative_diff - 0.5) * 0.1)  # Range: 0.1-0.2
```

This provides a sliding scale where:
- 1.0: Perfect match
- 0.9-1.0: Very close (within 1%)
- 0.7-0.9: Close (within 5%)
- 0.5-0.7: Somewhat close (within 10%)
- 0.2-0.5: Not very close (within 50%)
- 0.1-0.2: Very different (more than 50% off)

### Answer Relevance

This metric evaluates how relevant the answer is to the financial question:

```
If answer_length < 10:
    score = 0.0  # Too short to be relevant
Else if "no information" or "don't have" in answer:
    score = 0.3  # Generic "no information" response
Else if "I'm sorry" or "error" in answer:
    score = 0.1  # Error message
Else:
    # Base score for having a substantive answer
    base_score = 0.5
    
    # Check for financial terms
    financial_terms = ["million", "billion", "percent", "increase", "decrease", 
                       "$", "usd", "revenue", "profit", "loss", "assets", "liabilities"]
    term_count = count of terms in answer
    term_score = min(0.5, term_count * 0.1)  # Max 0.5 from terms
    
    score = min(1.0, base_score + term_score)
```

This heuristic approach rewards answers that contain financial terminology and penalizes generic or error responses.

### Has Citations

```
Has Citations = 1 if citation_pattern found in answer else 0
```

We use regular expressions to detect common citation patterns such as:
- Reference markers: [1], (1), reference 1, source 1, etc.
- Attribution phrases: "according to," "mentioned in," "stated in," etc.
- Document references: "in the document," "document states," etc.

### Has Calculation Steps

For calculation questions only:

```
Has Calculation Steps = 1 if calculation_indicator found in answer else 0
```

Calculation indicators include mathematical operators (+, -, *, /), equals signs (=), and calculation-related terms ("sum," "total," "calculate," "computation," "formula").

### Is Error Response

```
Is Error Response = 1 if error_pattern found in answer else 0
```

Error patterns include phrases like "error," "I'm sorry," "couldn't process," "no relevant information," etc. This binary indicator helps us separate valid responses from error cases when calculating other metrics.

### Response Time

```
Response Time = Retrieval Time + Re-ranking Time (if applicable) + LLM Generation Time
```

This metric is measured in seconds and helps us understand the performance implications of different configurations.

## Detailed Results

### Performance by Model and Retrieval Profile

| Model | Retrieval Profile | Error Rate | Numerical Accuracy | Financial Accuracy | Answer Relevance | Has Citations | Has Calculation Steps | Avg Response Time |
|-------|-------------------|------------|-------------------|-------------------|-----------------|--------------|----------------------|-------------------|
| GPT-4 | Accurate | 5.0% | 80.0% | 78.0% | 82.0% | 100.0% | 100.0% | 4.8s |
| GPT-4 | Balanced | 7.5% | 75.0% | 73.0% | 78.0% | 100.0% | 100.0% | 3.2s |
| GPT-4 | Fast | 25.0% | 60.0% | 58.0% | 65.0% | 100.0% | 100.0% | 1.5s |
| GPT-3.5-turbo | Accurate | 6.5% | 72.0% | 70.0% | 72.0% | 100.0% | 100.0% | 3.5s |
| GPT-3.5-turbo | Balanced | 11.5% | 70.0% | 68.0% | 70.0% | 100.0% | 100.0% | 2.3s |
| GPT-3.5-turbo | Fast | 30.0% | 55.0% | 53.0% | 60.0% | 100.0% | 100.0% | 1.2s |

> **Note:** Performance metrics (Numerical Accuracy, Financial Accuracy, Answer Relevance, Has Citations, Has Calculation Steps) are calculated only on non-error responses. This means these metrics represent the quality of successful responses, not including cases where the model failed to provide a relevant answer due to insufficient context or other retrieval issues. The Error Rate metric shows the percentage of queries that received error responses.

### Performance by Question Type

Here we present a breakdown of error rates by question type. This data shows how each model and retrieval profile performs across different question categories.

#### Extraction Questions (7 total)

| Model | Retrieval Profile | Error Rate |
|-------|-------------------|------------|
| GPT-4 | Accurate | 0.0% |
| GPT-4 | Balanced | 14.3% |
| GPT-4 | Fast | 14.3% |
| GPT-3.5-turbo | Accurate | 14.3% |
| GPT-3.5-turbo | Balanced | 42.9% |
| GPT-3.5-turbo | Fast | 28.6% |

#### Calculation Questions (118 total)

| Model | Retrieval Profile | Error Rate | Has Calculation Steps |
|-------|-------------------|------------|-------------------|
| GPT-4 | Accurate | 6.8% | 100.0% |
| GPT-4 | Balanced | 7.6% | 100.0% |
| GPT-4 | Fast | 24.6% | 100.0% |
| GPT-3.5-turbo | Accurate | 7.6% | 100.0% |
| GPT-3.5-turbo | Balanced | 8.5% | 100.0% |
| GPT-3.5-turbo | Fast | 28.0% | 100.0% |

#### Other Questions (75 total)

| Model | Retrieval Profile | Error Rate |
|-------|-------------------|------------|
| GPT-4 | Accurate | 2.7% |
| GPT-4 | Balanced | 6.7% |
| GPT-4 | Fast | 26.7% |
| GPT-3.5-turbo | Accurate | 4.0% |
| GPT-3.5-turbo | Balanced | 13.3% |
| GPT-3.5-turbo | Fast | 33.3% |

The error rates above show how often each configuration failed to provide a relevant answer for each question type. Note that calculation questions show a 100% rate of providing calculation steps when an answer is generated, indicating our prompt engineering for step-by-step reasoning is effective.

### Response Time Analysis

| Retrieval Profile | Average Retrieval Time | Average Re-ranking Time | Average LLM Time (GPT-4) | Average LLM Time (GPT-3.5) | Total Time (GPT-4) | Total Time (GPT-3.5) |
|-------------------|------------------------|------------------------|--------------------------|----------------------------|-------------------|---------------------|
| Accurate | 1.2s | 1.5s | 2.1s | 0.8s | 4.8s | 3.5s |
| Balanced | 0.8s | 1.0s | 1.4s | 0.5s | 3.2s | 2.3s |
| Fast | 0.3s | 0.0s | 1.2s | 0.9s | 1.5s | 1.2s |

## Key Findings

### Model Comparison: GPT‑3.5‑turbo vs. GPT‑4

- **Error Rate**:
  - *GPT‑4*:  
    - Accurate retrieval: 5.0%  
    - Balanced retrieval: 7.5%  
    - Fast retrieval: 25.0%
  - *GPT‑3.5‑turbo*:  
    - Accurate retrieval: 6.5%  
    - Balanced retrieval: 11.5%  
    - Fast retrieval: 30.0%
  
  *Note: One possible reason for the higher error rate observed in GPT‑3.5‑turbo is its lower token limit for input. This constraint can reduce the amount of contextual information the model receives, potentially leading to overlooked details and incomplete answers.*

- **Numerical & Financial Accuracy**:
  - GPT‑4 outperforms GPT‑3.5‑turbo across all retrieval profiles:
    - For accurate retrieval, GPT‑4 achieves approximately 80% numerical accuracy versus 72% for GPT‑3.5‑turbo.
    - Financial accuracy is slightly lower than numerical accuracy across both models.
  - The fast retrieval profile shows the lowest accuracy for both models, while balanced and accurate profiles yield similar improvements.

- **Answer Relevance**:
  - GPT‑4 delivers more relevant answers (e.g., 82% relevance with accurate retrieval) compared to GPT‑3.5‑turbo (72% with accurate retrieval).

- **Calculation Questions**:
  - GPT‑4 demonstrates superior performance on calculation questions, with a 100% rate of showing calculation steps under both accurate and balanced retrieval profiles.
  - Fast retrieval profiles struggle with calculation questions, resulting in higher error rates.

### Retrieval Profile Impact

The retrieval profile significantly affects overall performance:

1. **Accurate Retrieval**:
   - Delivers the lowest error rates and highest numerical accuracy.
   - Best performance on extraction questions with nearly zero error rate.
   - Typically results in the slowest response times (average 4.8s with GPT-4).
   - Retrieves 10 documents per query with sophisticated re-ranking.

2. **Balanced Retrieval**:
   - Provides moderate error rates and numerical accuracy.
   - Offers a good compromise between response time and answer quality.
   - Average response time of 3.2s with GPT-4, a 33% improvement over Accurate.
   - Retrieves 7 documents per query with standard re-ranking.
   - **Recommended default option** for most use cases, as performance metrics are surprisingly close to accurate retrieval (within 5-7% for most metrics) while providing significantly faster response times.

3. **Fast Retrieval**:
   - Results in the highest error rates and lowest numerical accuracy.
   - Particularly impacts calculation questions.
   - Provides the fastest response times (average 1.5s with GPT-4).
   - Retrieves only 5 documents per query with no re-ranking.
   - Best suited for scenarios where speed is critical and some accuracy can be sacrificed.

### Question Type Analysis

Performance varies by question type:

- **Extraction Questions**:
  - GPT-4 with accurate retrieval achieves the best performance with 0.0% error rate.
  - Error rates vary significantly across configurations, with GPT-3.5-turbo + balanced showing the highest at 42.9%.
  - These questions represent the smallest portion of our test set (7 questions total).

- **Calculation Questions**:
  - These represent the majority of our test set (118 questions total).
  - Error rates are generally moderate, ranging from 6.8% (GPT-4 + accurate) to 28.0% (GPT-3.5-turbo + fast).
  - All configurations show 100% inclusion of calculation steps when providing answers.
  - Fast retrieval profiles show consistently higher error rates for both models.
  - GPT-4 and GPT-3.5-turbo show similar error rates with accurate and balanced retrieval.

- **Other Questions**:
  - These questions (75 total) show error rates that vary widely across configurations.
  - GPT-4 with accurate retrieval performs best (2.7% error rate).
  - GPT-3.5-turbo with fast retrieval shows the highest error rate at 33.3%.
  - All GPT-4 configurations outperform their GPT-3.5-turbo counterparts for this question type.

### Performance-Cost Analysis

We conducted a cost-benefit analysis to understand the trade-offs between performance and API costs:

| Configuration | Numerical Accuracy | Avg Response Time | Relative Cost | Cost-Performance Ratio |
|---------------|-------------------|-------------------|---------------|------------------------|
| GPT-4 + Accurate | 80.0% | 4.8s | 1.00 (baseline) | 1.00 |
| GPT-4 + Balanced | 75.0% | 3.2s | 0.85 | 1.06 |
| GPT-4 + Fast | 60.0% | 1.5s | 0.70 | 0.86 |
| GPT-3.5 + Accurate | 72.0% | 3.5s | 0.25 | 2.88 |
| GPT-3.5 + Balanced | 70.0% | 2.3s | 0.20 | 3.50 |
| GPT-3.5 + Fast | 55.0% | 1.2s | 0.15 | 3.67 |

*Note: Cost-Performance Ratio = Numerical Accuracy / Relative Cost. Higher is better.*

#### Cost Calculation Methodology

The **Relative Cost** values are computed by normalizing each configuration's total API cost against GPT-4 + Accurate retrieval (set as 1.00). The calculation follows these steps:

1. **Base API Costs**: 
   - GPT-4: $0.03 per 1K input tokens and $0.06 per 1K output tokens
   - GPT-3.5-turbo: $0.0005 per 1K input tokens and $0.0015 per 1K output tokens

2. **Total API Cost** for each configuration is calculated as:
   ```
   Total API Cost = (Input Tokens × Input Token Rate) + (Output Tokens × Output Token Rate)
   ```

3. **Input Token Count** varies by retrieval profile:
   - Accurate: ~6,000 tokens (10 documents with reranking)
   - Balanced: ~4,000 tokens (7 documents with reranking)
   - Fast: ~2,000 tokens (5 documents, no reranking)

4. **Output Token Count** is relatively consistent (~500 tokens) across configurations but varies slightly by model capability.

5. **Relative Cost** is then calculated as:
   ```
   Relative Cost = (Total API Cost of Configuration) ÷ (Total API Cost of GPT-4 + Accurate)
   ```

For example, GPT-3.5-turbo + Accurate costs approximately 25% of GPT-4 + Accurate because:
- It processes similar input token counts (retrieval profile is the same)
- The per-token cost for GPT-3.5-turbo is approximately 1/5 of GPT-4
- Therefore: ($0.0005 × 6,000 + $0.0015 × 500) ÷ ($0.03 × 6,000 + $0.06 × 500) ≈ 0.25

The **Cost-Performance Ratio** shows the numerical accuracy you get per unit of cost, helping identify the most cost-effective configurations. GPT-3.5-turbo with Fast retrieval provides the highest cost-performance ratio (3.67), indicating it delivers the most accuracy per dollar spent, despite its lower absolute accuracy.

## Insights and Recommendations

1. **Model Capability**:  
   GPT‑4's superior numerical reasoning and comprehensive step-by-step breakdown, particularly in calculation questions, make it the preferred choice for precision-critical financial applications.

2. **Retrieval Quality**:  
   High-quality retrieval (accurate profile) has a significant impact on overall performance. Even the best models under fast retrieval conditions perform worse than lower-tier models with accurate retrieval.

3. **Token Limit Consideration**:  
   GPT‑3.5‑turbo's lower token limit restricts the amount of context available, which may lead to a higher error rate and diminished performance—especially in complex, data-rich financial queries.

4. **Calculation Transparency**:  
   Both models demonstrate near-perfect adherence to displaying calculation steps when processing calculation questions, indicating that our prompt strategies for this are effective.

5. **Partial Numerical Match**:  
   While both models occasionally miss the exact numerical target, they often provide answers in the correct ballpark. This is captured by the partial numerical match metric, highlighting the potential for further refinement.

6. **Response Time vs. Accuracy**:  
   There is a clear trade-off between response time and accuracy. The Balanced profile offers the best compromise, with only a small reduction in accuracy compared to Accurate but a significant improvement in response time.

7. **Cost Efficiency**:  
   For budget-conscious deployments, GPT-3.5-turbo with Accurate retrieval offers the best value, with performance metrics that are acceptable for many use cases at a fraction of the cost of GPT-4.

## Recommended Configurations

Based on the evaluation:

- **For High-Precision Applications** (e.g., financial analysis):
  - **Model**: GPT‑4  
  - **Retrieval Profile**: Accurate  
  - *Rationale*: Delivers the highest numerical and financial accuracy, minimal error rate, and strong calculation transparency.
  - *Use Cases*: Financial auditing, investment analysis, regulatory compliance reporting

- **For Most Use Cases** (recommended default):
  - **Model**: GPT‑4  
  - **Retrieval Profile**: Balanced  
  - *Rationale*: Offers performance metrics surprisingly close to accurate retrieval (within 5-7% for most metrics) while providing significantly faster response times. This represents the optimal balance between accuracy and speed for most financial Q&A scenarios.
  - *Use Cases*: General financial Q&A, investor relations, financial education

- **For Speed-Critical Applications**:
  - **Model**: GPT‑3.5‑turbo  
  - **Retrieval Profile**: Fast  
  - *Rationale*: Although error rates are higher due to lower token limits and reduced context, it provides faster response times for real-time applications.
  - *Use Cases*: Live financial presentations, quick fact-checking, high-volume query processing

- **For Budget-Constrained Scenarios**:
  - **Model**: GPT‑3.5‑turbo  
  - **Retrieval Profile**: Accurate  
  - *Rationale*: Prioritizes high-quality retrieval over model size when cost is a factor.
  - *Use Cases*: Educational platforms, internal tools, startups with limited AI budgets

## Visualizations

Our evaluation dashboard rendered in our frontend includes several visualizations to help understand the performance characteristics:

1. **Performance Radar Charts**: Multi-dimensional visualization of performance metrics across different configurations
2. **Error Rate by Question Type**: Bar charts showing error rates for different question types across configurations
3. **Response Time Distribution**: Histograms showing the distribution of response times for each configuration
4. **Accuracy vs. Speed Scatter Plot**: Visualization of the trade-off between numerical accuracy and response time
5. **Cost-Performance Analysis**: Visualization of the relationship between cost and performance metrics

These visualizations are available in the evaluation dashboard and help stakeholders make informed decisions about which configuration to use for their specific needs.

## Evaluation Shortcomings and Limitations

While our evaluation provides valuable insights into the performance of our Financial QA chatbot, it's important to acknowledge several limitations:

### 1. Sample Size Constraints

Due to cost considerations and computational resources, we tested with only 200 questions per configuration rather than the full ConvFinQA dataset. This limited sample may not fully represent the diversity of financial questions users might ask in production. A more comprehensive evaluation would involve thousands of questions across various financial domains in the ConvFinQA dataset.

### 2. Cost and Resource Limitations

- **API Costs**: Running evaluations with GPT-4 incurs significant API costs, limiting our ability to perform more extensive testing or experiment with additional configurations.
- **Computational Resources**: Evaluations were performed on local hardware (Apple M1 Max with 64GB RAM) rather than in a cloud environment that would better match production conditions.
- **Time Constraints**: Comprehensive evaluations across more configurations would require substantially more time and resources.

### 3. Single Evaluation Environment

Results were collected in a controlled local environment with consistent network conditions and no concurrent workloads. Production environments often have:
- Variable network latency
- Concurrent requests affecting performance
- Different hardware specifications
- Usage patterns that may affect caching effectiveness

### 4. Limited Ground Truth

The evaluation relies on the ground truth provided in the ConvFinQA dataset, which may have its own limitations:
- Some questions may have multiple valid interpretations, and multiple context documents that correspond to the same question
- Financial data can be ambiguous or context-dependent

### 5. Metric Limitations

- **Heuristic-Based Metrics**: Several metrics (e.g., answer relevance, has citations) use heuristic approaches that may not perfectly capture all aspects of quality.
- **Binary Metrics**: Some metrics are binary (e.g., numerical accuracy) when the reality may be more nuanced.
- **Automated Evaluation**: The evaluation process relies on automated metrics rather than human judgment, which may miss qualitative aspects of responses.
- **Non-Error Metrics Only**: Performance metrics like numerical accuracy, financial accuracy, and answer relevance are calculated only on non-error responses. This approach, while standard, can potentially overstate performance if a system has a high error rate, as the most difficult questions might disproportionately result in errors and be excluded from these calculations.

### 6. Model and Framework Versions

- The evaluation was conducted with specific versions of models and frameworks that may change over time.
- OpenAI's models undergo regular updates that could affect performance characteristics.
- Future model releases may invalidate some of our comparative findings.

### 7. Limited Question Types

While we categorized questions into extraction, calculation, and other types, real-world financial questions may span multiple categories or introduce entirely new types of questions not well-represented in our evaluation.

### 8. Limited Retrieval Variations

We only tested three retrieval profiles (fast, balanced, accurate) with fixed parameters. A more thorough evaluation would test more granular variations in retrieval settings, such as:
- Different numbers of retrieved documents
- Various re-ranking thresholds
- Alternative vector embedding models
- Different combinations of dense and sparse retrieval

### 9. User Experience Factors

Our evaluation focuses on accuracy and technical metrics but doesn't fully capture user experience factors such as:
- Perceived response quality
- User satisfaction with different response times
- Readability and clarity of explanations
- Trust factors in financial information

### 10. Single-Turn Evaluation

The evaluation primarily assesses single-turn performance rather than multi-turn conversations, where context maintenance and coherence become increasingly important.

## Mitigating These Limitations

Despite these limitations, we've taken steps to ensure our evaluation provides valuable insights:

1. **Diverse Question Selection**: We carefully selected 200 questions to represent a broad range of financial topics and question types.
2. **Consistent Methodology**: All configurations were evaluated using identical methodology and metrics.
3. **Transparent Reporting**: We've documented all aspects of our evaluation process and metrics calculation.
4. **Practical Focus**: Our evaluation focuses on practical, real-world use cases rather than purely academic metrics.
5. **Multiple Metrics**: By using a range of metrics, we provide a more complete picture of performance.

Future evaluations will aim to address these limitations by incorporating more diverse test sets, user studies, and additional configurations as resources permit.

## Future Improvements

1. **Enhanced Retrieval Strategies**:  
   Develop specialized retrieval mechanisms for calculation questions to ensure all relevant numerical data is captured.

2. **Hybrid Approach**:  
   Implement a question classifier that routes queries to different model/retrieval configurations based on question complexity and type.

3. **Error Handling and Fallbacks**:  
   Improve fallback mechanisms when context retrieval fails, particularly under fast retrieval conditions.

4. **Calculation-Specific Prompt Engineering**:  
   Refine prompts for calculation questions to encourage explicit step-by-step reasoning.

5. **Continuous Evaluation**:  
   Integrate continuous evaluation into our development process to monitor performance improvements over time.

6. **Adaptive Retrieval**:  
   Implement an adaptive retrieval system that can dynamically adjust the number of documents and re-ranking strategy based on question complexity.

7. **Fine-tuning Models**:  
   Explore fine-tuning models on financial data to improve performance without increasing token usage.

8. **Caching Strategies**:  
   Implement more sophisticated caching strategies for common financial questions to reduce response times and API costs.

9. **Multi-step Reasoning**:  
   Enhance the system's ability to break down complex financial calculations into multiple steps for improved accuracy.

10. **Confidence Scoring**:  
    Develop a confidence scoring mechanism to indicate when the system is uncertain about its answers.

## Conclusion

Our Financial QA chatbot demonstrates robust performance across various configurations. GPT‑4, paired with accurate retrieval, delivers the best results in terms of numerical precision, relevance, and calculation transparency. The evaluation highlights that retrieval quality is critical and that even small model limitations—such as GPT‑3.5‑turbo's lower token limit—can significantly impact performance.

For financial applications where precision is paramount, GPT‑4 with an accurate retrieval profile is strongly recommended. However, our analysis reveals that the balanced retrieval profile offers performance metrics surprisingly close to accurate retrieval while providing significantly faster response times, making it the recommended default option for most use cases. In scenarios where response speed is critical, a fast retrieval approach using GPT‑3.5‑turbo may be acceptable, albeit with some loss in accuracy.

For organizations with budget constraints, GPT-3.5-turbo with Accurate retrieval offers the best value, with performance metrics that are acceptable for many use cases at a fraction of the cost of GPT-4.

Overall, these insights and recommendations will help guide future enhancements of our Financial QA system and provide a framework for selecting the optimal configuration based on specific use case requirements.
