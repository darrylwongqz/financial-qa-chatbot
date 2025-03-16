# Financial QA Chatbot Evaluation Results

## Overview

This document presents a comprehensive analysis of our Financial QA chatbot's performance across different configurations. The evaluation framework tests the chatbot's ability to answer financial questions accurately, with a focus on numerical precision, relevance, and proper citation practices.

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

This uses the same calculation as Numerical Accuracy but is tracked separately to highlight performance specifically on financial values. In our current implementation, the tolerance is the same (1%), but this allows us to adjust financial accuracy requirements independently in the future.

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
   - Typically results in the slowest response times.

2. **Balanced Retrieval**:
   - Provides moderate error rates and numerical accuracy.
   - Offers a good compromise between response time and answer quality.
   - **Recommended default option** for most use cases, as performance metrics are surprisingly close to accurate retrieval (within 5-7% for most metrics) while providing significantly faster response times.

3. **Fast Retrieval**:
   - Results in the highest error rates and lowest numerical accuracy.
   - Particularly impacts calculation questions.
   - Provides the fastest response times.

### Question Type Analysis

Performance varies by question type:

- **Extraction Questions**:
  - GPT‑4 with accurate retrieval achieves near-perfect performance.
  - High citation rates are observed, reflecting effective extraction from source documents.

- **Calculation Questions**:
  - These show higher error rates overall.
  - GPT‑4's enhanced reasoning and complete presentation of calculation steps are particularly evident.
  - Fast retrieval profiles show the greatest struggle in this category.

- **Other Questions**:
  - Error rates and performance lie between extraction and calculation questions.
  - GPT‑4 handles these significantly better than GPT‑3.5‑turbo.

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

## Recommended Configurations

Based on the evaluation:

- **For High-Precision Applications** (e.g., financial analysis):
  - **Model**: GPT‑4  
  - **Retrieval Profile**: Accurate  
  - *Rationale*: Delivers the highest numerical and financial accuracy, minimal error rate, and strong calculation transparency.

- **For Most Use Cases** (recommended default):
  - **Model**: GPT‑4  
  - **Retrieval Profile**: Balanced  
  - *Rationale*: Offers performance metrics surprisingly close to accurate retrieval (within 5-7% for most metrics) while providing significantly faster response times. This represents the optimal balance between accuracy and speed for most financial Q&A scenarios.

- **For Speed-Critical Applications**:
  - **Model**: GPT‑3.5‑turbo  
  - **Retrieval Profile**: Fast  
  - *Rationale*: Although error rates are higher due to lower token limits and reduced context, it provides faster response times for real-time applications.

- **For Budget-Constrained Scenarios**:
  - **Model**: GPT‑3.5‑turbo  
  - **Retrieval Profile**: Accurate  
  - *Rationale*: Prioritizes high-quality retrieval over model size when cost is a factor.

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

## Conclusion

Our Financial QA chatbot demonstrates robust performance across various configurations. GPT‑4, paired with accurate retrieval, delivers the best results in terms of numerical precision, relevance, and calculation transparency. The evaluation highlights that retrieval quality is critical and that even small model limitations—such as GPT‑3.5‑turbo's lower token limit—can significantly impact performance.

For financial applications where precision is paramount, GPT‑4 with an accurate retrieval profile is strongly recommended. However, our analysis reveals that the balanced retrieval profile offers performance metrics surprisingly close to accurate retrieval while providing significantly faster response times, making it the recommended default option for most use cases. In scenarios where response speed is critical, a fast retrieval approach using GPT‑3.5‑turbo may be acceptable, albeit with some loss in accuracy.

Overall, these insights and recommendations will help guide future enhancements of our Financial QA system.
