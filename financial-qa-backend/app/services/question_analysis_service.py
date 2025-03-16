"""
This module serves as a blueprint for future enhancements to the Financial QA chatbot.
It demonstrates how we could improve the chat service by implementing sophisticated
question analysis and classification capabilities.

Key Features that could be implemented:
1. Automatic detection of question types (calculation, comparison, trend analysis, etc.)
2. Identification of specific calculation requirements (percentage change, ROI, etc.)
3. Entity extraction for financial terms and metrics
4. Keyword analysis for better context understanding

This would enable:
- More targeted and accurate responses
- Automatic switching between different processing modes (e.g., calculation vs explanation)
- Better handling of complex financial queries
- Integration with specialized calculation tools and services

Example Usage:
```python
analyzer = QuestionAnalysisService()
result = analyzer.analyze_question("What was the percentage change in Tesla's stock price between 2022 and 2023?")
# Returns: QuestionAnalysisResult(
#     question_type=QuestionType.CALCULATION,
#     requires_calculation=True,
#     calculation_type=CalculationType.PERCENTAGE_CHANGE,
#     entities=["Tesla", "2022", "2023"],
#     keywords=["stock price", "percentage change"]
# )
```

Note: This is currently not integrated into the chat service but serves as a reference
for future improvements to enhance the bot's question understanding capabilities.
"""

import re
from enum import Enum, auto
from typing import Dict, List, Any, Optional, NamedTuple, Tuple

class CalculationType(Enum):
    """Enum representing different types of financial calculations."""
    PERCENTAGE_CHANGE = "percentage_change"
    SIMPLE_INTEREST = "simple_interest"
    COMPOUND_INTEREST = "compound_interest"
    NET_PRESENT_VALUE = "net_present_value"
    ROI = "roi"
    PRICE_TO_EARNINGS = "price_to_earnings"
    GENERAL = "general"
    STATISTICAL = "statistical"
    RATIO = "ratio"
    ARITHMETIC = "arithmetic"

class QuestionType(Enum):
    """Enum representing different types of financial questions."""
    CALCULATION = "calculation"  # Questions requiring numerical computation
    COMPARISON = "comparison"    # Questions comparing two or more values
    EXTRACTION = "extraction"    # Questions asking for specific data points
    TREND = "trend"             # Questions about patterns over time
    YES_NO = "yes_no"           # Questions with boolean answers
    EXPLANATION = "explanation"  # Questions requiring textual explanation
    OTHER = "other"             # Questions that don't fit other categories

class QuestionAnalysisResult(NamedTuple):
    """Result of question analysis."""
    question_type: QuestionType
    requires_calculation: bool
    calculation_type: Optional[CalculationType] = None
    entities: List[str] = []
    keywords: List[str] = []

class QuestionAnalysisService:
    """Service for analyzing financial questions to determine their type and calculation requirements."""
    
    def __init__(self):
        """Initialize the question analysis service."""
        self.calculation_patterns = {
            CalculationType.PERCENTAGE_CHANGE: [
                r"percentage change",
                r"percent(age)? (increase|decrease)",
                r"how much did .* (increase|decrease) by",
                r"what is the (increase|decrease) in percentage",
            ],
            CalculationType.SIMPLE_INTEREST: [
                r"simple interest",
                r"interest rate .* simple",
                r"calculate .* interest .* simple",
            ],
            CalculationType.COMPOUND_INTEREST: [
                r"compound interest",
                r"interest .* compound(ed|ing)?",
                r"calculate .* compound(ed|ing)? interest",
            ],
            CalculationType.NET_PRESENT_VALUE: [
                r"net present value",
                r"npv",
                r"present value .* cash flows?",
            ],
            CalculationType.ROI: [
                r"return on investment",
                r"roi",
                r"investment return",
                r"return .* investment",
            ],
            CalculationType.PRICE_TO_EARNINGS: [
                r"price[- ]to[- ]earnings",
                r"p/?e ratio",
                r"earnings multiple",
            ]
        }
    
    def classify_question(self, question: str) -> Tuple[QuestionType, Optional[CalculationType]]:
        """
        Classify the type of financial question and its calculation subtype if applicable.
        
        Args:
            question: The question to classify
            
        Returns:
            Tuple of (QuestionType, Optional[CalculationType])
        """
        question_lower = question.lower()
        
        # Check for specific calculation types first
        for calc_type, patterns in self.calculation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return QuestionType.CALCULATION, calc_type
        
        # If no specific calculation type is found, analyze general patterns
        analysis_result = self.analyze_question(question)
        if analysis_result.question_type == QuestionType.CALCULATION:
            return QuestionType.CALCULATION, self._identify_general_calculation_type(question_lower)
        
        return analysis_result.question_type, None
    
    def analyze_question(self, question: str) -> QuestionAnalysisResult:
        """
        Analyze a question to determine its type and whether it requires calculation.
        
        Args:
            question: The question to analyze
            
        Returns:
            QuestionAnalysisResult with question type and calculation requirements
        """
        # Convert to lowercase for easier pattern matching
        question_lower = question.lower()
        
        # Extract potential entities and keywords
        entities = self._extract_entities(question)
        keywords = self._extract_keywords(question_lower)
        
        # First check if it's a specific calculation type
        for calc_type, patterns in self.calculation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return QuestionAnalysisResult(
                        question_type=QuestionType.CALCULATION,
                        requires_calculation=True,
                        calculation_type=calc_type,
                        entities=entities,
                        keywords=keywords
                    )
        
        # Check for general calculation patterns
        if self._contains_calculation_pattern(question_lower):
            calc_type = self._identify_general_calculation_type(question_lower)
            return QuestionAnalysisResult(
                question_type=QuestionType.CALCULATION,
                requires_calculation=True,
                calculation_type=calc_type,
                entities=entities,
                keywords=keywords
            )
        
        # Check for comparison patterns
        elif self._contains_comparison_pattern(question_lower):
            return QuestionAnalysisResult(
                question_type=QuestionType.COMPARISON,
                requires_calculation=True,
                calculation_type=CalculationType.RATIO,
                entities=entities,
                keywords=keywords
            )
        
        # Check for trend analysis patterns
        elif self._contains_trend_pattern(question_lower):
            return QuestionAnalysisResult(
                question_type=QuestionType.TREND,
                requires_calculation=True,
                calculation_type=CalculationType.STATISTICAL,
                entities=entities,
                keywords=keywords
            )
        
        # Check for extraction patterns
        elif self._contains_extraction_pattern(question_lower):
            return QuestionAnalysisResult(
                question_type=QuestionType.EXTRACTION,
                requires_calculation=False,
                entities=entities,
                keywords=keywords
            )
        
        # Check for yes/no questions
        elif self._contains_yes_no_pattern(question_lower):
            return QuestionAnalysisResult(
                question_type=QuestionType.YES_NO,
                requires_calculation=False,
                entities=entities,
                keywords=keywords
            )
        
        # Check for explanation questions
        elif self._contains_explanation_pattern(question_lower):
            return QuestionAnalysisResult(
                question_type=QuestionType.EXPLANATION,
                requires_calculation=False,
                entities=entities,
                keywords=keywords
            )
        
        # Default to OTHER
        else:
            return QuestionAnalysisResult(
                question_type=QuestionType.OTHER,
                requires_calculation=False,
                entities=entities,
                keywords=keywords
            )
    
    def _identify_general_calculation_type(self, question: str) -> CalculationType:
        """Identify the general type of calculation needed."""
        # Financial calculations
        if re.search(r"(compound interest|present value|future value|annuity|perpetuity|npv|irr|payback period)", question):
            return CalculationType.GENERAL
        
        # Statistical calculations
        elif re.search(r"(average|mean|median|mode|standard deviation|variance)", question):
            return CalculationType.STATISTICAL
        
        # Ratio calculations
        elif re.search(r"(ratio|percentage|proportion)", question):
            return CalculationType.RATIO
        
        # Basic arithmetic
        elif re.search(r"(add|subtract|multiply|divide|sum|difference|product|quotient)", question):
            return CalculationType.ARITHMETIC
        
        # Default to general
        return CalculationType.GENERAL
    
    def _contains_calculation_pattern(self, question: str) -> bool:
        """Check if the question contains patterns indicating a calculation is needed."""
        calculation_patterns = [
            r"calculate",
            r"compute",
            r"what is the (total|sum|difference|product|quotient|ratio|percentage|rate|amount)",
            r"how much",
            r"how many",
            r"find the (value|total|sum|difference|product|quotient|ratio|percentage|rate|amount)",
            r"(add|subtract|multiply|divide)",
            r"(increase|decrease) (by|of)",
            r"(percent|percentage) (of|change|increase|decrease)",
            r"(total|sum|difference|product|quotient|ratio) of",
            r"average",
            r"mean",
            r"median",
            r"mode",
            r"standard deviation",
            r"variance",
            r"compound (interest|growth)",
            r"(annual|monthly|quarterly|daily) (rate|return|interest)",
            r"(present|future) value",
            r"depreciation",
            r"amortization",
            r"(profit|loss) margin",
            r"(gross|net|operating) (profit|income|revenue|margin)",
            r"(return on|cost of) (investment|equity|assets)",
            r"(debt|equity|leverage) ratio",
            r"(price|earnings) ratio",
            r"(market|book) value",
            r"(dividend|payout) (yield|ratio)",
            r"(current|quick|cash) ratio",
            r"(inventory|asset|accounts receivable) turnover",
            r"(days|weeks|months|years) (outstanding|receivable|payable)",
            r"(break-even|breakeven) (point|analysis)",
            r"(fixed|variable|total) cost",
            r"(contribution|profit) margin",
            r"(net present|internal rate of return|payback period|discounted cash flow) value"
        ]
        
        for pattern in calculation_patterns:
            if re.search(pattern, question):
                return True
        
        # Check for mathematical operators
        if re.search(r"[+\-*/^%]", question):
            return True
        
        # Check for numbers followed by mathematical terms
        if re.search(r"\d+\s*(percent|%|times|divided by|plus|minus|multiplied by)", question):
            return True
        
        return False
    
    def _contains_comparison_pattern(self, question: str) -> bool:
        """Check if the question contains patterns indicating a comparison is needed."""
        comparison_patterns = [
            r"compare",
            r"(higher|lower) than",
            r"(more|less) than",
            r"(greater|smaller) than",
            r"(increase|decrease) (from|compared to)",
            r"difference between",
            r"(better|worse) than",
            r"(outperform|underperform)",
            r"(exceed|fall short of)",
            r"(above|below) (average|median|benchmark)",
            r"(highest|lowest)",
            r"(maximum|minimum)",
            r"(best|worst)",
            r"(strongest|weakest)",
            r"(fastest|slowest) (growing|declining)",
            r"(most|least) (profitable|expensive|valuable|efficient)"
        ]
        
        for pattern in comparison_patterns:
            if re.search(pattern, question):
                return True
        
        return False
    
    def _contains_trend_pattern(self, question: str) -> bool:
        """Check if the question contains patterns indicating trend analysis is needed."""
        trend_patterns = [
            r"trend",
            r"(over time|over the years|over the period|over the quarter|over the month)",
            r"(historical|history|past) (performance|data|trend|growth|decline)",
            r"(growing|declining|increasing|decreasing) (trend|rate|pattern)",
            r"(upward|downward) (trend|movement|trajectory)",
            r"(accelerating|decelerating)",
            r"(consistent|inconsistent|steady|volatile) (growth|decline|performance)",
            r"(year-over-year|quarter-over-quarter|month-over-month) (change|growth|decline)",
            r"(annual|quarterly|monthly) (growth|decline) rate",
            r"(long-term|short-term) (trend|pattern|performance)",
            r"(seasonal|cyclical) (pattern|trend|variation)",
            r"(forecast|projection|prediction) (based on|using) (historical|past) (data|performance)"
        ]
        
        for pattern in trend_patterns:
            if re.search(pattern, question):
                return True
        
        return False
    
    def _contains_extraction_pattern(self, question: str) -> bool:
        """Check if the question contains patterns indicating data extraction is needed."""
        extraction_patterns = [
            r"what (is|was|are|were) the",
            r"(tell|show) me the",
            r"(provide|give) the",
            r"(list|enumerate)",
            r"(identify|specify)",
            r"(report|state)",
            r"(extract|pull)",
            r"(find|locate)",
            r"(retrieve|fetch)",
            r"(get|obtain)"
        ]
        
        for pattern in extraction_patterns:
            if re.search(pattern, question):
                return True
        
        return False
    
    def _contains_yes_no_pattern(self, question: str) -> bool:
        """Check if the question is a yes/no question."""
        yes_no_patterns = [
            r"^(is|are|does|do|has|have|can|could|should|would|will|did)",
            r"^(was|were)",
            r"true or false"
        ]
        
        for pattern in yes_no_patterns:
            if re.search(pattern, question):
                return True
        
        return False
    
    def _contains_explanation_pattern(self, question: str) -> bool:
        """Check if the question asks for an explanation."""
        explanation_patterns = [
            r"why",
            r"how (does|do|did)",
            r"explain",
            r"describe",
            r"elaborate",
            r"clarify",
            r"what (causes|caused)",
            r"what is the reason",
            r"what factors",
            r"what led to"
        ]
        
        for pattern in explanation_patterns:
            if re.search(pattern, question):
                return True
        
        return False
    
    def _extract_entities(self, question: str) -> List[str]:
        """Extract financial entities from the question."""
        # This is a simple implementation that could be enhanced with NER
        financial_entities = [
            r"\$\d+(?:,\d{3})*(?:\.\d{2})?",  # Dollar amounts
            r"€\d+(?:,\d{3})*(?:\.\d{2})?",  # Euro amounts
            r"£\d+(?:,\d{3})*(?:\.\d{2})?",  # Pound amounts
            r"¥\d+(?:,\d{3})*(?:\.\d{2})?",  # Yen amounts
            r"\d+(?:,\d{3})*(?:\.\d{2})?\s*%",  # Percentages
            r"\d{4}(?:-\d{2})?",  # Years or year-month
            r"Q[1-4]\s*\d{4}",  # Quarters (e.g., Q1 2023)
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*\d{4}",  # Months with year
            r"(?:stocks?|bonds?|shares?|securities)",  # Financial instruments
            r"(?:company|corporation|enterprise|business|firm)",  # Business entities
            r"(?:market|exchange|index|fund)",  # Financial markets
            r"(?:revenue|profit|loss|earnings|income|expense)",  # Financial metrics
            r"(?:asset|liability|equity|debt|capital)",  # Financial terms
            r"(?:dividend|interest|yield|return)",  # Investment terms
            r"(?:portfolio|investment|holding)",  # Investment vehicles
            r"(?:rate|ratio|percentage|margin)",  # Financial ratios
            r"(?:bull|bear|volatile|stable)",  # Market conditions
            r"(?:growth|decline|trend|pattern)",  # Market movements
            r"(?:fiscal|financial|accounting|tax) (?:year|quarter|period)",  # Time periods
            r"(?:balance sheet|income statement|cash flow|statement)",  # Financial statements
        ]
        
        entities = []
        for pattern in financial_entities:
            matches = re.finditer(pattern, question, re.IGNORECASE)
            entities.extend(match.group() for match in matches)
        
        return entities
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract important keywords from the question."""
        # This is a simple implementation that could be enhanced with NLP
        financial_keywords = [
            "profit", "loss", "revenue", "cost", "expense", "income",
            "asset", "liability", "equity", "debt", "capital",
            "dividend", "interest", "yield", "return",
            "market", "stock", "bond", "share", "security",
            "portfolio", "investment", "fund", "index",
            "growth", "decline", "trend", "volatility",
            "ratio", "percentage", "rate", "margin",
            "balance", "statement", "report", "analysis",
            "fiscal", "financial", "accounting", "tax",
            "quarter", "annual", "monthly", "yearly"
        ]
        
        keywords = []
        for keyword in financial_keywords:
            if keyword in question:
                keywords.append(keyword)
        
        return keywords 