import unittest
from decimal import Decimal
from unittest.mock import Mock, patch
from app.services.numerical_reasoning_service import NumericalReasoningService
from app.services.question_analysis_service import QuestionType, CalculationType

class TestNumericalReasoningService(unittest.TestCase):
    def setUp(self):
        self.service = NumericalReasoningService()
        
    def test_extract_numbers(self):
        """Test number extraction from text with various formats"""
        test_cases = [
            {
                "input": "The price increased from $100 to $150",
                "expected": [Decimal('100'), Decimal('150')]
            },
            {
                "input": "A 5% interest rate on €1,000 for 2 years",
                "expected": [Decimal('0.05'), Decimal('1000'), Decimal('2')]
            },
            {
                "input": "ROI of -15% on investment of £2,500.50",
                "expected": [Decimal('-0.15'), Decimal('2500.50')]
            },
            {
                "input": "¥1,234,567.89 with 3.5% compound interest",
                "expected": [Decimal('1234567.89'), Decimal('0.035')]
            }
        ]
        
        for case in test_cases:
            with self.subTest(input=case["input"]):
                result = self.service.extract_numbers(case["input"])
                self.assertEqual(len(result), len(case["expected"]))
                for actual, expected in zip(result, case["expected"]):
                    self.assertEqual(actual, expected)
                    
    def test_process_calculation_percentage_change(self):
        """Test percentage change calculation"""
        question = "What is the percentage change from 100 to 150?"
        
        # Mock the question analyzer
        self.service.question_analyzer.classify_question = Mock(return_value=(QuestionType.CALCULATION, CalculationType.PERCENTAGE_CHANGE))
        
        # Mock the financial formulas
        self.service.financial_formulas.percentage_change = Mock(return_value=Decimal('0.5'))
        
        result, explanation = self.service.process_calculation(question)
        
        self.assertEqual(result, Decimal('0.5'))
        self.assertIn("50.00%", explanation)
        
    def test_process_calculation_simple_interest(self):
        """Test simple interest calculation"""
        question = "Calculate simple interest on $1000 at 5% for 2 years"
        
        # Mock the question analyzer
        self.service.question_analyzer.classify_question = Mock(return_value=(QuestionType.CALCULATION, CalculationType.SIMPLE_INTEREST))
        
        # Mock the financial formulas
        self.service.financial_formulas.simple_interest = Mock(return_value=Decimal('100'))
        
        result, explanation = self.service.process_calculation(question)
        
        self.assertEqual(result, Decimal('100'))
        self.assertIn("principal 1000", explanation)
        self.assertIn("5.00%", explanation)
        self.assertIn("2 years", explanation)
        
    def test_process_calculation_invalid_input(self):
        """Test handling of invalid input"""
        question = "What is the ROI?"  # No numbers provided
        
        # Mock the question analyzer to return ROI type
        self.service.question_analyzer.classify_question = Mock(return_value=(QuestionType.CALCULATION, CalculationType.ROI))
        
        result, explanation = self.service.process_calculation(question)
        
        self.assertIsNone(result)
        self.assertIn("No numerical values", explanation)
        
    def test_process_non_calculation_question(self):
        """Test handling of questions that don't require calculation"""
        question = "What is the company's mission statement?"
        
        # Mock the question analyzer to return non-calculation type
        self.service.question_analyzer.classify_question = Mock(return_value=(QuestionType.EXTRACTION, None))
        
        result, explanation = self.service.process_calculation(question)
        
        self.assertIsNone(result)
        self.assertIn("does not require numerical calculation", explanation)
        
    def test_process_calculation_error_handling(self):
        """Test error handling during calculation"""
        question = "What is the ROI on -100 and 0?"  # Division by zero case
        
        # Mock the question analyzer
        self.service.question_analyzer.classify_question = Mock(return_value=(QuestionType.CALCULATION, CalculationType.ROI))
        
        # Mock the financial formulas to raise an exception
        self.service.financial_formulas.return_on_investment = Mock(side_effect=ZeroDivisionError("Division by zero"))
        
        result, explanation = self.service.process_calculation(question)
        
        self.assertIsNone(result)
        self.assertIn("Error performing calculation", explanation)
        
    def test_validate_result(self):
        """Test result validation for different calculation types"""
        test_cases = [
            {
                "result": Decimal('0.5'),
                "calculation_type": CalculationType.PERCENTAGE_CHANGE,
                "expected": True
            },
            {
                "result": Decimal('15'),
                "calculation_type": CalculationType.PERCENTAGE_CHANGE,
                "expected": False  # Above 1000% increase
            },
            {
                "result": Decimal('-2'),
                "calculation_type": CalculationType.ROI,
                "expected": False  # Below -100%
            },
            {
                "result": Decimal('15.5'),
                "calculation_type": CalculationType.PRICE_TO_EARNINGS,
                "expected": True
            },
            {
                "result": Decimal('-1000'),
                "calculation_type": CalculationType.NET_PRESENT_VALUE,
                "expected": True  # NPV can be any value
            },
            {
                "result": Decimal('42.5'),
                "calculation_type": CalculationType.STATISTICAL,
                "expected": True  # Statistical measures can be any value
            },
            {
                "result": Decimal('1.5'),
                "calculation_type": CalculationType.RATIO,
                "expected": True  # Ratios can be any value
            },
            {
                "result": Decimal('100'),
                "calculation_type": CalculationType.ARITHMETIC,
                "expected": True  # Arithmetic results can be any value
            }
        ]
        
        for case in test_cases:
            with self.subTest(result=case["result"], calculation_type=case["calculation_type"]):
                result = self.service.validate_result(case["result"], case["calculation_type"])
                self.assertEqual(result, case["expected"])
                
    def test_validate_result_none(self):
        """Test validation of None result"""
        result = self.service.validate_result(None, CalculationType.ROI)
        self.assertFalse(result)
        
    def test_validate_result_none_type(self):
        """Test validation with None calculation type"""
        result = self.service.validate_result(Decimal('100'), None)
        self.assertFalse(result)
        
    def test_validate_result_invalid_type(self):
        """Test validation with invalid result type"""
        result = self.service.validate_result("not a decimal", CalculationType.ROI)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main() 