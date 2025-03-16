#!/usr/bin/env python
# scripts/run_tests.py
"""
Script to run tests for the Financial QA chatbot.
"""

import os
import sys
import pytest
from pathlib import Path

# Add the project root to the Python path if running from scripts directory
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def main():
    """
    Main entry point for the script.
    """
    # Get the directory containing the tests
    tests_dir = os.path.join(os.path.dirname(__file__), "tests")
    
    # Run the tests
    print(f"Running tests in {tests_dir}...")
    result = pytest.main(["-xvs", tests_dir])
    
    # Exit with the pytest result code
    sys.exit(result)

if __name__ == "__main__":
    main() 