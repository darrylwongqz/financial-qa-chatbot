"""
Centralized logging utilities for the application.
This module provides functions and classes to handle logging in a consistent way.
"""
import logging
import sys
import functools
import os
from typing import Callable, Any, Optional
import coloredlogs


# Create a custom logger class that respects the global logging level
class DevModeLogger:
    """
    A logger wrapper that respects the global logging level.
    Uses the underlying Python logging system but provides a consistent interface.
    """
    def __init__(self, name: Optional[str] = None):
        self.logger = logging.getLogger(name)
    
    def debug(self, msg: str, *args, **kwargs):
        # Use the standard logging system which will respect the global level
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        # Use the standard logging system which will respect the global level
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        # Use the standard logging system which will respect the global level
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        # Use the standard logging system which will respect the global level
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        # Use the standard logging system which will respect the global level
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        # Use the standard logging system which will respect the global level
        self.logger.exception(msg, *args, **kwargs)

# Create a global instance for easy import
logger = DevModeLogger("app")

# Decorator for logging function calls
def log_function_call(func: Callable) -> Callable:
    """
    Decorator that logs function calls.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Completed {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

# Decorator for logging API endpoints
def log_endpoint(func: Callable) -> Callable:
    """
    Decorator that logs API endpoint calls.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger.info(f"API endpoint called: {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            logger.info(f"API endpoint completed: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in API endpoint {func.__name__}: {str(e)}")
            raise
    return wrapper 