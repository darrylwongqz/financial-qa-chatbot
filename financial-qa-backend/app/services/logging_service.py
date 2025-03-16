import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import coloredlogs
import logging

# Import configuration
from app.config import (
    DEVELOPMENT_MODE,
    LOG_LEVEL
)

# Import our centralized logger
from app.utils.logging_utils import logger

# Create a logger instance
logger = logging.getLogger()

def setup_logger(
    log_level: Optional[int] = None,
    log_file: Optional[str] = None,
    force_console_output: bool = True
) -> None:
    """
    Set up the logger with the specified configuration.
    
    Args:
        log_level: The log level to use (defaults to LOG_LEVEL from config)
        log_file: The file to log to (defaults to None, which means log to console only)
        force_console_output: Force output to console even if running in a non-interactive environment
    """
    # Use the log level from config if not specified
    if log_level is None:
        log_level = LOG_LEVEL
    
    # In development mode, use INFO level as the minimum
    if DEVELOPMENT_MODE and log_level > logging.INFO:
        log_level = logging.INFO
    elif not DEVELOPMENT_MODE and log_level > logging.WARNING:
        # In production, use at least WARNING level
        log_level = logging.WARNING
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Create formatter with more detailed information
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Use coloredlogs for prettier output (we know it's available)
    coloredlogs.install(
        level=log_level,
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level_styles={
            'debug': {'color': 'blue'},
            'info': {'color': 'green'},
            'warning': {'color': 'yellow', 'bold': True},
            'error': {'color': 'red', 'bold': True},
            'critical': {'color': 'red', 'bold': True, 'background': 'white'}
        },
        field_styles={
            'asctime': {'color': 'cyan'},
            'levelname': {'color': 'magenta', 'bold': True},
            'name': {'color': 'blue'},
            'lineno': {'color': 'white'}
        }
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
    
    # Force all existing loggers to use the same level
    for logger_name in logging.root.manager.loggerDict:
        module_logger = logging.getLogger(logger_name)
        module_logger.setLevel(log_level)
        # Ensure propagation is enabled
        module_logger.propagate = True
    
    # Configure specific third-party loggers
    for logger_name in ["uvicorn", "uvicorn.error", "fastapi"]:
        module_logger = logging.getLogger(logger_name)
        module_logger.handlers = []  # Remove default handlers
        module_logger.propagate = True  # Ensure messages propagate to root logger
        module_logger.setLevel(log_level)
    
    # Log initialization message
    logging.info(f"Logger initialized with level: {logging.getLevelName(log_level)}") 