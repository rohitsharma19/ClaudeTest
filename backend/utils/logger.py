"""
Logging utility for RAG Application.

Provides centralized logging configuration and utilities.
"""

import logging
import os
from typing import Optional
from pathlib import Path


def setup_logger(
    name: str,
    level: str = "INFO",
    log_to_file: bool = True,
    log_file_path: str = "./logs/app.log"
) -> logging.Logger:
    """
    Set up and configure a logger.

    Args:
        name: Logger name (typically __name__ of the module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_file_path: Path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str, config: Optional[dict] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name
        config: Optional logging configuration dictionary

    Returns:
        Logger instance
    """
    if config is None:
        config = {
            'level': 'INFO',
            'log_to_file': True,
            'log_file_path': './logs/app.log'
        }

    return setup_logger(
        name=name,
        level=config.get('level', 'INFO'),
        log_to_file=config.get('log_to_file', True),
        log_file_path=config.get('log_file_path', './logs/app.log')
    )
