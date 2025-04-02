"""Logging configuration for the Activity Recommendation System."""

import logging
import os
from datetime import datetime
from config.settings import config

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file (optional)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove any existing handlers
    logger.handlers = []

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file is provided
    if log_file:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

# Create default loggers
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = config.logs_dir

# Application logger
app_logger = setup_logger(
    'app',
    os.path.join(log_dir, f'app_{timestamp}.log'),
    level=logging.INFO
)

# Model logger
model_logger = setup_logger(
    'model',
    os.path.join(log_dir, f'model_{timestamp}.log'),
    level=logging.INFO
)

# Recognition logger
recognition_logger = setup_logger(
    'recognition',
    os.path.join(log_dir, f'recognition_{timestamp}.log'),
    level=logging.INFO
) 