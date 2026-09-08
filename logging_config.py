# logging_config.py
"""
Advanced logging configuration for LUMS RAG System
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
import json

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    # ANSI color codes
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    blue = "\x1b[34;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    # Format string
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional specific log file name
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Only add handlers if they haven't been added yet
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)
        
        # File handler for all logs (JSON format)
        if log_file is None:
            log_file = f"app_{datetime.now().strftime('%Y%m%d')}.log"
        file_path = LOG_DIR / log_file
        
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
        
        # Error-only file handler
        error_file = LOG_DIR / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.FileHandler(error_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        logger.addHandler(error_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger

class LoggerContext:
    """Context manager for logging execution time."""
    
    def __init__(self, logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"⏱️ Starting: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if exc_type:
            self.logger.error(f"❌ Failed: {self.operation_name} in {elapsed:.2f}s")
        else:
            self.logger.info(f"✅ Completed: {self.operation_name} in {elapsed:.2f}s")

def log_function_call(logger):
    """Decorator to log function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"🔧 Calling: {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"✅ Success: {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"❌ Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator