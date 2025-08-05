""
Utility functions for the Medical Record Summarizer
"""
import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(log_file: str = None, log_level: str = "INFO"):
    """Configure logging for the application"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Clear existing handlers
    logging.getLogger().handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.setLevel(level)
    
    # File handler if log file is provided
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        file_handler.setLevel(level)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(level=level, handlers=handlers)
    logger.info("Logging configured")

def load_json_file(file_path: Union[str, Path]) -> Any:
    """Load JSON data from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        raise

def save_json_file(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """Save data to a JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        raise

def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """Calculate file hash"""
    hash_func = getattr(hashlib, algorithm, hashlib.md5)
    with open(file_path, 'rb') as f:
        return hash_func(f.read()).hexdigest()

def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """Ensure a directory exists, create if it doesn't"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

def clean_text(text: str) -> str:
    """Basic text cleaning"""
    if not text:
        return ""
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()

def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
