"""
Configuration settings for the Medical Record Summarizer
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Model paths
MODEL_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_RECORDS_DIR = DATA_DIR / "raw_records"
PROCESSED_DIR = DATA_DIR / "processed"

# Output paths
OUTPUT_DIR = BASE_DIR / "outputs"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"

# Knowledge base
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"

# Logging
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

# Create directories if they don't exist
for directory in [
    MODEL_DIR, CHECKPOINT_DIR, 
    DATA_DIR, RAW_RECORDS_DIR, PROCESSED_DIR,
    OUTPUT_DIR, SUMMARIES_DIR,
    KNOWLEDGE_BASE_DIR, LOG_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# Application settings
class Settings:
    APP_NAME = "AI Medical Record Summarizer"
    VERSION = "0.1.0"
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # API settings
    API_PREFIX = "/api/v1"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Model settings
    MODEL_NAME = os.getenv("MODEL_NAME", "emilyalsentzer/Bio_ClinicalBERT")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # File processing
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

settings = Settings()
