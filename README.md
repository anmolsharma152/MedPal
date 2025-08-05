# AI Medical Record Summarizer

An AI-powered application for processing and summarizing medical records using state-of-the-art NLP techniques, including OCR for scanned documents and RAG (Retrieval-Augmented Generation) for enhanced medical knowledge retrieval.

## Features

- **Document Processing**: Handles various medical record formats (PDF, TXT, DOCX, scanned images)
- **OCR Support**: Extracts text from scanned documents and images
- **Entity Extraction**: Identifies medical entities (conditions, medications, procedures) using Bio_ClinicalBERT
- **Knowledge Retrieval**: RAG system with medical ontologies (ICD-10, SNOMED CT)
- **Clinical Insights**: Generates AI-powered insights from medical records
- **Summarization**: Creates concise, structured summaries of medical records
- **Web Interface**: User-friendly interface for uploading and viewing results

## Project Structure

```
AI_Medical_Record_Summarizer/
├── backend/                     # Backend application
│   └── app/                    # Main application code
│       ├── __init__.py
│       ├── api.py              # FastAPI application
│       ├── config.py           # Configuration settings
│       ├── pipeline.py         # Core processing pipeline
│       ├── ocr_processor.py    # OCR for scanned documents
│       ├── utils.py            # Utility functions
│       └── retrieval/          # RAG components
│           ├── __init__.py
│           ├── vector_store.py # Vector database operations
│           ├── medical_knowledge.py # Knowledge base integration
│           └── retriever.py    # High-level retrieval interface
├── frontend/                   # Frontend React application
│   ├── public/
│   └── src/
│       ├── components/
│       ├── App.jsx
│       └── main.jsx
├── data/                       # Data storagee
│   ├── raw_records/           # Original uploaded files
│   └── processed/             # Processed data
├── models/                    # ML models and embeddings
│   ├── checkpoints/           # Model checkpoints
│   └── download_models.py     # Script to download models
├── knowledge_base/            # Medical knowledge base
│   ├── ontologies/            # ICD-10, SNOMED CT
│   │   ├── icd10_codes.json
│   │   └── snomed_ct_codes.json
│   └── guidelines/            # Clinical guidelines
│       └── clinical_guidelines.md
├── outputs/                   # Generated outputs
│   └── summaries/
├── logs/                      # Application logs
└── scripts/                   # Utility scripts
    └── setup_directories.py
```

## Prerequisites

- Python 3.8+
- Node.js 16+
- pip (Python package manager)
- npm or yarn (Node.js package manager)

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd AI_Medical_Record_Summarizer
```

### 2. Set up the backend

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install OCR dependencies (optional, for scanned documents)
# Requires Tesseract OCR to be installed on your system
# Ubuntu: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

# Set up environment variables
cp .env.example .env
# Edit .env file with your configuration
```

### 3. Download ML Models

```bash
# Download required ML models
python models/download_models.py

# Follow the prompts to download the models
# This includes Bio_ClinicalBERT and other medical NLP models
```

### 4. Set up the frontend

```bash
cd frontend
npm install
cd ..
```

### 5. Create required directories and initialize knowledge base

```bash
python scripts/setup_directories.py
```

## Running the Application

### Start the backend server

```bash
uvicorn backend.app.api:app --reload
```

The API will be available at `http://localhost:8000`

### Start the frontend development server

In a new terminal:

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Key Components

### 1. Document Processing Pipeline
- **OCR Support**: Extracts text from scanned documents and images
- **Text Extraction**: Handles PDF, DOCX, and plain text formats
- **Preprocessing**: Cleans and normalizes medical text

### 2. Medical Knowledge Base
- **ICD-10 Codes**: Standardized diagnostic codes
- **SNOMED CT**: Comprehensive clinical terminology
- **Clinical Guidelines**: Evidence-based medical guidelines

### 3. RAG System
- **Vector Database**: Stores and retrieves medical knowledge
- **Semantic Search**: Finds relevant medical information
- **Knowledge Integration**: Enhances AI responses with retrieved knowledge

## API Documentation

Once the backend is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints
- `POST /api/process`: Process a medical record
- `GET /api/knowledge`: Query the medical knowledge base
- `POST /api/ocr`: Process scanned documents with OCR

## Development

### Adding New Medical Knowledge
1. Add new ontologies to `knowledge_base/ontologies/`
2. Update clinical guidelines in `knowledge_base/guidelines/`
3. Restart the application to rebuild the vector store

### Extending the Pipeline
1. Add new processing steps to `backend/app/pipeline.py`
2. Implement new entity extractors in the `retrieval` module
3. Update the API endpoints in `backend/app/api.py`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with FastAPI, React, and Transformers
- Uses Bio_ClinicalBERT and BioGPT for medical text understanding
- Integrates with ChromaDB for vector search
- Inspired by the need for better medical record processing tools
