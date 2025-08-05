# Development Guide

This guide provides detailed information for developers working on the AI Medical Record Summarizer project.

## Table of Contents

- [Project Structure](#project-structure)
- [Development Setup](#development-setup)
- [Backend Development](#backend-development)
- [Frontend Development](#frontend-development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Project Structure

```
AI_Medical_Record_Summarizer/
├── backend/               # Backend application (FastAPI)
│   └── app/              # Main application code
│       ├── __init__.py
│       ├── api.py        # FastAPI application
│       ├── config.py     # Configuration settings
│       ├── processor.py  # Core processing logic
│       └── utils.py      # Utility functions
├── frontend/             # Frontend application (React)
│   ├── public/           # Static files
│   └── src/              # Source code
│       ├── components/   # React components
│       ├── App.jsx       # Main application component
│       └── main.jsx      # Application entry point
├── data/                 # Data storage
│   ├── raw_records/     # Original uploaded files
│   └── processed/       # Processed data
├── models/              # ML models and embeddings
│   └── checkpoints/     # Model checkpoints
├── knowledge_base/      # Medical knowledge base
├── outputs/             # Generated outputs
│   └── summaries/       # Generated summaries
├── logs/                # Application logs
└── scripts/             # Utility scripts
    └── setup_directories.py  # Script to set up directories
```

## Development Setup

### Prerequisites

- Python 3.8+
- Node.js 16+
- pip (Python package manager)
- npm or yarn (Node.js package manager)

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AI_Medical_Record_Summarizer
   ```

2. **Set up Python virtual environment**:
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up frontend dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

6. **Create required directories**:
   ```bash
   python scripts/setup_directories.py
   ```

## Backend Development

### Running the Backend

```bash
# From project root
uvicorn backend.app.api:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

Run these tools before committing:

```bash
black .
isort .
flake8
```

## Frontend Development

### Running the Frontend

```bash
# From the frontend directory
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Code Style

We use:
- **Prettier** for code formatting
- **ESLint** for JavaScript/TypeScript linting

## Testing

### Backend Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=backend tests/
```

### Frontend Tests

```bash
# From the frontend directory
cd frontend
npm test
```

## Deployment

### Production Build

1. **Build the frontend**:
   ```bash
   cd frontend
   npm run build
   ```

2. **Set up production environment variables**:
   ```bash
   cp .env.production .env
   # Edit .env with production configuration
   ```

3. **Start the production server**:
   ```bash
   uvicorn backend.app.api:app --host 0.0.0.0 --port 8000
   ```

### Docker Deployment

1. **Build the Docker image**:
   ```bash
   docker build -t medical-record-summarizer .
   ```

2. **Run the container**:
   ```bash
   docker run -d -p 8000:8000 --env-file .env medical-record-summarizer
   ```

## Troubleshooting

### Common Issues

1. **Python package installation fails**:
   - Ensure you have the latest version of pip: `pip install --upgrade pip`
   - Make sure you're using Python 3.8 or higher

2. **Frontend not connecting to backend**:
   - Check that the backend is running
   - Verify the API URL in the frontend configuration
   - Check CORS settings in the backend

3. **Missing dependencies**:
   - Run `pip install -r requirements.txt` for Python dependencies
   - Run `npm install` in the frontend directory for frontend dependencies

### Getting Help

If you encounter any issues not covered here, please [open an issue](https://github.com/your-username/AI_Medical_Record_Summarizer/issues) with details about the problem.
