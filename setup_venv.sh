#!/bin/bash

# Exit on error
set -e

# Check if Python 3.8+ is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3.8+ is required but not installed. Please install it first."
    exit 1
fi

# Check Python version (3.8+ required)
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$PYTHON_VERSION" < "3.8" ]]; then
    echo "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
echo "🔄 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install backend dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install development dependencies (optional)
echo "🔧 Installing development dependencies..."
pip install black isort flake8 pytest pytest-cov

# Install spaCy models
echo "🤖 Downloading spaCy models..."
python -m spacy download en_core_web_sm

# Initialize git (if not already a git repo)
if [ ! -d .git ]; then
    echo "🔄 Initializing git repository..."
    git init
    
    echo "📄 Creating .gitignore..."
    cat > .gitignore << 'EOL'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# Environment variables
.env

# Logs
logs/
*.log

# Data files
*.db
*.sqlite3
chroma_db/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter Notebook
.ipynb_checkpoints

# Local development
frontend/node_modules/
frontend/dist/
frontend/.vite/
EOL

    echo "✅ Git repository initialized"
fi

echo "✨ Setup complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To start the backend server: uvicorn backend.app.api:app --reload"
echo "To start the frontend: cd frontend && npm install && npm run dev"
