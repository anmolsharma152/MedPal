#!/usr/bin/env python3
"""
Setup script for AI Medical Record Summarizer
"""
import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install

# Get the long description from the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Get requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Define the package version
VERSION = "0.1.0"

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # Run the standard install
        install.run(self)
        
        # Create necessary directories
        self.create_directories()
        
        # Download required models
        self.download_models()
    
    def create_directories(self):
        """Create required directories."""
        dirs = [
            "data/raw_records",
            "data/processed",
            "models/checkpoints",
            "models/embeddings",
            "knowledge_base/ontologies",
            "knowledge_base/guidelines",
            "outputs/summaries",
            "outputs/reports",
            "logs/api",
            "logs/processing",
        ]
        
        for dir_path in dirs:
            full_path = Path(dir_path)
            full_path.mkdir(parents=True, exist_ok=True)
            (full_path / ".gitkeep").touch()
    
    def download_models(self):
        """Download required NLP models."""
        try:
            import spacy
            import subprocess
            
            print("\n\u2699\ufe0f Downloading required models...")
            
            # Download spaCy models
            print("  \U0001F4E6 Downloading spaCy models...")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            
            # Download scispaCy models
            print("  \U0001F4E6 Downloading scispaCy models...")
            subprocess.run([
                "pip", "install", 
                "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz"
            ], check=True)
            
            print("\n\u2705 All models downloaded successfully!")
            
        except Exception as e:
            print(f"\n\u274C Error downloading models: {str(e)}")
            print("You may need to download the models manually.")

setup(
    name="ai-medical-summarizer",
    version=VERSION,
    author="Your Name",
    author_email="your.email@example.com",
    description="AI Medical Record Summarizer - Process and summarize medical records using NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/AI_Medical_Record_Summarizer",
    packages=find_packages(where="backend"),
    package_dir={"": "backend"},
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="nlp medical healthcare summarization ai",
    entry_points={
        "console_scripts": [
            "medsum=app.cli:main",
        ],
    },
    cmdclass={
        'install': PostInstallCommand,
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/AI_Medical_Record_Summarizer/issues",
        "Source": "https://github.com/your-username/AI_Medical_Record_Summarizer",
    },
)

if __name__ == "__main__":
    # Add any additional setup steps here
    pass
