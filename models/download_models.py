#!/usr/bin/env python3
"""
Medical Model Downloader

This script handles downloading and managing medical NLP models.
"""
import os
import shutil
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
import torch

# Model configurations
MODELS = {
    "bio_clinicalbert": {
        "name": "emilyalsentzer/Bio_ClinicalBERT",
        "type": "bert",
        "description": "BERT model pre-trained on MIMIC-III clinical notes"
    },
    "biogpt": {
        "name": "microsoft/BioGPT-Large",
        "type": "gpt",
        "description": "Large biomedical language model for text generation"
    },
    "bioclinical_ner": {
        "name": "d4data/biomedical-ner-all",
        "type": "ner",
        "description": "Biomedical named entity recognition model"
    }
}

def setup_directories():
    """Create necessary directories for models."""
    base_dir = Path(__file__).parent
    os.makedirs(base_dir / "checkpoints", exist_ok=True)
    os.makedirs(base_dir / "tokenizers", exist_ok=True)
    return base_dir

def download_model(model_key, base_dir):
    """Download and save a specific model."""
    if model_key not in MODELS:
        print(f"Error: Unknown model {model_key}")
        return False
    
    model_info = MODELS[model_key]
    print(f"Downloading {model_key} - {model_info['description']}")
    
    try:
        # Download model and tokenizer
        model = AutoModel.from_pretrained(model_info["name"])
        tokenizer = AutoTokenizer.from_pretrained(model_info["name"])
        
        # Save to disk
        model_dir = base_dir / "checkpoints" / model_key
        model_dir.mkdir(exist_ok=True, parents=True)
        
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        print(f"Successfully downloaded and saved {model_key} to {model_dir}")
        return True
        
    except Exception as e:
        print(f"Error downloading {model_key}: {str(e)}")
        return False

def list_available_models():
    """List all available models for download."""
    print("\nAvailable Models:" + "="*50)
    for i, (key, info) in enumerate(MODELS.items(), 1):
        print(f"{i}. {key.upper()}")
        print(f"   {info['description']}")
        print(f"   Type: {info['type']}")
        print()

def main():
    """Main function to handle model downloads."""
    base_dir = setup_directories()
    
    while True:
        print("\n" + "="*50)
        print("Medical Model Downloader")
        print("1. List available models")
        print("2. Download a model")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            list_available_models()
        elif choice == "2":
            list_available_models()
            model_choice = input("\nEnter model name to download: ").strip().lower()
            if model_choice in MODELS:
                download_model(model_choice, base_dir)
            else:
                print("Invalid model choice.")
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
