"""
Retrieval-Augmented Generation (RAG) components for medical knowledge.

This package provides functionality for retrieving relevant medical information
from knowledge bases to augment language model responses.
"""
from pathlib import Path

# Create the retrieval directory if it doesn't exist
Path(__file__).parent.mkdir(parents=True, exist_ok=True)

__all__ = ['vector_store', 'medical_knowledge', 'retriever']
