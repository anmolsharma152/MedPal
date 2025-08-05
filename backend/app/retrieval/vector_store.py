"""
Vector Store for Medical Knowledge Retrieval

This module implements a vector store for efficient similarity search
over medical knowledge using ChromaDB.
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)

class MedicalVectorStore:
    """Vector store for medical knowledge retrieval."""
    
    def __init__(self, collection_name: str = "medical_knowledge"):
        """Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection to use
        """
        self.collection_name = collection_name
        self.persist_directory = settings.VECTOR_STORE_PATH
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        ))
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL_NAME,
            device=settings.EMBEDDING_DEVICE
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except ValueError:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata' keys
            batch_size: Number of documents to add in each batch
        """
        if not documents:
            return
        
        # Prepare data for batch addition
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            if not doc.get('text'):
                continue
                
            doc_id = f"doc_{len(self.collection.get()['ids']) + len(ids)}"
            metadata = doc.get('metadata', {})
            
            ids.append(doc_id)
            texts.append(doc['text'])
            metadatas.append(metadata)
            
            # Add in batches
            if len(ids) >= batch_size:
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                ids, texts, metadatas = [], [], []
        
        # Add any remaining documents
        if ids:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        logger.info(f"Added {len(documents)} documents to the vector store")
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: The query string
            k: Number of results to return
            filter_conditions: Dictionary of metadata filters
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of dictionaries with 'document', 'metadata', and 'score' keys
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=filter_conditions
            )
            
            # Process results
            documents = []
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to similarity score (assuming cosine distance)
                # Cosine distance = 1 - cosine_similarity
                score = 1.0 - distance if distance is not None else 0.0
                
                if score >= min_score:
                    documents.append({
                        'document': doc,
                        'metadata': metadata,
                        'score': score
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def delete_collection(self) -> None:
        """Delete the current collection."""
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'embedding_model': settings.EMBEDDING_MODEL_NAME
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}

# Singleton instance
_vector_store_instance = None

def get_vector_store() -> MedicalVectorStore:
    """Get or create a singleton instance of the vector store."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = MedicalVectorStore()
    return _vector_store_instance
