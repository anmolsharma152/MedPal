"""
Medical Knowledge Retriever

This module provides functionality to retrieve relevant medical knowledge
from the vector store using semantic search.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .vector_store import get_vector_store
from ..config import settings

logger = logging.getLogger(__name__)

class MedicalKnowledgeRetriever:
    """Retrieves relevant medical knowledge for a given query."""
    
    def __init__(self, collection_name: str = "medical_knowledge"):
        """Initialize the retriever.
        
        Args:
            collection_name: Name of the ChromaDB collection to use
        """
        self.vector_store = get_vector_store()
        self.collection_name = collection_name
        self.knowledge_base_path = settings.KNOWLEDGE_BASE_PATH
        
        # Initialize with default knowledge if collection is empty
        if self.vector_store.get_collection_stats().get('document_count', 0) == 0:
            self._initialize_default_knowledge()
    
    def _initialize_default_knowledge(self) -> None:
        """Load default medical knowledge into the vector store if empty."""
        logger.info("Initializing default medical knowledge base...")
        
        # Load ICD-10 codes
        icd10_path = Path(self.knowledge_base_path) / "ontologies" / "icd10_codes.json"
        if icd10_path.exists():
            with open(icd10_path, 'r') as f:
                icd10_data = json.load(f)
                
            documents = []
            for code, info in icd10_data.items():
                doc_text = f"ICD-10 Code: {code}\n"
                doc_text += f"Description: {info.get('description', '')}\n"
                doc_text += f"Category: {info.get('category', '')}\n"
                if 'synonyms' in info:
                    doc_text += f"Also known as: {', '.join(info['synonyms'])}\n"
                documents.append({
                    'text': doc_text,
                    'metadata': {
                        'source': 'icd10',
                        'code': code,
                        'type': 'diagnosis_code'
                    }
                })
            
            self.vector_store.add_documents(documents)
            logger.info(f"Loaded {len(documents)} ICD-10 codes into the knowledge base")
    
    def retrieve_relevant_knowledge(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.6,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant medical knowledge for a query.
        
        Args:
            query: The query string
            max_results: Maximum number of results to return
            min_score: Minimum similarity score (0-1)
            filter_conditions: Optional metadata filters
            
        Returns:
            List of relevant knowledge items with scores
        """
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=max_results,
                filter_conditions=filter_conditions,
                min_score=min_score
            )
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append({
                    'rank': i,
                    'content': result['document'],
                    'metadata': result['metadata'],
                    'relevance_score': result['score']
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving medical knowledge: {str(e)}")
            return []
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return self.vector_store.get_collection_stats()
    
    def clear_knowledge_base(self) -> None:
        """Clear all knowledge from the vector store."""
        self.vector_store.delete_collection()
        logger.info("Cleared all knowledge from the vector store")

# Singleton instance
_retriever_instance = None

def get_medical_retriever() -> MedicalKnowledgeRetriever:
    """Get or create a singleton instance of the medical retriever."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = MedicalKnowledgeRetriever()
    return _retriever_instance
