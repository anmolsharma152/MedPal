"""
Retriever Module for Medical Record Summarizer

This module provides a high-level interface for retrieving relevant medical knowledge
using the vector store and knowledge retriever.
"""
import logging
from typing import List, Dict, Any, Optional

from .medical_knowledge import get_medical_retriever
from ..config import settings

logger = logging.getLogger(__name__)

class MedicalRetriever:
    """High-level interface for medical knowledge retrieval."""
    
    def __init__(self):
        """Initialize the retriever with default settings."""
        self.retriever = get_medical_retriever()
        self.default_max_results = settings.DEFAULT_RETRIEVAL_LIMIT
        self.default_min_score = settings.MIN_SIMILARITY_SCORE
    
    def get_relevant_knowledge(
        self,
        query: str,
        max_results: Optional[int] = None,
        min_score: Optional[float] = None,
        source: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get relevant medical knowledge for a query.
        
        Args:
            query: The query string
            max_results: Maximum number of results to return
            min_score: Minimum similarity score (0-1)
            source: Filter by source (e.g., 'icd10', 'snomed')
            content_type: Filter by content type (e.g., 'diagnosis_code', 'treatment_guideline')
            
        Returns:
            List of relevant knowledge items with metadata
        """
        # Use defaults if not specified
        max_results = max_results or self.default_max_results
        min_score = min_score or self.default_min_score
        
        # Build filter conditions
        filter_conditions = {}
        if source:
            filter_conditions['source'] = source
        if content_type:
            filter_conditions['type'] = content_type
        
        try:
            results = self.retriever.retrieve_relevant_knowledge(
                query=query,
                max_results=max_results,
                min_score=min_score,
                filter_conditions=filter_conditions or None
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in get_relevant_knowledge: {str(e)}")
            return []
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return self.retriever.get_knowledge_stats()
    
    def clear_knowledge_base(self) -> None:
        """Clear all knowledge from the vector store."""
        self.retriever.clear_knowledge_base()
        logger.info("Knowledge base cleared")

# Singleton instance
_retriever = None

def get_retriever() -> MedicalRetriever:
    """Get or create a singleton instance of the retriever."""
    global _retriever
    if _retriever is None:
        _retriever = MedicalRetriever()
    return _retriever
