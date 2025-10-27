"""
Hybrid RAG System - AI-Powered Document Retrieval & Question Answering

A comprehensive RAG system that combines dense vector search and sparse keyword 
search for superior document retrieval and question answering capabilities.

Main Components:
- HybridRAGSystem: Main system orchestrator
- VectorDBManager: Dense retrieval using embeddings
- BM25Retriever: Sparse retrieval using keyword matching
- DocumentProcessor: Document loading and chunking
- DatabaseManager: SQLite database operations

Usage:
    from src.hybrid_rag import HybridRAGSystem
    
    # Initialize system
    rag = HybridRAGSystem()
    
    # Add documents
    rag.add_document("path/to/document.pdf")
    
    # Query system
    result = rag.query("What is AI?", "hybrid")
    print(result['answer'])
"""

__version__ = "1.0.0"
__author__ = "Hybrid RAG System"
__email__ = "contact@hybridrag.com"

# Import main classes for easy access
from .hybrid_rag import HybridRAGSystem
from .database import DatabaseManager
from .vector_db import VectorDBManager
from .sparse_retrieval import BM25Retriever
from .document_processor import DocumentProcessor

__all__ = [
    'HybridRAGSystem',
    'DatabaseManager', 
    'VectorDBManager',
    'BM25Retriever',
    'DocumentProcessor'
]