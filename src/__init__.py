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
__author__ = "Breye Foka"

# Exports load on first use, so importing a light module such as src.fusion
# doesn't pull in Ollama, ChromaDB or the embedding model.
_EXPORTS = {
    'HybridRAGSystem': '.hybrid_rag',
    'DatabaseManager': '.database',
    'VectorDBManager': '.vector_db',
    'BM25Retriever': '.sparse_retrieval',
    'DocumentProcessor': '.document_processor',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        from importlib import import_module
        return getattr(import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
