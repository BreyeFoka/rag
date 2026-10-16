"""
Hybrid RAG System - Combines dense and sparse retrieval with Ollama LLM integration.
This is the main RAG system that orchestrates all components.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
import ollama
from datetime import datetime

# Local imports
from .database import DatabaseManager
from .vector_db import VectorDBManager
from .sparse_retrieval import BM25Retriever
from .fusion import combine_results
from .document_processor import DocumentProcessor
try:
    from .. import config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config


class HybridRAGSystem:
    """Main Hybrid RAG System class"""
    
    def __init__(self):
        """Initialize the Hybrid RAG system"""
        # Initialize database
        self.db_manager = DatabaseManager()
        
        # Initialize components
        self.document_processor = DocumentProcessor(self.db_manager)
        self.vector_db = VectorDBManager(self.db_manager)
        self.sparse_retriever = BM25Retriever(self.db_manager)
        
        # Ollama client
        self.ollama_client = ollama.Client(host=config.OLLAMA_BASE_URL)
        self.model_name = config.OLLAMA_MODEL
        
        # Verify Ollama connection
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            # List available models
            models = self.ollama_client.list()
            available_models = []
            
            # Handle different response formats
            if isinstance(models, dict) and 'models' in models:
                for model in models['models']:
                    if isinstance(model, dict):
                        model_name = model.get('name', model.get('model', ''))
                        if model_name:
                            available_models.append(model_name.split(':')[0])
            
            if available_models and self.model_name not in available_models:
                print(f"Warning: Model '{self.model_name}' not found.")
                print(f"Available models: {available_models}")
                print(f"You may need to pull the model: ollama pull {self.model_name}")
            elif available_models:
                print(f"✓ Ollama connection successful. Using model: {self.model_name}")
            else:
                print(f"✓ Connected to Ollama. Will use model: {self.model_name}")
                
        except Exception as e:
            print(f"Error connecting to Ollama: {str(e)}")
            print("Make sure Ollama is running: ollama serve")
    
    def add_document(self, file_path: str) -> Optional[int]:
        """
        Add a document to the RAG system
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document ID if successful, None otherwise
        """
        # Process document and add to database
        doc_id = self.document_processor.process_document(file_path)
        
        if doc_id:
            # Get processed chunks
            chunks = self.db_manager.get_chunks_by_document(doc_id)
            
            # Prepare chunks for vector database
            chunk_data = []
            for chunk in chunks:
                chunk_dict = {
                    'id': chunk.id,
                    'content': chunk.content,
                    'metadata': {
                        'document_id': chunk.document_id,
                        'chunk_index': chunk.chunk_index,
                        'content_length': chunk.content_length
                    }
                }
                chunk_data.append(chunk_dict)
            
            # Add to vector database
            vector_ids = self.vector_db.add_chunks(chunk_data)
            
            # Update chunks with vector IDs
            for chunk, vector_id in zip(chunks, vector_ids):
                chunk.vector_id = vector_id
                self.db_manager.session.commit()
            
            # Add to sparse retrieval index
            for chunk_dict in chunk_data:
                self.sparse_retriever.add_document_chunk({
                    'id': chunk_dict['id'],
                    'content': chunk_dict['content'],
                    'document_id': chunk_dict['metadata']['document_id'],
                    'chunk_index': chunk_dict['metadata']['chunk_index']
                })
            
            print(f"✓ Document added successfully. Document ID: {doc_id}")
            return doc_id
        
        return None
    
    def add_documents_from_directory(self, directory_path: str) -> List[int]:
        """
        Add all supported documents from a directory
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of successfully processed document IDs
        """
        doc_ids = self.document_processor.process_directory(directory_path)
        
        # Process each document for vector and sparse retrieval
        for doc_id in doc_ids:
            self._index_document_chunks(doc_id)
        
        return doc_ids
    
    def _index_document_chunks(self, doc_id: int):
        """Index document chunks in vector and sparse retrievers"""
        chunks = self.db_manager.get_chunks_by_document(doc_id)
        
        # Prepare chunks for vector database
        chunk_data = []
        for chunk in chunks:
            chunk_dict = {
                'id': chunk.id,
                'content': chunk.content,
                'metadata': {
                    'document_id': chunk.document_id,
                    'chunk_index': chunk.chunk_index,
                    'content_length': chunk.content_length
                }
            }
            chunk_data.append(chunk_dict)
        
        # Add to vector database
        if chunk_data:
            vector_ids = self.vector_db.add_chunks(chunk_data)
            
            # Update chunks with vector IDs
            for chunk, vector_id in zip(chunks, vector_ids):
                chunk.vector_id = vector_id
            self.db_manager.session.commit()
    
    def dense_retrieval(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Perform dense retrieval using vector similarity
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of dense retrieval results
        """
        top_k = top_k or config.TOP_K_DENSE
        return self.vector_db.search(query, top_k)
    
    def sparse_retrieval(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Perform sparse retrieval using BM25
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of sparse retrieval results
        """
        top_k = top_k or config.TOP_K_SPARSE
        return self.sparse_retriever.search(query, top_k)
    
    def hybrid_retrieval(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval combining dense and sparse methods
        
        Args:
            query: Search query
            top_k: Number of final results to return
            
        Returns:
            List of reranked hybrid retrieval results
        """
        top_k = top_k or config.TOP_K_FINAL
        
        # Get dense results
        dense_results = self.dense_retrieval(query, config.TOP_K_DENSE)
        
        # Get sparse results
        sparse_results = self.sparse_retrieval(query, config.TOP_K_SPARSE)
        
        # Combine and rerank results
        combined_results = self._combine_and_rerank(
            dense_results, sparse_results, query, top_k
        )
        
        return combined_results
    
    def _combine_and_rerank(self, dense_results: List[Dict], sparse_results: List[Dict],
                           query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Combine dense and sparse results with weighted scoring of normalised scores.
        See src/fusion.py for why each list is normalised before weighting.
        """
        return combine_results(
            dense_results, sparse_results,
            config.DENSE_WEIGHT, config.SPARSE_WEIGHT, top_k,
        )

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate response using Ollama LLM with retrieved context
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Generated response
        """
        # Prepare context
        context_text = "\n\n".join([
            f"Context {i+1}: {chunk['content']}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please say "I don't have enough information to answer this question based on the provided context."

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            # Generate response using Ollama
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'max_tokens': config.MAX_TOKENS
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def query(self, question: str, retrieval_method: str = "hybrid") -> Dict[str, Any]:
        """
        Main query method for the RAG system
        
        Args:
            question: User question
            retrieval_method: "dense", "sparse", or "hybrid"
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        start_time = time.time()
        
        # Perform retrieval based on method
        if retrieval_method == "dense":
            retrieved_chunks = self.dense_retrieval(question)
        elif retrieval_method == "sparse":
            retrieved_chunks = self.sparse_retrieval(question)
        elif retrieval_method == "hybrid":
            retrieved_chunks = self.hybrid_retrieval(question)
        else:
            raise ValueError(f"Unknown retrieval method: {retrieval_method}")
        
        retrieval_time = time.time() - start_time
        
        # Generate response
        gen_start_time = time.time()
        answer = self.generate_response(question, retrieved_chunks)
        generation_time = time.time() - gen_start_time
        
        total_time = time.time() - start_time
        
        # Log query to database
        query_log = self.db_manager.log_search_query(
            query_text=question,
            search_type=retrieval_method,
            response_time=total_time,
            results_count=len(retrieved_chunks)
        )
        
        # Log results
        for i, chunk in enumerate(retrieved_chunks):
            self.db_manager.log_query_result(
                query_id=query_log.id,
                chunk_id=int(chunk['chunk_id']),
                relevance_score=chunk.get('combined_score', chunk.get('similarity_score', chunk.get('bm25_score', 0.0))),
                rank_position=i + 1
            )
        
        # Prepare response
        response = {
            'question': question,
            'answer': answer,
            'sources': retrieved_chunks,
            'retrieval_method': retrieval_method,
            'metadata': {
                'total_time': total_time,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
                'num_sources': len(retrieved_chunks),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return response
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        vector_stats = self.vector_db.get_collection_stats()
        bm25_stats = self.sparse_retriever.get_index_stats()
        
        all_docs = self.db_manager.get_all_documents()
        
        return {
            'total_documents': len(all_docs),
            'vector_database': vector_stats,
            'sparse_index': bm25_stats,
            'database_path': self.db_manager.db_path,
            'model_name': self.model_name
        }
    
    def rebuild_indices(self):
        """Rebuild both vector and sparse indices"""
        print("Rebuilding indices...")
        
        # Get all documents and chunks
        all_docs = self.db_manager.get_all_documents()
        
        for doc in all_docs:
            self._index_document_chunks(doc.id)
        
        # Rebuild sparse index
        self.sparse_retriever.rebuild_index()
        
        print("✓ Indices rebuilt successfully")