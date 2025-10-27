"""
Vector database management for dense retrieval in the RAG system.
Supports both ChromaDB and FAISS for vector storage and similarity search.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path

# Vector databases
import chromadb
from chromadb.config import Settings
import faiss

# Embeddings
from sentence_transformers import SentenceTransformer

# Local imports
from .database import DatabaseManager
try:
    from .. import config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config


class VectorDBManager:
    """Manager for vector database operations"""
    
    def __init__(self, db_manager: DatabaseManager, db_type: str = None):
        self.db_manager = db_manager
        self.db_type = db_type or config.VECTOR_DB_TYPE
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize vector database
        if self.db_type == "chroma":
            self._init_chroma()
        elif self.db_type == "faiss":
            self._init_faiss()
        else:
            raise ValueError(f"Unsupported vector database type: {self.db_type}")
    
    def _init_chroma(self):
        """Initialize ChromaDB"""
        # Create data directory if it doesn't exist
        persist_dir = Path(config.CHROMA_PERSIST_DIRECTORY)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_dir)
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_chunks",
            metadata={"description": "RAG document chunks for retrieval"}
        )
    
    def _init_faiss(self):
        """Initialize FAISS index"""
        self.faiss_index_path = Path(config.FAISS_INDEX_PATH)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing index
        if self.faiss_index_path.exists():
            self.faiss_index = faiss.read_index(str(self.faiss_index_path))
            # Load metadata
            metadata_path = self.faiss_index_path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.faiss_metadata = json.load(f)
            else:
                self.faiss_metadata = {}
        else:
            # Create new index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            self.faiss_metadata = {}
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a given text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Add document chunks to the vector database
        
        Args:
            chunks: List of chunk dictionaries with 'id', 'content', and 'metadata'
            
        Returns:
            List of vector IDs
        """
        if self.db_type == "chroma":
            return self._add_chunks_chroma(chunks)
        elif self.db_type == "faiss":
            return self._add_chunks_faiss(chunks)
    
    def _add_chunks_chroma(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Add chunks to ChromaDB"""
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for chunk in chunks:
            chunk_id = str(chunk['id'])
            content = chunk['content']
            metadata = chunk.get('metadata', {})
            
            # Generate embedding
            embedding = self.generate_embedding(content)
            
            ids.append(chunk_id)
            documents.append(content)
            metadatas.append(metadata)
            embeddings.append(embedding.tolist())
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        return ids
    
    def _add_chunks_faiss(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Add chunks to FAISS"""
        embeddings = []
        ids = []
        
        for chunk in chunks:
            chunk_id = str(chunk['id'])
            content = chunk['content']
            metadata = chunk.get('metadata', {})
            
            # Generate embedding
            embedding = self.generate_embedding(content)
            embeddings.append(embedding)
            ids.append(chunk_id)
            
            # Store metadata
            self.faiss_metadata[chunk_id] = {
                'content': content,
                'metadata': metadata
            }
        
        # Add embeddings to FAISS
        embeddings_array = np.array(embeddings).astype('float32')
        self.faiss_index.add(embeddings_array)
        
        # Save index and metadata
        faiss.write_index(self.faiss_index, str(self.faiss_index_path))
        metadata_path = self.faiss_index_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.faiss_metadata, f)
        
        return ids
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform dense vector similarity search
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of search results with similarity scores
        """
        if self.db_type == "chroma":
            return self._search_chroma(query, top_k)
        elif self.db_type == "faiss":
            return self._search_faiss(query, top_k)
    
    def _search_chroma(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using ChromaDB"""
        query_embedding = self.generate_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        search_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity_score': 1.0 - results['distances'][0][i]  # Convert distance to similarity
            }
            search_results.append(result)
        
        return search_results
    
    def _search_faiss(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using FAISS"""
        query_embedding = self.generate_embedding(query).astype('float32').reshape(1, -1)
        
        # Perform search
        similarities, indices = self.faiss_index.search(query_embedding, top_k)
        
        search_results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == -1:  # No more results
                break
            
            # Get chunk ID (for FAISS, we need to maintain ID mapping)
            chunk_ids = list(self.faiss_metadata.keys())
            if idx < len(chunk_ids):
                chunk_id = chunk_ids[idx]
                chunk_data = self.faiss_metadata[chunk_id]
                
                result = {
                    'id': chunk_id,
                    'content': chunk_data['content'],
                    'metadata': chunk_data['metadata'],
                    'similarity_score': float(similarity)
                }
                search_results.append(result)
        
        return search_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        if self.db_type == "chroma":
            count = self.collection.count()
            return {
                'total_vectors': count,
                'embedding_dimension': self.embedding_dim,
                'db_type': 'ChromaDB'
            }
        elif self.db_type == "faiss":
            return {
                'total_vectors': self.faiss_index.ntotal,
                'embedding_dimension': self.embedding_dim,
                'db_type': 'FAISS'
            }
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk from the vector database
        
        Args:
            chunk_id: ID of the chunk to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.db_type == "chroma":
                self.collection.delete(ids=[chunk_id])
                return True
            elif self.db_type == "faiss":
                # FAISS doesn't support direct deletion, would need to rebuild index
                # For now, just remove from metadata
                if chunk_id in self.faiss_metadata:
                    del self.faiss_metadata[chunk_id]
                    metadata_path = self.faiss_index_path.with_suffix('.json')
                    with open(metadata_path, 'w') as f:
                        json.dump(self.faiss_metadata, f)
                    return True
            return False
        except Exception as e:
            print(f"Error deleting chunk {chunk_id}: {str(e)}")
            return False