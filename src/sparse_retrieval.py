"""
Sparse retrieval module using BM25 algorithm for keyword-based search.
This provides the sparse component for the Hybrid RAG system.
"""

import re
import math
from typing import List, Dict, Any, Set
from collections import Counter, defaultdict
import pickle
import os
from pathlib import Path

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


class BM25Retriever:
    """BM25 sparse retrieval implementation"""
    
    def __init__(self, db_manager: DatabaseManager, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever
        
        Args:
            db_manager: Database manager instance
            k1: BM25 parameter controlling term frequency saturation
            b: BM25 parameter controlling field length normalization
        """
        self.db_manager = db_manager
        self.k1 = k1
        self.b = b
        
        # BM25 data structures
        self.documents = []  # List of document chunks
        self.doc_freqs = defaultdict(int)  # Document frequency for each term
        self.idf_cache = {}  # IDF cache
        self.doc_lengths = []  # Length of each document
        self.avgdl = 0  # Average document length
        
        # Index file path
        self.index_path = Path("./data/bm25_index.pkl")
        
        # Load existing index or build new one
        self.load_index()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and extract alphanumeric tokens
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def build_index(self):
        """Build BM25 index from database chunks"""
        print("Building BM25 index...")
        
        # Get all documents from database
        all_documents = self.db_manager.get_all_documents()
        all_chunks = []
        
        for doc in all_documents:
            chunks = self.db_manager.get_chunks_by_document(doc.id)
            all_chunks.extend(chunks)
        
        # Reset index data
        self.documents = []
        self.doc_freqs = defaultdict(int)
        self.idf_cache = {}
        self.doc_lengths = []
        
        # Process each chunk
        for chunk in all_chunks:
            # Tokenize chunk content
            tokens = self.tokenize(chunk.content)
            
            # Store document data
            doc_data = {
                'id': chunk.id,
                'content': chunk.content,
                'tokens': tokens,
                'token_count': Counter(tokens),
                'document_id': chunk.document_id,
                'chunk_index': chunk.chunk_index
            }
            self.documents.append(doc_data)
            
            # Update document frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
            
            # Store document length
            self.doc_lengths.append(len(tokens))
        
        # Calculate average document length
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Precompute IDF values
        self._compute_idf()
        
        # Save index
        self.save_index()
        
        print(f"BM25 index built with {len(self.documents)} chunks")
    
    def _compute_idf(self):
        """Compute IDF values for all terms"""
        N = len(self.documents)
        
        for term in self.doc_freqs:
            df = self.doc_freqs[term]
            idf = math.log((N - df + 0.5) / (df + 0.5))
            self.idf_cache[term] = idf
    
    def get_idf(self, term: str) -> float:
        """
        Get IDF value for a term
        
        Args:
            term: The term
            
        Returns:
            IDF value
        """
        return self.idf_cache.get(term, 0.0)
    
    def score_document(self, query_terms: List[str], doc_idx: int) -> float:
        """
        Calculate BM25 score for a document
        
        Args:
            query_terms: List of query terms
            doc_idx: Document index
            
        Returns:
            BM25 score
        """
        if doc_idx >= len(self.documents):
            return 0.0
        
        doc = self.documents[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        score = 0.0
        
        for term in query_terms:
            if term in doc['token_count']:
                tf = doc['token_count'][term]
                idf = self.get_idf(term)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
                score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform BM25 search
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of search results with BM25 scores
        """
        # Tokenize query
        query_terms = self.tokenize(query)
        
        if not query_terms or not self.documents:
            return []
        
        # Score all documents
        doc_scores = []
        for i in range(len(self.documents)):
            score = self.score_document(query_terms, i)
            if score > 0:
                doc_scores.append((i, score))
        
        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Prepare results
        results = []
        for i, (doc_idx, score) in enumerate(doc_scores[:top_k]):
            doc = self.documents[doc_idx]
            result = {
                'id': str(doc['id']),
                'content': doc['content'],
                'bm25_score': score,
                'document_id': doc['document_id'],
                'chunk_index': doc['chunk_index'],
                'rank': i + 1
            }
            results.append(result)
        
        return results
    
    def save_index(self):
        """Save BM25 index to disk"""
        index_data = {
            'documents': self.documents,
            'doc_freqs': dict(self.doc_freqs),
            'idf_cache': self.idf_cache,
            'doc_lengths': self.doc_lengths,
            'avgdl': self.avgdl,
            'k1': self.k1,
            'b': self.b
        }
        
        # Create directory if it doesn't exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to pickle file
        with open(self.index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"BM25 index saved to {self.index_path}")
    
    def load_index(self):
        """Load BM25 index from disk"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'rb') as f:
                    index_data = pickle.load(f)
                
                self.documents = index_data['documents']
                self.doc_freqs = defaultdict(int, index_data['doc_freqs'])
                self.idf_cache = index_data['idf_cache']
                self.doc_lengths = index_data['doc_lengths']
                self.avgdl = index_data['avgdl']
                self.k1 = index_data['k1']
                self.b = index_data['b']
                
                print(f"BM25 index loaded from {self.index_path}")
                print(f"Index contains {len(self.documents)} chunks")
                
            except Exception as e:
                print(f"Error loading BM25 index: {str(e)}")
                print("Building new index...")
                self.build_index()
        else:
            print("No existing BM25 index found. Building new index...")
            self.build_index()
    
    def rebuild_index(self):
        """Force rebuild of the BM25 index"""
        self.build_index()
    
    def add_document_chunk(self, chunk_data: Dict[str, Any]):
        """
        Add a new document chunk to the index
        
        Args:
            chunk_data: Dictionary containing chunk information
        """
        # Tokenize chunk content
        tokens = self.tokenize(chunk_data['content'])
        
        # Create document data
        doc_data = {
            'id': chunk_data['id'],
            'content': chunk_data['content'],
            'tokens': tokens,
            'token_count': Counter(tokens),
            'document_id': chunk_data.get('document_id'),
            'chunk_index': chunk_data.get('chunk_index', len(self.documents))
        }
        
        # Add to documents
        self.documents.append(doc_data)
        self.doc_lengths.append(len(tokens))
        
        # Update document frequencies
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.doc_freqs[token] += 1
        
        # Recalculate average document length
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        
        # Recompute IDF values (this could be optimized)
        self._compute_idf()
        
        # Save updated index
        self.save_index()
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the BM25 index"""
        return {
            'total_documents': len(self.documents),
            'total_terms': len(self.doc_freqs),
            'average_document_length': self.avgdl,
            'k1_parameter': self.k1,
            'b_parameter': self.b
        }