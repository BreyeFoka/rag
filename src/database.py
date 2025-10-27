"""
Database models for the RAG system using SQLAlchemy and SQLite.
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json
import config

Base = declarative_base()

class Document(Base):
    """Model for storing document metadata"""
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    title = Column(String(500))
    content_type = Column(String(50))  # pdf, txt, docx, etc.
    file_path = Column(String(500))
    file_size = Column(Integer)
    upload_date = Column(DateTime, default=datetime.utcnow)
    processed_date = Column(DateTime)
    status = Column(String(20), default='pending')  # pending, processed, failed
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    """Model for storing document chunks with metadata"""
    __tablename__ = 'document_chunks'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_length = Column(Integer)
    start_char = Column(Integer)
    end_char = Column(Integer)
    
    # Vector embedding ID (for referencing in vector DB)
    vector_id = Column(String(100))
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

class SearchQuery(Base):
    """Model for storing search queries and results for analytics"""
    __tablename__ = 'search_queries'
    
    id = Column(Integer, primary_key=True)
    query_text = Column(Text, nullable=False)
    search_type = Column(String(20))  # dense, sparse, hybrid
    timestamp = Column(DateTime, default=datetime.utcnow)
    response_time = Column(Float)
    results_count = Column(Integer)
    
class QueryResult(Base):
    """Model for storing individual query results"""
    __tablename__ = 'query_results'
    
    id = Column(Integer, primary_key=True)
    query_id = Column(Integer, ForeignKey('search_queries.id'), nullable=False)
    chunk_id = Column(Integer, ForeignKey('document_chunks.id'), nullable=False)
    relevance_score = Column(Float)
    rank_position = Column(Integer)
    
    # Relationships
    query = relationship("SearchQuery")
    chunk = relationship("DocumentChunk")

class DatabaseManager:
    """Manager class for database operations"""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or config.SQLITE_DB_PATH
        self.engine = create_engine(f'sqlite:///{self.db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def add_document(self, filename, title=None, content_type=None, file_path=None, file_size=None):
        """Add a new document to the database"""
        document = Document(
            filename=filename,
            title=title or filename,
            content_type=content_type,
            file_path=file_path,
            file_size=file_size
        )
        self.session.add(document)
        self.session.commit()
        return document
    
    def add_chunk(self, document_id, chunk_index, content, content_length, 
                  start_char=None, end_char=None, vector_id=None):
        """Add a document chunk to the database"""
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            content_length=content_length,
            start_char=start_char,
            end_char=end_char,
            vector_id=vector_id
        )
        self.session.add(chunk)
        self.session.commit()
        return chunk
    
    def get_document(self, document_id):
        """Retrieve a document by ID"""
        return self.session.query(Document).filter_by(id=document_id).first()
    
    def get_all_documents(self):
        """Retrieve all documents"""
        return self.session.query(Document).all()
    
    def get_chunks_by_document(self, document_id):
        """Retrieve all chunks for a document"""
        return self.session.query(DocumentChunk).filter_by(document_id=document_id).all()
    
    def update_document_status(self, document_id, status, processed_date=None):
        """Update document processing status"""
        document = self.get_document(document_id)
        if document:
            document.status = status
            if processed_date:
                document.processed_date = processed_date
            self.session.commit()
    
    def log_search_query(self, query_text, search_type, response_time=None, results_count=None):
        """Log a search query"""
        query = SearchQuery(
            query_text=query_text,
            search_type=search_type,
            response_time=response_time,
            results_count=results_count
        )
        self.session.add(query)
        self.session.commit()
        return query
    
    def log_query_result(self, query_id, chunk_id, relevance_score, rank_position):
        """Log a query result"""
        result = QueryResult(
            query_id=query_id,
            chunk_id=chunk_id,
            relevance_score=relevance_score,
            rank_position=rank_position
        )
        self.session.add(result)
        self.session.commit()
        return result
    
    def search_chunks_by_text(self, search_text, limit=10):
        """Simple text search in chunks (for sparse retrieval)"""
        return self.session.query(DocumentChunk).filter(
            DocumentChunk.content.contains(search_text)
        ).limit(limit).all()
    
    def close(self):
        """Close database connection"""
        self.session.close()