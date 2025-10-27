"""
Document processing module for the RAG system.
Handles loading, parsing, and chunking of various document formats.
"""


from pathlib import Path
from typing import List, Dict, Any, Optional


# Text processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    DirectoryLoader,
    UnstructuredWordDocumentLoader
)

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


class DocumentProcessor:
    """Handles document loading, processing, and chunking"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Supported file types
        self.supported_extensions = {
            '.txt': 'text/plain',
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.md': 'text/markdown',
            '.html': 'text/html',
            '.htm': 'text/html'
        }
    
    def load_document(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load and parse a document based on its file type
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks with metadata
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine file type
            extension = file_path.suffix.lower()
            content_type = self.supported_extensions.get(extension)
            
            if not content_type:
                raise ValueError(f"Unsupported file type: {extension}")
            
            # Load document based on type
            if extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif extension in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            elif extension in ['.txt', '.md', '.html', '.htm']:
                loader = TextLoader(str(file_path), encoding='utf-8')
            else:
                raise ValueError(f"No loader available for {extension}")
            
            # Load and split document
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            
            # Prepare document metadata
            file_stat = file_path.stat()
            doc_metadata = {
                'filename': file_path.name,
                'title': file_path.stem,
                'content_type': content_type,
                'file_path': str(file_path),
                'file_size': file_stat.st_size
            }
            
            # Process chunks
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'chunk_index': i,
                    'content': chunk.page_content,
                    'content_length': len(chunk.page_content),
                    # metadata is stored separately in the document, not in chunks
                    # 'metadata': chunk.metadata
                }
                processed_chunks.append(chunk_data)
            
            return doc_metadata, processed_chunks
            
        except Exception as e:
            print(f"Error loading document {file_path}: {str(e)}")
            return None
    
    def process_document(self, file_path: str) -> Optional[int]:
        """
        Process a document and store it in the database
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            # Load document
            result = self.load_document(file_path)
            if not result:
                return None
            
            doc_metadata, chunks = result
            
            # Add document to database
            document = self.db_manager.add_document(**doc_metadata)
            
            # Add chunks to database
            for chunk_data in chunks:
                self.db_manager.add_chunk(
                    document_id=document.id,
                    **chunk_data
                )
            
            # Update document status
            from datetime import datetime
            self.db_manager.update_document_status(
                document.id, 
                'processed', 
                datetime.utcnow()
            )
            
            print(f"Successfully processed document: {doc_metadata['filename']}")
            print(f"Created {len(chunks)} chunks")
            
            return document.id
            
        except Exception as e:
            print(f"Error processing document {file_path}: {str(e)}")
            return None
    
    def process_directory(self, directory_path: str) -> List[int]:
        """
        Process all supported documents in a directory
        
        Args:
            directory_path: Path to the directory containing documents
            
        Returns:
            List of successfully processed document IDs
        """
        processed_docs = []
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            print(f"Directory not found: {directory_path}")
            return processed_docs
        
        # Find all supported files
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                doc_id = self.process_document(str(file_path))
                if doc_id:
                    processed_docs.append(doc_id)
        
        print(f"Processed {len(processed_docs)} documents from {directory_path}")
        return processed_docs
    
    def get_document_text(self, document_id: int) -> str:
        """
        Get the full text content of a document by combining all chunks
        
        Args:
            document_id: Database ID of the document
            
        Returns:
            Full text content of the document
        """
        chunks = self.db_manager.get_chunks_by_document(document_id)
        return "\n".join([chunk.content for chunk in chunks])
    
    def update_chunk_content(self, chunk_id: int, new_content: str) -> bool:
        """
        Update the content of a specific chunk
        
        Args:
            chunk_id: Database ID of the chunk
            new_content: New content for the chunk
            
        Returns:
            True if successful, False otherwise
        """
        try:
            chunk = self.db_manager.session.query(self.db_manager.DocumentChunk).filter_by(id=chunk_id).first()
            if chunk:
                chunk.content = new_content
                chunk.content_length = len(new_content)
                self.db_manager.session.commit()
                return True
            return False
        except Exception as e:
            print(f"Error updating chunk {chunk_id}: {str(e)}")
            return False