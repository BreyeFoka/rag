# Hybrid RAG System 🚀

A comprehensive **Retrieval-Augmented Generation (RAG)** system that combines **dense vector search** and **sparse keyword search** for superior document retrieval and question answering. Built with Python, Ollama, and modern AI libraries.

## 🌟 Features

### Core RAG Capabilities
- **Hybrid Retrieval**: Combines dense (vector similarity) and sparse (BM25) search methods
- **Multiple Document Formats**: PDF, DOCX, TXT, MD, HTML support
- **Vector Database**: ChromaDB and FAISS support
- **Ollama Integration**: Local LLM inference with multiple model support
- **SQLite Database**: Persistent storage for documents and metadata

### Interfaces
- **REST API**: Complete web API with Flask
- **Interactive CLI**: Command-line interface for all operations
- **Batch Processing**: Directory-based document ingestion

### Advanced Features
- **Configurable Chunking**: Customizable text splitting strategies
- **Weighted Scoring**: Adjustable dense/sparse retrieval weights
- **Query Analytics**: Performance tracking and result logging
- **Index Management**: Rebuild and optimize search indices

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│ Document         │───▶│ Text Chunking   │
│ (PDF,TXT,DOCX)  │    │ Processor        │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                              ┌─────────────────────────┼─────────────────────────┐
                              ▼                         ▼                         ▼
                    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
                    │ Vector Database │    │ Sparse Index    │    │ SQLite Database │
                    │ (ChromaDB/FAISS)│    │ (BM25)          │    │ (Metadata)      │
                    └─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                         │                         │
                              └─────────────────────────┼─────────────────────────┘
                                                       ▼
                                            ┌─────────────────┐
                                            │ Hybrid RAG      │
                                            │ System          │
                                            └─────────────────┘
                                                       │
                                                       ▼
                                            ┌─────────────────┐
                                            │ Ollama LLM      │
                                            │ (llama2, etc.)  │
                                            └─────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install Ollama
# Visit: https://ollama.ai/download

# Start Ollama server
ollama serve

# Pull a model (e.g., Llama 2)
ollama pull llama2
```

### 2. Installation

```bash
# Clone or download the project
cd hybrid-rag-system

# Install dependencies
pip install -r requirements.txt
```

### 3. First Run

```bash
# Setup system with sample documents and run tests
python main.py setup
```

This will:
- Initialize the RAG system
- Process sample documents
- Run test queries
- Display system statistics

### 4. Usage Options

#### Interactive CLI Mode
```bash
python main.py cli

# Interactive queries
Query: What is artificial intelligence?
Query: dense: explain neural networks
Query: sparse: Python programming features
Query: help
Query: quit
```

#### Web API Server
```bash
python main.py api

# Server runs at http://localhost:5000
# Access API documentation at /api endpoints
```

#### Command-Line Operations
```bash
# Add documents
python main.py cli add ./documents/my_document.pdf
python main.py cli add ./my_documents/ --recursive

# Single query
python main.py cli query "What is machine learning?" --method hybrid

# List documents
python main.py cli list

# System statistics
python main.py cli stats

# Rebuild indices
python main.py cli rebuild
```

## 📚 API Reference

### Query Endpoint
```bash
POST /api/query
Content-Type: application/json

{
    "question": "What is artificial intelligence?",
    "retrieval_method": "hybrid"  # "dense", "sparse", or "hybrid"
}
```

### Document Upload
```bash
POST /api/documents
Content-Type: multipart/form-data

# Upload file in 'file' field
```

### System Statistics
```bash
GET /api/system/stats
```

### Available Endpoints
- `GET /` - Health check
- `GET /api/health` - Detailed health status
- `POST /api/query` - Ask questions
- `POST /api/documents` - Upload documents
- `POST /api/documents/directory` - Process directory
- `GET /api/documents` - List documents
- `GET /api/documents/<id>` - Get document details
- `POST /api/search/dense` - Dense search only
- `POST /api/search/sparse` - Sparse search only
- `GET /api/system/stats` - System statistics
- `POST /api/system/rebuild-indices` - Rebuild indices
- `GET /api/config` - System configuration

## ⚙️ Configuration

Edit `config.py` to customize the system:

```python
# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama2"  # or "llama3", "mistral", "codellama"

# Vector Database
VECTOR_DB_TYPE = "chroma"  # or "faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Configuration
TOP_K_DENSE = 10
TOP_K_SPARSE = 5
TOP_K_FINAL = 5

# Hybrid Search Weights
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
```

## 🔧 Advanced Usage

### Custom Document Processing
```python
from src.hybrid_rag import HybridRAGSystem

# Initialize system
rag = HybridRAGSystem()

# Add single document
doc_id = rag.add_document("path/to/document.pdf")

# Perform different types of retrieval
dense_results = rag.dense_retrieval("query", top_k=10)
sparse_results = rag.sparse_retrieval("query", top_k=10)
hybrid_results = rag.hybrid_retrieval("query", top_k=5)

# Generate response
response = rag.query("What is AI?", retrieval_method="hybrid")
```

### Retrieval Methods Comparison

1. **Dense Retrieval** (Vector Similarity)
   - Best for: Semantic similarity, conceptual queries
   - Uses: Sentence transformers, vector embeddings
   - Example: "What are the benefits of AI?" → finds semantically similar content

2. **Sparse Retrieval** (BM25)
   - Best for: Exact keyword matches, specific terms
   - Uses: BM25 algorithm, inverted index
   - Example: "Python programming" → finds documents with exact terms

3. **Hybrid Retrieval** (Combined)
   - Best for: Most queries, balanced relevance
   - Uses: Weighted combination of dense + sparse
   - Example: Combines semantic understanding with keyword precision

### Database Schema

The system uses SQLite with the following main tables:

- **documents**: Document metadata (filename, size, status, etc.)
- **document_chunks**: Text chunks with content and references
- **search_queries**: Query history and performance metrics
- **query_results**: Individual search results with relevance scores

## 🛠️ Development

### Project Structure
```
hybrid-rag-system/
├── src/                    # Core system code
│   ├── database.py         # SQLite database models
│   ├── document_processor.py  # Document loading and chunking
│   ├── vector_db.py        # Vector database management
│   ├── sparse_retrieval.py # BM25 sparse search
│   ├── hybrid_rag.py       # Main RAG system
│   └── api.py              # Flask REST API
├── documents/              # Sample documents
├── data/                   # Generated database files
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── main.py               # Main entry point
├── cli.py                # Command-line interface
└── README.md             # This file
```

### Adding New Document Types

1. Extend `DocumentProcessor` class
2. Add new loader in `load_document` method
3. Update `supported_extensions` dictionary

### Customizing Retrieval

1. Modify weights in `config.py`
2. Implement custom reranking in `_combine_and_rerank`
3. Add new embedding models

### Performance Optimization

1. **Vector Database**: Use FAISS for large datasets
2. **Chunking**: Adjust `CHUNK_SIZE` based on document types
3. **Caching**: Implement embedding caching for repeated queries
4. **Indexing**: Rebuild indices periodically for new documents

## 🔍 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Start Ollama server
   ollama serve
   
   # Check if model is available
   ollama list
   
   # Pull model if needed
   ollama pull llama2
   ```

2. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt
   
   # Check Python version (requires 3.8+)
   python --version
   ```

3. **Memory Issues**
   - Reduce `CHUNK_SIZE` in config
   - Use FAISS instead of ChromaDB
   - Process documents in smaller batches

4. **Slow Queries**
   - Reduce `TOP_K_DENSE` and `TOP_K_SPARSE`
   - Use smaller embedding models
   - Implement query caching

### Performance Tips

- Use SSD storage for vector databases
- Allocate sufficient RAM for embeddings
- Consider GPU acceleration for large models
- Monitor system resources during indexing

## 📊 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB disk space
- Ollama compatible system

### Recommended Requirements
- Python 3.10+
- 8GB+ RAM
- SSD storage
- GPU for large models (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Format code
black src/
isort src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** for local LLM inference
- **ChromaDB** for vector database
- **LangChain** for document processing
- **Sentence Transformers** for embeddings
- **Flask** for web API

---

## 📞 Support

For questions and support:

1. Check the [troubleshooting section](#-troubleshooting)
2. Search existing issues
3. Create a new issue with details
4. Join community discussions

**Happy RAG-ing! 🎉**