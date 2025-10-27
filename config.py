# Hybrid RAG System Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama2"  # You can change to llama3, mistral, codellama, etc.

# Vector Database Configuration
VECTOR_DB_TYPE = "chroma"  # Options: chroma, faiss
CHROMA_PERSIST_DIRECTORY = "./data/chroma_db"
FAISS_INDEX_PATH = "./data/faiss_index"

# SQLite Database
SQLITE_DB_PATH = "./data/knowledge_base.db"

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 4000

# Retrieval Configuration
TOP_K_DENSE = 10  # Dense retrieval (vector similarity)
TOP_K_SPARSE = 5  # Sparse retrieval (BM25/keyword)
TOP_K_FINAL = 5   # Final results after reranking

# Hybrid Search Weights
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3

# API Configuration
API_HOST = "localhost"
API_PORT = 5000
DEBUG_MODE = True