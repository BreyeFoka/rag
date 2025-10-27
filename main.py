"""
Main entry point for the Hybrid RAG System.
This script provides multiple ways to run the system:
1. Interactive CLI mode
2. Web API server
3. One-time setup and testing
"""

import sys
import os
from pathlib import Path
import argparse

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def run_api_server():
    """Run the Flask API server"""
    print("Starting Hybrid RAG API Server...")
    from src.api import app, initialize_rag_system
    
    if initialize_rag_system():
        import config
        print(f"Server starting at http://{config.API_HOST}:{config.API_PORT}")
        print("Available endpoints:")
        print(f"  - Health: http://{config.API_HOST}:{config.API_PORT}/api/health")
        print(f"  - Query: POST http://{config.API_HOST}:{config.API_PORT}/api/query")
        print(f"  - Upload: POST http://{config.API_HOST}:{config.API_PORT}/api/documents")
        print(f"  - Stats: http://{config.API_HOST}:{config.API_PORT}/api/system/stats")
        
        app.run(
            host=config.API_HOST,
            port=config.API_PORT,
            debug=config.DEBUG_MODE
        )
    else:
        print("Failed to initialize RAG system")
        sys.exit(1)

def run_cli():
    """Run the command-line interface"""
    from cli import main as cli_main
    cli_main()

def setup_and_test():
    """Setup the system with sample documents and run tests"""
    print("Setting up Hybrid RAG System with sample documents...")
    
    try:
        from src.hybrid_rag import HybridRAGSystem
        
        # Initialize system
        print("1. Initializing RAG system...")
        rag = HybridRAGSystem()
        
        # Add sample documents
        print("2. Adding sample documents...")
        documents_dir = Path("./documents")
        if documents_dir.exists():
            doc_ids = rag.add_documents_from_directory(str(documents_dir))
            print(f"   Added {len(doc_ids)} documents")
        else:
            print("   No documents directory found")
        
        # Show system stats
        print("3. System Statistics:")
        stats = rag.get_system_stats()
        print(f"   Total Documents: {stats['total_documents']}")
        print(f"   Vector Database: {stats['vector_database']['db_type']}")
        print(f"   Total Vectors: {stats['vector_database']['total_vectors']}")
        print(f"   Model: {stats['model_name']}")
        
        # Run test queries
        print("4. Running test queries...")
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning types",
            "What are Python's key features?",
            "How does deep learning work?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Test {i}: {query}")
            result = rag.query(query, "hybrid")
            print(f"   Answer: {result['answer'][:100]}...")
            print(f"   Sources: {result['metadata']['num_sources']}")
            print(f"   Time: {result['metadata']['total_time']:.2f}s")
        
        print("\n✓ Setup and testing completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python main.py cli' for interactive mode")
        print("2. Run 'python main.py api' to start web server")
        print("3. Add more documents using the CLI or API")
        
    except Exception as e:
        print(f"✗ Setup failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Pull the required model: ollama pull llama2")
        print("3. Install dependencies: pip install -r requirements.txt")
        sys.exit(1)

def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking dependencies...")
    
    missing_deps = []
    
    try:
        import ollama
    except ImportError:
        missing_deps.append("ollama")
    
    try:
        import chromadb
    except ImportError:
        missing_deps.append("chromadb")
    
    try:
        import sentence_transformers
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        missing_deps.append("langchain-text-splitters")
    
    try:
        import flask
    except ImportError:
        missing_deps.append("flask")
    
    if missing_deps:
        print(f"✗ Missing dependencies: {', '.join(missing_deps)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("✓ All dependencies are available")
    return True

def show_help():
    """Show help information"""
    help_text = """
Hybrid RAG System - AI-Powered Document Retrieval and Question Answering

Usage: python main.py [command] [options]

Commands:
  setup     - Setup system with sample documents and run tests
  api       - Start the web API server
  cli       - Start interactive command-line interface
  check     - Check system dependencies
  help      - Show this help message

Examples:
  python main.py setup          # First-time setup and testing
  python main.py api            # Start web server
  python main.py cli            # Interactive mode
  python main.py cli query "What is AI?"  # Single query

Features:
- Hybrid retrieval (dense + sparse)
- Multiple document formats (PDF, TXT, DOCX, MD)
- Ollama integration for LLM responses
- Vector database (ChromaDB/FAISS)
- BM25 sparse retrieval
- REST API and CLI interfaces
- Configurable parameters

Requirements:
- Python 3.8+
- Ollama server running
- Required Python packages (see requirements.txt)

For more information, visit: https://github.com/your-repo/hybrid-rag
    """
    print(help_text)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Hybrid RAG System - AI-Powered Document Retrieval",
        add_help=False
    )
    
    parser.add_argument('command', nargs='?', default='help',
                       choices=['setup', 'api', 'cli', 'check', 'help'],
                       help='Command to execute')
    
    # Parse known args to allow CLI subcommands
    args, remaining = parser.parse_known_args()
    
    # Execute command
    if args.command == 'setup':
        if not check_dependencies():
            sys.exit(1)
        setup_and_test()
    
    elif args.command == 'api':
        if not check_dependencies():
            sys.exit(1)
        run_api_server()
    
    elif args.command == 'cli':
        if not check_dependencies():
            sys.exit(1)
        # Pass remaining args to CLI
        sys.argv = ['cli.py'] + remaining
        run_cli()
    
    elif args.command == 'check':
        check_dependencies()
    
    elif args.command == 'help':
        show_help()
    
    else:
        show_help()

if __name__ == '__main__':
    main()