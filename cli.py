"""
Command-line interface for the Hybrid RAG System.
Provides interactive CLI for document management and querying.
"""

import argparse
import sys
import os
from pathlib import Path
import json
from typing import List

# Add src directory to path
current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, current_dir)
sys.path.insert(0, src_dir)

from src.hybrid_rag import HybridRAGSystem
import config


class RAGCli:
    """Command-line interface for the RAG system"""
    
    def __init__(self):
        self.rag_system = None
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the RAG system"""
        try:
            print("Initializing Hybrid RAG System...")
            self.rag_system = HybridRAGSystem()
            print("✓ RAG system initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing RAG system: {str(e)}")
            print("Make sure Ollama is running and required dependencies are installed.")
            sys.exit(1)
    
    def add_document(self, file_path: str):
        """Add a single document"""
        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            return
        
        print(f"Processing document: {file_path}")
        doc_id = self.rag_system.add_document(file_path)
        
        if doc_id:
            print(f"✓ Document added successfully (ID: {doc_id})")
        else:
            print("✗ Failed to add document")
    
    def add_directory(self, directory_path: str):
        """Add all documents from a directory"""
        if not os.path.exists(directory_path):
            print(f"✗ Directory not found: {directory_path}")
            return
        
        print(f"Processing directory: {directory_path}")
        doc_ids = self.rag_system.add_documents_from_directory(directory_path)
        
        if doc_ids:
            print(f"✓ Added {len(doc_ids)} documents successfully")
            print(f"Document IDs: {doc_ids}")
        else:
            print("✗ No documents were processed")
    
    def query_interactive(self):
        """Interactive query mode"""
        print("\n" + "="*60)
        print("Interactive Query Mode")
        print("Type 'quit' or 'exit' to stop")
        print("Type 'help' for available commands")
        print("="*60 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("\nQuery: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                
                if not user_input:
                    continue
                
                # Parse retrieval method (optional)
                retrieval_method = "hybrid"
                if user_input.startswith("dense:"):
                    retrieval_method = "dense"
                    user_input = user_input[6:].strip()
                elif user_input.startswith("sparse:"):
                    retrieval_method = "sparse"
                    user_input = user_input[7:].strip()
                elif user_input.startswith("hybrid:"):
                    retrieval_method = "hybrid"
                    user_input = user_input[7:].strip()
                
                # Perform query
                print(f"\nSearching using {retrieval_method} retrieval...")
                result = self.rag_system.query(user_input, retrieval_method)
                
                # Display results
                self.display_query_result(result)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"✗ Error: {str(e)}")
    
    def query_single(self, question: str, retrieval_method: str = "hybrid"):
        """Perform a single query"""
        print(f"Query: {question}")
        print(f"Method: {retrieval_method}")
        print("-" * 50)
        
        result = self.rag_system.query(question, retrieval_method)
        self.display_query_result(result)
    
    def display_query_result(self, result: dict):
        """Display query results in a formatted way"""
        print(f"\n📋 Answer:")
        print("-" * 50)
        print(result['answer'])
        
        print(f"\n📚 Sources ({len(result['sources'])}):")
        print("-" * 50)
        for i, source in enumerate(result['sources'][:3]):  # Show top 3 sources
            print(f"\n{i+1}. Score: {source.get('combined_score', source.get('similarity_score', source.get('bm25_score', 0))):.3f}")
            content_preview = source['content'][:200]
            if len(source['content']) > 200:
                content_preview += "..."
            print(f"   Content: {content_preview}")
        
        print(f"\n⏱️  Metadata:")
        print("-" * 50)
        metadata = result['metadata']
        print(f"Total Time: {metadata['total_time']:.2f}s")
        print(f"Retrieval Time: {metadata['retrieval_time']:.2f}s")
        print(f"Generation Time: {metadata['generation_time']:.2f}s")
        print(f"Sources Found: {metadata['num_sources']}")
    
    def show_help(self):
        """Show help information"""
        help_text = """
Available Commands:
- help          : Show this help message
- stats         : Show system statistics
- quit/exit/q   : Exit the program

Query Prefixes:
- dense: <query>    : Use dense (vector) retrieval only
- sparse: <query>   : Use sparse (BM25) retrieval only  
- hybrid: <query>   : Use hybrid retrieval (default)
- <query>          : Use hybrid retrieval

Examples:
- What is machine learning?
- dense: explain neural networks
- sparse: definition of AI
- hybrid: how does deep learning work?
        """
        print(help_text)
    
    def show_stats(self):
        """Show system statistics"""
        stats = self.rag_system.get_system_stats()
        
        print("\n📊 System Statistics:")
        print("-" * 50)
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Model: {stats['model_name']}")
        print(f"Database: {stats['database_path']}")
        
        print(f"\n🔍 Vector Database:")
        vector_stats = stats['vector_database']
        print(f"  Type: {vector_stats['db_type']}")
        print(f"  Total Vectors: {vector_stats['total_vectors']}")
        print(f"  Embedding Dimension: {vector_stats['embedding_dimension']}")
        
        print(f"\n📝 Sparse Index:")
        sparse_stats = stats['sparse_index']
        print(f"  Total Documents: {sparse_stats['total_documents']}")
        print(f"  Total Terms: {sparse_stats['total_terms']}")
        print(f"  Average Doc Length: {sparse_stats['average_document_length']:.1f}")
    
    def list_documents(self):
        """List all documents in the system"""
        documents = self.rag_system.db_manager.get_all_documents()
        
        if not documents:
            print("No documents found in the system.")
            return
        
        print(f"\n📄 Documents ({len(documents)}):")
        print("-" * 80)
        print(f"{'ID':<5} {'Filename':<30} {'Status':<12} {'Chunks':<8} {'Size':<10}")
        print("-" * 80)
        
        for doc in documents:
            chunks = self.rag_system.db_manager.get_chunks_by_document(doc.id)
            size_str = f"{doc.file_size:,} B" if doc.file_size else "N/A"
            filename = doc.filename[:27] + "..." if len(doc.filename) > 30 else doc.filename
            
            print(f"{doc.id:<5} {filename:<30} {doc.status:<12} {len(chunks):<8} {size_str:<10}")
    
    def rebuild_indices(self):
        """Rebuild vector and sparse indices"""
        print("Rebuilding indices...")
        self.rag_system.rebuild_indices()
        print("✓ Indices rebuilt successfully")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Hybrid RAG System CLI")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add document command
    add_parser = subparsers.add_parser('add', help='Add documents to the RAG system')
    add_parser.add_argument('path', help='File or directory path to add')
    add_parser.add_argument('--recursive', '-r', action='store_true', 
                           help='Process directory recursively')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('question', nargs='?', help='Question to ask')
    query_parser.add_argument('--method', '-m', choices=['dense', 'sparse', 'hybrid'],
                             default='hybrid', help='Retrieval method')
    query_parser.add_argument('--interactive', '-i', action='store_true',
                             help='Start interactive query mode')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List documents in the system')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show system statistics')
    
    # Rebuild command
    rebuild_parser = subparsers.add_parser('rebuild', help='Rebuild indices')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize CLI
    cli = RAGCli()
    
    # Execute command
    if args.command == 'add':
        if os.path.isfile(args.path):
            cli.add_document(args.path)
        elif os.path.isdir(args.path):
            cli.add_directory(args.path)
        else:
            print(f"✗ Path not found: {args.path}")
    
    elif args.command == 'query':
        if args.interactive or not args.question:
            cli.query_interactive()
        else:
            cli.query_single(args.question, args.method)
    
    elif args.command == 'list':
        cli.list_documents()
    
    elif args.command == 'stats':
        cli.show_stats()
    
    elif args.command == 'rebuild':
        cli.rebuild_indices()
    
    else:
        # No command provided, show help
        parser.print_help()


if __name__ == '__main__':
    main()