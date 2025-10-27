"""
Demo script for the Hybrid RAG System.
Showcases the capabilities of the system with sample queries and comparisons.
"""

import sys
import time
import os
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.hybrid_rag import HybridRAGSystem
import config


def print_banner():
    """Print demo banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                    🚀 HYBRID RAG SYSTEM DEMO                 ║
║                                                               ║
║  AI-Powered Document Retrieval & Question Answering          ║
║  Combining Dense Vector Search + Sparse BM25 Retrieval       ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_section(title, char="="):
    """Print a section header"""
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}")


def demonstrate_retrieval_methods(rag_system):
    """Demonstrate different retrieval methods"""
    print_section("🔍 RETRIEVAL METHODS COMPARISON")
    
    test_query = "What is machine learning and how does it work?"
    print(f"Query: '{test_query}'")
    
    methods = [
        ("Dense (Vector Similarity)", "dense"),
        ("Sparse (BM25 Keywords)", "sparse"),  
        ("Hybrid (Combined)", "hybrid")
    ]
    
    results = {}
    
    for method_name, method_code in methods:
        print(f"\n📊 {method_name}:")
        print("-" * 40)
        
        start_time = time.time()
        result = rag_system.query(test_query, method_code)
        end_time = time.time()
        
        results[method_code] = result
        
        print(f"Time: {end_time - start_time:.2f}s")
        print(f"Sources: {len(result['sources'])}")
        print(f"Answer Preview: {result['answer'][:100]}...")
        
        # Show top source
        if result['sources']:
            top_source = result['sources'][0]
            score_key = 'combined_score' if method_code == 'hybrid' else \
                       'similarity_score' if method_code == 'dense' else 'bm25_score'
            score = top_source.get(score_key, 0)
            print(f"Top Source Score: {score:.3f}")
    
    return results


def demonstrate_question_types(rag_system):
    """Demonstrate different types of questions"""
    print_section("❓ DIFFERENT QUESTION TYPES")
    
    question_types = [
        ("Factual", "What is artificial intelligence?"),
        ("Conceptual", "How does deep learning differ from traditional machine learning?"),
        ("Technical", "What are the key features of Python programming language?"),
        ("Comparative", "What's the difference between supervised and unsupervised learning?"),
        ("Practical", "How can I get started with Python programming?")
    ]
    
    for qtype, question in question_types:
        print(f"\n📝 {qtype} Question:")
        print(f"Q: {question}")
        print("-" * 50)
        
        result = rag_system.query(question, "hybrid")
        
        print(f"A: {result['answer'][:200]}...")
        print(f"Sources: {len(result['sources'])}")
        print(f"Time: {result['metadata']['total_time']:.2f}s")


def show_system_capabilities(rag_system):
    """Show system capabilities and statistics"""
    print_section("📊 SYSTEM CAPABILITIES")
    
    stats = rag_system.get_system_stats()
    
    print("🏗️  Architecture:")
    print(f"   • Vector Database: {stats['vector_database']['db_type']}")
    print(f"   • Embedding Model: {config.EMBEDDING_MODEL}")
    print(f"   • LLM Model: {stats['model_name']}")
    print(f"   • Database: SQLite")
    
    print(f"\n📚 Knowledge Base:")
    print(f"   • Total Documents: {stats['total_documents']}")
    print(f"   • Vector Embeddings: {stats['vector_database']['total_vectors']}")
    print(f"   • Sparse Terms: {stats['sparse_index']['total_terms']}")
    print(f"   • Avg Document Length: {stats['sparse_index']['average_document_length']:.0f} tokens")
    
    print(f"\n⚙️  Configuration:")
    print(f"   • Chunk Size: {config.CHUNK_SIZE} characters")
    print(f"   • Chunk Overlap: {config.CHUNK_OVERLAP} characters")
    print(f"   • Dense Weight: {config.DENSE_WEIGHT}")
    print(f"   • Sparse Weight: {config.SPARSE_WEIGHT}")


def demonstrate_document_processing(rag_system):
    """Show document processing capabilities"""
    print_section("📄 DOCUMENT PROCESSING")
    
    documents = rag_system.db_manager.get_all_documents()
    
    print("Supported Formats:")
    print("   • PDF documents")
    print("   • Word documents (DOCX, DOC)")
    print("   • Text files (TXT)")
    print("   • Markdown files (MD)")
    print("   • HTML files")
    
    if documents:
        print(f"\nCurrently Loaded Documents ({len(documents)}):")
        for doc in documents:
            chunks = rag_system.db_manager.get_chunks_by_document(doc.id)
            print(f"   • {doc.filename} ({len(chunks)} chunks, {doc.status})")
    else:
        print("\n⚠️  No documents currently loaded.")
        print("   Add documents with: python main.py cli add <file_path>")


def run_interactive_demo(rag_system):
    """Run interactive demo mode"""
    print_section("🎮 INTERACTIVE DEMO MODE")
    
    print("Try asking questions about the loaded documents!")
    print("Commands:")
    print("  • Type your question and press Enter")
    print("  • 'stats' - Show system statistics")
    print("  • 'docs' - List loaded documents")
    print("  • 'quit' - Exit demo")
    
    while True:
        try:
            user_input = input("\n🤖 Ask me anything: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for trying the Hybrid RAG System!")
                break
            
            if user_input.lower() == 'stats':
                show_system_capabilities(rag_system)
                continue
            
            if user_input.lower() == 'docs':
                demonstrate_document_processing(rag_system)
                continue
            
            if not user_input:
                continue
            
            print("\n🔍 Searching...")
            result = rag_system.query(user_input, "hybrid")
            
            print(f"\n📋 Answer:")
            print(f"{result['answer']}")
            
            print(f"\n📚 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'][:2], 1):
                preview = source['content'][:100] + "..." if len(source['content']) > 100 else source['content']
                score = source.get('combined_score', 0)
                print(f"   {i}. Score: {score:.3f} | {preview}")
            
            print(f"\n⏱️  Performance:")
            print(f"   Total: {result['metadata']['total_time']:.2f}s | "
                  f"Retrieval: {result['metadata']['retrieval_time']:.2f}s | "
                  f"Generation: {result['metadata']['generation_time']:.2f}s")
            
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def main():
    """Main demo function"""
    print_banner()
    
    try:
        # Initialize RAG system
        print("🚀 Initializing Hybrid RAG System...")
        rag_system = HybridRAGSystem()
        print("✅ System initialized successfully!")
        
        # Check if documents are loaded
        documents = rag_system.db_manager.get_all_documents()
        if not documents:
            print("\n⚠️  No documents found in the system.")
            print("📁 Loading sample documents...")
            
            documents_dir = Path("./documents")
            if documents_dir.exists():
                doc_ids = rag_system.add_documents_from_directory(str(documents_dir))
                print(f"✅ Loaded {len(doc_ids)} sample documents")
            else:
                print("❌ No documents directory found.")
                print("   Create some documents in ./documents/ or run: python main.py setup")
                return
        
        # Show system capabilities
        show_system_capabilities(rag_system)
        
        # Demonstrate document processing
        demonstrate_document_processing(rag_system)
        
        # Demonstrate retrieval methods
        demonstrate_retrieval_methods(rag_system)
        
        # Demonstrate question types
        demonstrate_question_types(rag_system)
        
        # Ask user if they want interactive mode
        print_section("🎮 INTERACTIVE MODE")
        response = input("Would you like to try interactive mode? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            run_interactive_demo(rag_system)
        else:
            print("\n🎉 Demo completed!")
            print("\nNext steps:")
            print("  • Run 'python main.py cli' for full CLI interface")
            print("  • Run 'python main.py api' to start web server")
            print("  • Visit http://localhost:5000 for web interface")
    
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("  1. Make sure Ollama is running: ollama serve")
        print("  2. Check dependencies: python main.py check")
        print("  3. Run setup: python main.py setup")


if __name__ == "__main__":
    main()