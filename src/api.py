"""
Flask REST API for the Hybrid RAG System.
Provides endpoints for document management, querying, and system administration.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path
import traceback
from werkzeug.utils import secure_filename

# Local imports
from .hybrid_rag import HybridRAGSystem
try:
    from .. import config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config

# Initialize Flask app
app = Flask(__name__, static_folder='../static', static_url_path='')
CORS(app)  # Enable CORS for frontend integration

# Initialize RAG system
rag_system = None

def initialize_rag_system():
    """Initialize the RAG system"""
    global rag_system
    try:
        rag_system = HybridRAGSystem()
        print("✓ RAG system initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Error initializing RAG system: {str(e)}")
        return False

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.doc', '.md', '.html', '.htm'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the web interface"""
    return app.send_static_file('index.html')

@app.route('/api')
def api_info():
    """API health check"""
    return jsonify({
        'status': 'running',
        'message': 'Hybrid RAG System API',
        'version': '1.0.0'
    })

@app.route('/api/health')
def health():
    """Detailed health check"""
    try:
        stats = rag_system.get_system_stats() if rag_system else None
        return jsonify({
            'status': 'healthy' if rag_system else 'unhealthy',
            'system_stats': stats,
            'timestamp': str(Path().resolve())
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/query', methods=['POST'])
def query():
    """Main query endpoint"""
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question in request body'}), 400
        
        question = data['question']
        retrieval_method = data.get('retrieval_method', 'hybrid')
        
        # Validate retrieval method
        if retrieval_method not in ['dense', 'sparse', 'hybrid']:
            return jsonify({'error': 'Invalid retrieval method. Use: dense, sparse, or hybrid'}), 400
        
        # Perform query
        result = rag_system.query(question, retrieval_method)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/documents', methods=['POST'])
def upload_document():
    """Upload and process a document"""
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Unsupported file type. Allowed: {list(ALLOWED_EXTENSIONS)}'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = Path('./documents') / filename
        upload_path.parent.mkdir(exist_ok=True)
        file.save(str(upload_path))
        
        # Process document
        doc_id = rag_system.add_document(str(upload_path))
        
        if doc_id:
            return jsonify({
                'message': 'Document uploaded and processed successfully',
                'document_id': doc_id,
                'filename': filename
            })
        else:
            return jsonify({'error': 'Failed to process document'}), 500
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/documents/directory', methods=['POST'])
def process_directory():
    """Process all documents in a directory"""
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'directory_path' not in data:
            return jsonify({'error': 'Missing directory_path in request body'}), 400
        
        directory_path = data['directory_path']
        
        if not os.path.exists(directory_path):
            return jsonify({'error': 'Directory not found'}), 404
        
        # Process directory
        doc_ids = rag_system.add_documents_from_directory(directory_path)
        
        return jsonify({
            'message': f'Processed {len(doc_ids)} documents successfully',
            'document_ids': doc_ids,
            'directory_path': directory_path
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all documents in the system"""
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        documents = rag_system.db_manager.get_all_documents()
        
        doc_list = []
        for doc in documents:
            chunks = rag_system.db_manager.get_chunks_by_document(doc.id)
            doc_info = {
                'id': doc.id,
                'filename': doc.filename,
                'title': doc.title,
                'content_type': doc.content_type,
                'file_size': doc.file_size,
                'upload_date': doc.upload_date.isoformat() if doc.upload_date else None,
                'processed_date': doc.processed_date.isoformat() if doc.processed_date else None,
                'status': doc.status,
                'num_chunks': len(chunks)
            }
            doc_list.append(doc_info)
        
        return jsonify({
            'documents': doc_list,
            'total_count': len(doc_list)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/documents/<int:doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get details of a specific document"""
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        document = rag_system.db_manager.get_document(doc_id)
        
        if not document:
            return jsonify({'error': 'Document not found'}), 404
        
        chunks = rag_system.db_manager.get_chunks_by_document(doc_id)
        
        doc_info = {
            'id': document.id,
            'filename': document.filename,
            'title': document.title,
            'content_type': document.content_type,
            'file_path': document.file_path,
            'file_size': document.file_size,
            'upload_date': document.upload_date.isoformat() if document.upload_date else None,
            'processed_date': document.processed_date.isoformat() if document.processed_date else None,
            'status': document.status,
            'chunks': [
                {
                    'id': chunk.id,
                    'chunk_index': chunk.chunk_index,
                    'content_preview': chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content,
                    'content_length': chunk.content_length
                }
                for chunk in chunks
            ]
        }
        
        return jsonify(doc_info)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/search/dense', methods=['POST'])
def dense_search():
    """Perform dense (vector) search only"""
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query in request body'}), 400
        
        query = data['query']
        top_k = data.get('top_k', config.TOP_K_DENSE)
        
        results = rag_system.dense_retrieval(query, top_k)
        
        return jsonify({
            'query': query,
            'method': 'dense',
            'results': results,
            'num_results': len(results)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/search/sparse', methods=['POST'])
def sparse_search():
    """Perform sparse (BM25) search only"""
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query in request body'}), 400
        
        query = data['query']
        top_k = data.get('top_k', config.TOP_K_SPARSE)
        
        results = rag_system.sparse_retrieval(query, top_k)
        
        return jsonify({
            'query': query,
            'method': 'sparse',
            'results': results,
            'num_results': len(results)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/system/stats')
def system_stats():
    """Get system statistics"""
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        stats = rag_system.get_system_stats()
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/system/rebuild-indices', methods=['POST'])
def rebuild_indices():
    """Rebuild vector and sparse indices"""
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        rag_system.rebuild_indices()
        
        return jsonify({
            'message': 'Indices rebuilt successfully'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/config')
def get_config():
    """Get current system configuration"""
    return jsonify({
        'ollama_model': config.OLLAMA_MODEL,
        'vector_db_type': config.VECTOR_DB_TYPE,
        'embedding_model': config.EMBEDDING_MODEL,
        'chunk_size': config.CHUNK_SIZE,
        'chunk_overlap': config.CHUNK_OVERLAP,
        'top_k_dense': config.TOP_K_DENSE,
        'top_k_sparse': config.TOP_K_SPARSE,
        'top_k_final': config.TOP_K_FINAL,
        'dense_weight': config.DENSE_WEIGHT,
        'sparse_weight': config.SPARSE_WEIGHT
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize RAG system
    if initialize_rag_system():
        print(f"Starting server on {config.API_HOST}:{config.API_PORT}")
        app.run(
            host=config.API_HOST,
            port=config.API_PORT,
            debug=config.DEBUG_MODE
        )
    else:
        print("Failed to initialize RAG system. Exiting.")
        exit(1)