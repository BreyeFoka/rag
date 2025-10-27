# 🚀 Quick Start Guide

Get your Hybrid RAG System up and running in minutes!

## 📋 Prerequisites

1. **Python 3.8+** - [Download Python](https://python.org/downloads/)
2. **Ollama** - [Download Ollama](https://ollama.ai/download)

## ⚡ 5-Minute Setup

### Step 1: Install Ollama & Start Server
```bash
# Install Ollama (visit ollama.ai/download for your OS)

# Start Ollama server
ollama serve

# In another terminal, pull a model
ollama pull llama2
```

### Step 2: Setup RAG System
```bash
# Navigate to project directory
cd hybrid-rag-system

# Run automated setup
python setup.py

# OR install manually:
pip install -r requirements.txt
```

### Step 3: Initialize with Sample Documents
```bash
# Setup system with sample documents
python main.py setup
```

### Step 4: Start Using!

#### Option A: Web Interface (Recommended)
```bash
python main.py api
# Visit: http://localhost:5000
```

#### Option B: Interactive CLI
```bash
python main.py cli
# Type your questions interactively
```

#### Option C: Try the Demo
```bash
python demo.py
# Guided demonstration of features
```

## 🎯 First Query

Once running, try these example queries:

- `"What is artificial intelligence?"`
- `"Explain machine learning types"`
- `"What are Python's key features?"`
- `"How does deep learning work?"`

## 📄 Adding Your Documents

### Via Web Interface:
1. Visit http://localhost:5000
2. Drag & drop files to upload area
3. Supported: PDF, DOCX, TXT, MD, HTML

### Via CLI:
```bash
# Add single file
python main.py cli add path/to/document.pdf

# Add entire directory
python main.py cli add path/to/documents/ --recursive
```

### Via API:
```bash
curl -X POST http://localhost:5000/api/documents \
  -F "file=@document.pdf"
```

## 🔧 Quick Configuration

Edit `config.py` to customize:

```python
# Change AI model
OLLAMA_MODEL = "llama3"  # or "mistral", "codellama"

# Adjust chunk size
CHUNK_SIZE = 1500  # Default: 1000

# Change retrieval balance
DENSE_WEIGHT = 0.8  # More semantic search
SPARSE_WEIGHT = 0.2  # Less keyword search
```

## 📊 Check System Status

```bash
# Check dependencies
python main.py check

# View system stats
python main.py cli stats

# API health check
curl http://localhost:5000/api/health
```

## 🆘 Troubleshooting

### Problem: "Ollama not found"
**Solution:** Install Ollama from https://ollama.ai/download

### Problem: "Model not available"
**Solution:** 
```bash
ollama pull llama2
# or your preferred model
```

### Problem: "Import errors"
**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: "No documents found"
**Solution:**
```bash
# Add sample documents
python main.py setup

# Or add your own
python main.py cli add ./documents/
```

## 🎉 You're Ready!

Your Hybrid RAG System is now running with:
- ✅ Dense vector search (semantic similarity)
- ✅ Sparse keyword search (BM25)
- ✅ Hybrid retrieval (best of both)
- ✅ Local AI models (via Ollama)
- ✅ Web interface + API + CLI

## 📚 Next Steps

1. **Read the full README.md** for advanced features
2. **Explore the web interface** at http://localhost:5000
3. **Try different retrieval methods** (dense/sparse/hybrid)
4. **Add your own documents** and start querying!
5. **Customize configuration** in config.py

## 💡 Pro Tips

- Use **hybrid retrieval** for best results
- **Dense search** is better for conceptual queries
- **Sparse search** is better for exact keyword matches
- Larger **chunk sizes** work better for longer documents
- Check **system stats** to monitor performance

---

**Need help?** Check the [full documentation](README.md) or run `python main.py help`

**Happy RAG-ing! 🎯**