#!/usr/bin/env python3
"""
Installation and setup script for the Hybrid RAG System.
This script helps set up the environment and dependencies.
"""

import subprocess
import sys

from pathlib import Path


def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False


def check_ollama():
    """Check if Ollama is installed and running"""
    print("\n🤖 Checking Ollama installation...")
    
    # Check if ollama command exists
    if not run_command("ollama --version", "Checking Ollama version", check=False):
        print("   ❌ Ollama not found. Please install from: https://ollama.ai/download")
        return False
    
    # Check if ollama is running
    if not run_command("ollama list", "Checking if Ollama is running", check=False):
        print("   ⚠️  Ollama is installed but not running")
        print("   📋 Start Ollama with: ollama serve")
        return False
    
    print("   ✅ Ollama is installed and running")
    return True


def install_python_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("   ❌ requirements.txt not found")
        return False
    
    command = f"{sys.executable} -m pip install -r requirements.txt"
    return run_command(command, "Installing packages from requirements.txt")


def setup_ollama_model():
    """Download required Ollama model"""
    print("\n🔽 Setting up Ollama model...")
    
    # Default model
    model = "llama2"
    
    # Check if model exists
    result = subprocess.run("ollama list", shell=True, capture_output=True, text=True)
    if model in result.stdout:
        print(f"   ✅ Model '{model}' already available")
        return True
    
    print(f"   📥 Downloading model '{model}' (this may take several minutes)...")
    return run_command(f"ollama pull {model}", f"Downloading {model} model")


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ["data", "documents"]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir()
            print(f"   ✅ Created {dir_name}/ directory")
        else:
            print(f"   ✅ {dir_name}/ directory already exists")
    
    return True


def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    # Test imports
    test_imports = [
        ("ollama", "Ollama client"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("langchain", "LangChain"),
        ("flask", "Flask"),
        ("sqlalchemy", "SQLAlchemy")
    ]
    
    all_passed = True
    
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {description} - OK")
        except ImportError:
            print(f"   ❌ {description} - Failed to import")
            all_passed = False
    
    return all_passed


def show_next_steps():
    """Show next steps after installation"""
    print("\n🎉 Installation Complete!")
    print("\n📋 Next Steps:")
    print("   1. Make sure Ollama is running: ollama serve")
    print("   2. Run setup with sample documents: python main.py setup")
    print("   3. Start interactive mode: python main.py cli")
    print("   4. Start web server: python main.py api")
    print("   5. Access web interface: http://localhost:5000")
    
    print("\n🆘 Troubleshooting:")
    print("   - If Ollama fails: Visit https://ollama.ai/download")
    print("   - If imports fail: pip install -r requirements.txt")
    print("   - For help: python main.py help")
    
    print("\n📚 Documentation:")
    print("   - README.md for detailed instructions")
    print("   - config.py for system configuration")


def main():
    """Main setup function"""
    print("🚀 Hybrid RAG System Setup")
    print("=" * 50)
    
    # Track setup success
    setup_steps = []
    
    # Check Python version
    if check_python_version():
        setup_steps.append(("Python version", True))
    else:
        setup_steps.append(("Python version", False))
        print("\n❌ Setup failed: Incompatible Python version")
        sys.exit(1)
    
    # Create directories
    if create_directories():
        setup_steps.append(("Directories", True))
    else:
        setup_steps.append(("Directories", False))
    
    # Install Python dependencies
    if install_python_dependencies():
        setup_steps.append(("Python packages", True))
    else:
        setup_steps.append(("Python packages", False))
    
    # Test imports
    if test_installation():
        setup_steps.append(("Package imports", True))
    else:
        setup_steps.append(("Package imports", False))
    
    # Check Ollama
    if check_ollama():
        setup_steps.append(("Ollama", True))
        
        # Setup Ollama model
        if setup_ollama_model():
            setup_steps.append(("Ollama model", True))
        else:
            setup_steps.append(("Ollama model", False))
    else:
        setup_steps.append(("Ollama", False))
    
    # Show setup summary
    print("\n📊 Setup Summary:")
    print("-" * 30)
    
    all_success = True
    for step, success in setup_steps:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {step:<20} {status}")
        if not success:
            all_success = False
    
    print("-" * 30)
    
    if all_success:
        print("🎉 All setup steps completed successfully!")
        show_next_steps()
    else:
        print("⚠️  Some setup steps failed. Check the errors above.")
        print("   You may still be able to use parts of the system.")
        print("   Run 'python main.py check' to verify dependencies.")


if __name__ == "__main__":
    main()