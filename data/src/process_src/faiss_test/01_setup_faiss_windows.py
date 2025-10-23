#!/usr/bin/env python3
"""
FAISS Installation and Setup Script for Windows 11
Automates the installation of FAISS and required dependencies
"""

import subprocess
import sys
import platform
import os

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print("Please upgrade your Python installation.")
        return False
    
    print("✓ Python version is compatible")
    return True

def check_os():
    """Verify running on Windows"""
    system = platform.system()
    print(f"Operating System: {system}")
    
    if system != "Windows":
        print("⚠ This script is designed for Windows 11")
        return False
    
    print("✓ Windows detected")
    return True

def install_package(package_name, display_name=None):
    """Install a Python package using pip"""
    if display_name is None:
        display_name = package_name
    
    print(f"\n📦 Installing {display_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--upgrade"])
        print(f"✓ {display_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {display_name}: {e}")
        return False

def main():
    print("="*70)
    print("FAISS Setup for Windows 11 - PyTorch Lightning RAG Project")
    print("="*70)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_os():
        print("⚠ Continuing anyway...")
    
    print("\n" + "="*70)
    print("Installing Required Packages")
    print("="*70)
    
    # List of packages to install
    packages = [
        # Core packages
        ("pip", "pip (upgrade)"),
        ("wheel", "wheel"),
        ("setuptools", "setuptools"),
        
        # FAISS (CPU version for Windows compatibility)
        ("faiss-cpu", "FAISS (CPU version)"),
        
        # Embedding and NLP packages
        ("sentence-transformers", "Sentence Transformers"),
        ("transformers", "Hugging Face Transformers"),
        ("torch", "PyTorch"),
        
        # Data processing
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scikit-learn", "Scikit-learn"),
        
        # Utilities
        ("tqdm", "tqdm (progress bars)"),
        ("jsonlines", "JSON Lines"),
    ]
    
    failed_packages = []
    
    for package_info in packages:
        if len(package_info) == 2:
            package, display = package_info
        else:
            package = display = package_info
        
        if not install_package(package, display):
            failed_packages.append(display)
    
    print("\n" + "="*70)
    print("Installation Summary")
    print("="*70)
    
    if failed_packages:
        print(f"❌ {len(failed_packages)} package(s) failed to install:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
        print("\nPlease install these packages manually:")
        for pkg in failed_packages:
            print(f"   pip install {pkg}")
    else:
        print("✓ All packages installed successfully!")
    
    # Verify FAISS installation
    print("\n" + "="*70)
    print("Verifying FAISS Installation")
    print("="*70)
    
    try:
        import faiss
        print(f"✓ FAISS version: {faiss.__version__}")
        print(f"✓ FAISS CPU support: Available")
        
        # Test basic FAISS functionality
        d = 64  # dimension
        nb = 100  # database size
        import numpy as np
        np.random.seed(1234)
        xb = np.random.random((nb, d)).astype('float32')
        
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        print(f"✓ FAISS test index created with {index.ntotal} vectors")
        print("✓ FAISS is working correctly!")
        
    except ImportError:
        print("❌ FAISS could not be imported")
        print("Try installing manually: pip install faiss-cpu")
    except Exception as e:
        print(f"❌ FAISS test failed: {e}")
    
    print("\n" + "="*70)
    print("Setup Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run: python 02_test_faiss_retrieval.py <path_to_dataset>")
    print("2. Create manual baseline: python 03_create_manual_baseline.py")
    print("3. Evaluate results: python 04_evaluate_retrieval.py")
    print("="*70)

if __name__ == "__main__":
    main()
