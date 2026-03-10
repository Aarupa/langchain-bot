"""
Main application entry point for the Voice-based Chatbot with LangChain RAG
Run this file with: streamlit run ui.py
"""

import os
import sys
from pathlib import Path
from config import create_directories

# Create necessary directories
create_directories()

# Import UI module
try:
    import streamlit as st
    from ui import main
    
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("\nPlease install required dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)
