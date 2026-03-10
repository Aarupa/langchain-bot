"""
Configuration file for the voicebot application
Centralized settings for API keys, paths, and model parameters
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DOCS_PATH = os.getenv("DOCS_PATH", str(PROJECT_ROOT / "documents"))
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", str(PROJECT_ROOT / "vector_store"))

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# LangChain RAG Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
SEARCH_K = int(os.getenv("SEARCH_K", "3"))

# Speech Recognition Configuration
SPEECH_RECOGNITION_ENGINE = os.getenv("SPEECH_RECOGNITION_ENGINE", "google")  # google, azure, etc.

# Streamlit Configuration
STREAMLIT_THEME = "light"
PAGE_TITLE = "🎤 Voice Chatbot"
PAGE_ICON = "🤖"

# Custom prompt template for RAG
SYSTEM_PROMPT = """You are a helpful assistant powered by Retrieval Augmented Generation (RAG). 
Use the provided context to answer questions accurately and helpfully.
If you don't know the answer based on the provided context, say so clearly.

Context: {context}
Question: {question}
Answer:"""

# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist"""
    Path(DOCS_PATH).mkdir(parents=True, exist_ok=True)
    Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    create_directories()
    print(f"✓ Project Root: {PROJECT_ROOT}")
    print(f"✓ Documents Path: {DOCS_PATH}")
    print(f"✓ Vector Store Path: {VECTOR_STORE_PATH}")
