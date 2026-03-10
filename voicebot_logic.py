"""
Voicebot Logic Module - LangChain RAG Integration
Handles all backend logic for the voice-based chatbot
"""

import os
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
import speech_recognition as sr
from pathlib import Path
from io import BytesIO
import pyttsx3


class VoicebotRAG:
    """Main class for managing voicebot RAG functionality"""
    
    def __init__(self, groq_api_key: str = None, model_name: str = "llama-3.1-70b-versatile"):
        """
        Initialize the voicebot with RAG capabilities using Groq
        
        Args:
            groq_api_key: Groq API key (uses env var if not provided)
            model_name: Groq model to use
        """
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.qa_chain = None
        self.recognizer = sr.Recognizer()
        
    def load_documents(self, docs_path: str) -> None:
        """
        Load documents from a directory for RAG
        Supports: .txt, .pdf, .md, .docx
        
        Args:
            docs_path: Path to directory containing documents
        """
        try:
            documents = []
            docs_dir = Path(docs_path)
            
            if not docs_dir.exists():
                raise FileNotFoundError(f"Documents directory not found: {docs_path}")
            
            # Load text files
            txt_files = list(docs_dir.glob("*.txt"))
            if txt_files:
                txt_loader = DirectoryLoader(docs_path, glob="*.txt", loader_cls=TextLoader)
                documents.extend(txt_loader.load())
            
            # Load PDF files
            pdf_files = list(docs_dir.glob("*.pdf"))
            if pdf_files:
                try:
                    pdf_loader = DirectoryLoader(docs_path, glob="*.pdf", loader_cls=PyPDFLoader)
                    documents.extend(pdf_loader.load())
                except Exception as e:
                    print(f"Warning: Error loading PDF files: {e}")
            
            # Load markdown files
            md_files = list(docs_dir.glob("*.md"))
            if md_files:
                md_loader = DirectoryLoader(docs_path, glob="*.md", loader_cls=TextLoader)
                documents.extend(md_loader.load())
            
            # Load DOCX files if python-docx is available
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                docx_files = list(docs_dir.glob("*.docx"))
                if docx_files:
                    for docx_file in docx_files:
                        loader = Docx2txtLoader(str(docx_file))
                        documents.extend(loader.load())
            except Exception as e:
                print(f"Warning: DOCX loading not available: {e}")
            
            if not documents:
                raise ValueError(f"No documents found in {docs_path}")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            print(f"✓ Loaded {len(chunks)} document chunks from {len(documents)} documents")
            
        except Exception as e:
            print(f"Error loading documents: {e}")
            raise
    
    def initialize_qa_chain(self, system_prompt: str = None) -> None:
        """
        Initialize the QA chain with RAG using modern LangChain LCEL
        
        Args:
            system_prompt: Custom system prompt for the LLM
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Load documents first.")
        
        llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.7
        )
        
        print(f"🤖 Using Groq Model: {self.model_name}")
        
        custom_prompt = system_prompt or """Use the following pieces of context to answer the question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}
Answer:"""
        
        prompt_template = PromptTemplate(
            template=custom_prompt,
            input_variables=["context", "question"]
        )
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Modern LangChain LCEL chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.qa_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt_template
            | llm
        )
    
    def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribe audio to text using speech recognition
        
        Args:
            audio_data: Audio bytes
            
        Returns:
            Transcribed text
        """
        try:
            audio = sr.AudioData(audio_data, 16000, 2)
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Error with speech recognition: {e}"
    
    def process_voice_query(self, audio_data: bytes) -> Dict[str, str]:
        """
        Process voice query end-to-end: transcribe and get answer
        
        Args:
            audio_data: Audio bytes
            
        Returns:
            Dictionary with transcribed_text and answer
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Initialize it first.")
        
        # Transcribe audio
        user_input = self.transcribe_audio(audio_data)
        
        # Get answer from RAG
        result = self.qa_chain.invoke(user_input)
        answer = result.content if hasattr(result, 'content') else str(result)
        
        return {
            "user_input": user_input,
            "answer": answer
        }
    
    def process_text_query(self, query: str) -> str:
        """
        Process text query (fallback option)
        
        Args:
            query: Text query
            
        Returns:
            Answer from RAG
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Initialize it first.")
        
        print(f"📝 Processing query with {self.model_name}: {query[:100]}...")
        result = self.qa_chain.invoke(query)
        # Extract text content from the response
        if hasattr(result, 'content'):
            response = result.content
        else:
            response = str(result)
        
        print(f"✅ Response: {response[:200]}..." if len(response) > 200 else f"✅ Response: {response}")
        return response
    
    def save_vector_store(self, path: str) -> None:
        """Save vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(path)
    
    def load_vector_store(self, path: str) -> None:
        """Load vector store from disk"""
        self.vector_store = FAISS.load_local(path, self.embeddings)
    
    def text_to_speech(self, text: str, output_file: str = "response.mp3") -> str:
        """
        Convert text to speech using pyttsx3
        
        Args:
            text: Text to convert
            output_file: Output audio file path
            
        Returns:
            Path to the audio file
        """
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Speed of speech
            engine.setProperty('volume', 1)   # Volume 0-1
            engine.save_to_file(text, output_file)
            engine.runAndWait()
            print(f"🔊 Audio saved to {output_file}")
            return output_file
        except Exception as e:
            print(f"Error converting text to speech: {e}")
            return None
