"""
Streamlit UI for Voice-based Chatbot with LangChain RAG
User interface for interacting with the voicebot
"""

import streamlit as st
import os
from voicebot_logic import VoicebotRAG
import speech_recognition as sr
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="🎤 Voice Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .chat-container {
        max-width: 100%;
        margin: 20px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .bot-message {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4caf50;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if "voicebot" not in st.session_state:
    st.session_state.voicebot = None
    st.session_state.chat_history = []
    st.session_state.initialized = False


def initialize_voicebot():
    """Initialize the voicebot with RAG"""
    api_key = st.session_state.groq_key
    docs_path = st.session_state.docs_path
    model = st.session_state.model  # Use selected model from UI
    
    if not api_key:
        st.error("❌ Please enter Groq API Key")
        return False
    
    if not docs_path or not os.path.exists(docs_path):
        st.error("❌ Please provide a valid documents directory path")
        return False
    
    try:
        with st.spinner("🔄 Initializing voicebot..."):
            voicebot = VoicebotRAG(groq_api_key=api_key, model_name=model)
            voicebot.load_documents(docs_path)
            voicebot.initialize_qa_chain()
            st.session_state.voicebot = voicebot
            st.session_state.initialized = True
        st.success("✅ Voicebot initialized successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Error initializing voicebot: {e}")
        return False


def process_text_input(user_input: str):
    """Process text input and get response"""
    if not st.session_state.initialized:
        st.error("Voicebot not initialized. Please click 'Initialize Voicebot' first.")
        return
    
    try:
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.voicebot.process_text_query(user_input)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "type": "user",
            "content": user_input,
            "mode": "text"
        })
        st.session_state.chat_history.append({
            "type": "bot",
            "content": response,
            "mode": "text"
        })
        print(f"✅ Added to chat history - User: {user_input}, Bot: {response[:100]}...")
    except Exception as e:
        print(f"❌ Error in process_text_input: {e}")
        st.error(f"Error processing query: {e}")


def display_chat_history():
    """Display chat history"""
    for message in st.session_state.chat_history:
        if message["type"] == "user":
            st.markdown(
                f"<div class='user-message'><b>📝 You:</b> {message['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='bot-message'><b>🤖 Bot:</b> {message['content']}</div>",
                unsafe_allow_html=True
            )


def main():
    # Header
    st.markdown("<h1 class='main-header'>🎤 AI Voice Chatbot - LangChain RAG</h1>", unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            help="Enter your Groq API key from https://console.groq.com"
        )
        st.session_state.groq_key = api_key
        
        # Documents path input
        docs_path = st.text_input(
            "📁 Documents Directory Path",
            value="./documents",
            help="Path to directory containing your knowledge documents"
        )
        st.session_state.docs_path = docs_path
        
        # Model selection
        model = st.selectbox(
            "🤖 Groq Language Model",
            [
                "llama-3.1-8b-instant",
                "llama-3.2-90b-vision-preview",
                "llama-3.2-70b-8192",
                "deepseek-r1-distill-llama-70b",
                "openai/gpt-oss-120b",
                "qwen/qwen3-32b",
                "moonshotai/kimi-k2-instruct-0905"
            ],
            help="Select the Groq LLM to use (check https://console.groq.com/docs/models for latest)"
        )
        st.session_state.model = model
        
        st.divider()
        
        # Initialize button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Initialize Voicebot", use_container_width=True):
                initialize_voicebot()
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Status
        st.divider()
        if st.session_state.initialized:
            st.markdown(
                "<div class='status-box' style='background-color: #c8e6c9; color: #2e7d32;'>"
                "<b>✅ Status: Initialized</b></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='status-box' style='background-color: #ffcdd2; color: #c62828;'>"
                "<b>❌ Status: Not Initialized</b></div>",
                unsafe_allow_html=True
            )
        
        st.divider()
        st.subheader("📖 Instructions")
        st.markdown("""
        1. **Setup**: Enter your OpenAI API key and documents directory
        2. **Initialize**: Click 'Initialize Voicebot'
        3. **Query**: Choose between text or voice input
        4. **Interact**: Ask questions about your documents
        
        **Note**: Voice input requires a microphone and internet connection
        """)
    
    # Main content area
    if not st.session_state.initialized:
        st.info("👈 Configure your API key and documents path in the sidebar, then click 'Initialize Voicebot'")
    else:
        # Tabs for input methods
        tab1, tab2, tab3 = st.tabs(["💬 Text Chat", "🎤 Voice Chat", "📊 Chat History"])
        
        with tab1:
            st.subheader("Text-Based Chat")
            
            # Display chat history
            if st.session_state.chat_history:
                st.write("---")
                for message in st.session_state.chat_history:
                    if message["type"] == "user":
                        st.markdown(
                            f"<div class='user-message'><b>📝 You:</b> {message['content']}</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='bot-message'><b>🤖 Bot:</b> {message['content']}</div>",
                            unsafe_allow_html=True
                        )
                st.write("---")
            
            # Input area
            col1, col2 = st.columns([4, 1])
            with col1:
                user_input = st.text_input(
                    "Ask me anything about your documents:",
                    placeholder="Type your question here..."
                )
            with col2:
                send_btn = st.button("Send", use_container_width=True)
            
            if send_btn and user_input:
                process_text_input(user_input)
                st.rerun()
        
        with tab2:
            st.subheader("🎤 Voice-Based Chat")
            
            st.info("""
            **How to use:**
            1. **Record Audio**: Use your device's audio recorder to create an audio file
            2. **Upload Audio**: Upload the audio file (WAV, MP3, etc.)
            3. **Get Response**: AI will transcribe and respond in voice format
            """)
            
            st.subheader("📁 Upload & Process Audio")
            audio_file = st.file_uploader(
                "Upload audio file (WAV, MP3, M4A, FLAC, OGG)",
                type=["wav", "mp3", "m4a", "flac", "ogg"],
                key="voice_upload"
            )
            
            if audio_file:
                # Display the audio player
                st.audio(audio_file)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🎯 Transcribe & Respond", key="process_audio", use_container_width=True):
                        if not st.session_state.initialized:
                            st.error("❌ Voicebot not initialized. Initialize first in the sidebar!")
                        else:
                            try:
                                with st.spinner("🔄 Processing audio..."):
                                    audio_data = audio_file.read()
                                    result = st.session_state.voicebot.process_voice_query(audio_data)
                                    
                                    # Display results
                                    st.success("✅ Processing complete!")
                                    st.write("---")
                                    st.write(f"**🗣️ Your question:** {result['user_input']}")
                                    st.write(f"**🤖 Bot response:** {result['answer']}")
                                    st.write("---")
                                    
                            except Exception as e:
                                st.error(f"❌ Error processing audio: {e}")
                
                with col2:
                    if st.button("🔊 Convert Response to Audio", key="tts_btn", use_container_width=True):
                        if not st.session_state.chat_history or not st.session_state.chat_history[-1]['type'] == 'bot':
                            st.warning("⚠️ No response to convert. Process an audio file first!")
                        else:
                            try:
                                last_response = st.session_state.chat_history[-1]['content']
                                with st.spinner("🔊 Generating voice response..."):
                                    audio_path = st.session_state.voicebot.text_to_speech(last_response)
                                    if audio_path and os.path.exists(audio_path):
                                        st.success("✅ Voice response ready!")
                                        with open(audio_path, 'rb') as f:
                                            st.audio(f.read(), format="audio/mp3")
                                    else:
                                        st.error("❌ Failed to generate audio")
                            except Exception as e:
                                st.error(f"❌ Error generating audio: {e}")
            else:
                st.info("👆 Upload an audio file to get started!")
        
        with tab3:
            st.subheader("📊 Conversation History")
            if st.session_state.chat_history:
                display_chat_history()
            else:
                st.info("No messages yet. Start by asking a question in the Text Chat or Voice Chat tab!")


if __name__ == "__main__":
    main()
