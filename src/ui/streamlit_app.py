"""
Premium Streamlit UI for Research Assistant - OPTIMIZED VERSION
Fast startup, drag-and-drop files, clear progress, easy to use
"""

import hashlib
import logging
import os
from pathlib import Path
import sys
import traceback
from datetime import datetime
import time
import json

import streamlit as st
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.research_agent import ResearchAgent
from src.rag.vector_store import VectorStore
from src.rag.document_loader import load_document, chunk_text

# ============================================================
# PAGE CONFIG - MUST BE FIRST
# ============================================================

st.set_page_config(
    page_title="DocuMind",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS FOR PREMIUM LOOK
# ============================================================

def load_css():
    """Load custom CSS for premium styling"""
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* Global Styles */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Premium Header */
        .premium-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem 2.5rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            color: white;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .premium-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.5px;
        }
        
        .premium-header p {
            font-size: 1.1rem;
            opacity: 0.9;
            margin: 0.5rem 0 0 0;
        }
        
        .premium-header .badge {
            background: rgba(255, 255, 255, 255);
            padding: 0.25rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            display: inline-block;
            margin-top: 0.5rem;
        }
        
        /* Chat Messages */
        .chat-message {
            padding: 1.2rem 1.5rem;
            border-radius: 16px;
            margin-bottom: 1rem;
            animation: slideIn 0.3s ease-out;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .chat-message.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 20%;
            border-bottom-right-radius: 4px;
        }
        
        .chat-message.assistant {
            background: white;
            border: 1px solid #e2e8f0;
            margin-right: 20%;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        
        .chat-message .message-time {
            font-size: 0.7rem;
            opacity: 0.6;
            margin-top: 0.5rem;
        }
        
        .chat-message.user .message-time {
            color: rgba(255, 255, 255, 0.8);
        }
        
        /* Stats Cards */
        .stat-card {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid #e2e8f0;
            text-align: center;
            transition: all 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.05);
        }
        
        .stat-card .number {
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
        }
        
        .stat-card .label {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 0.25rem;
        }
        
        /* File Tags */
        .file-tag {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: #eef2ff;
            color: #667eea;
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 500;
            margin: 0.25rem 0;
            width: 100%;
        }
        
        .file-tag.success {
            background: #dcfce7;
            color: #16a34a;
        }
        
        .file-tag.warning {
            background: #fef3c7;
            color: #d97706;
        }
        
        /* Status Indicator */
        .status-online {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
            color: #22c55e;
        }
        
        .status-online::before {
            content: '';
            width: 8px;
            height: 8px;
            background: #22c55e;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            color: #94a3b8;
        }
        
        .empty-state .icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }
        
        .empty-state h3 {
            color: #334155;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        /* Tips Box */
        .tips-box {
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            font-size: 0.85rem;
            color: #1e40af;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .premium-header {
                padding: 1.5rem;
            }
            
            .premium-header h1 {
                font-size: 1.8rem;
            }
            
            .chat-message.user {
                margin-left: 0;
            }
            
            .chat-message.assistant {
                margin-right: 0;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# LOGGING SETUP
# ============================================================

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# ============================================================
# PERSISTENT FILE STORAGE & MODEL CACHE
# ============================================================

def get_upload_dir():
    """Get persistent directory for uploaded files"""
    upload_dir = Path("uploaded_files")
    upload_dir.mkdir(exist_ok=True)
    return upload_dir

def get_model_cache_file():
    """Get cached model name"""
    cache_file = Path(".model_cache.json")
    return cache_file

def load_cached_model():
    """Load previously selected model from cache"""
    cache_file = get_model_cache_file()
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get('model')
        except:
            pass
    return None

def save_cached_model(model_name):
    """Save selected model to cache"""
    cache_file = get_model_cache_file()
    try:
        with open(cache_file, 'w') as f:
            json.dump({'model': model_name}, f)
    except:
        pass

# ============================================================
# SIMPLE VECTOR STORE
# ============================================================

class SimpleVectorStore:
    """Simple in-memory vector store - no external dependencies"""
    def __init__(self):
        self.documents = []
    
    def add_documents(self, documents):
        """Add documents to store"""
        if isinstance(documents, list):
            self.documents.extend(documents)
        else:
            self.documents.append(documents)
    
    def search(self, query, top_k=5):
        """Search documents"""
        return self.documents[:top_k]
    
    def get_count(self):
        """Get total document count"""
        return len(self.documents)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def init_session():
    """Initialize session state with fast startup"""
    if "initialized" not in st.session_state:
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("GEMINI_API_KEY not found in .env file!")
                st.stop()
            
            # Initialize vector store (simple, fast)
            st.session_state.vector_store = SimpleVectorStore()
            
            # Initialize agent (lazy - won't test models yet)
            st.session_state.agent = ResearchAgent(api_key, st.session_state.vector_store)
            
            # Session state
            st.session_state.messages = []
            st.session_state.uploaded_files = []
            st.session_state.file_hashes = set()
            st.session_state.initialized = True
            st.session_state.processing = False
            
            logger.info("Session initialized successfully (fast startup)")
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            st.error(f"Initialization Error: {str(e)}")
            st.stop()

# ============================================================
# FILE PROCESSING - OPTIMIZED
# ============================================================

def get_file_hash(content: bytes) -> str:
    """Get MD5 hash of file content."""
    return hashlib.md5(content).hexdigest()

def get_file_size_mb(content: bytes) -> float:
    """Get file size in MB"""
    return len(content) / (1024 * 1024)

def process_uploaded_files(uploaded_files):
    """Process uploaded files - optimized for speed"""
    if not uploaded_files:
        return
    
    upload_dir = get_upload_dir()
    processed_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            # Show progress
            progress = (idx + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Quick checks
            file_hash = get_file_hash(uploaded_file.getvalue())
            if file_hash in st.session_state.file_hashes:
                logger.info(f"Skipping duplicate: {uploaded_file.name}")
                continue
            
            file_size_mb = get_file_size_mb(uploaded_file.getvalue())
            if file_size_mb > 50:
                st.warning(f" {uploaded_file.name} is large ({file_size_mb:.1f}MB) - analysis may be slow")
            
            # Save file
            file_path = upload_dir / uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            file_type = uploaded_file.type
            
            # Process by type
            if file_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            'application/vnd.ms-excel'] or uploaded_file.name.endswith(('.xlsx', '.xls')):
                # Validate Excel
                try:
                    pd.read_excel(file_path, sheet_name=0, nrows=1)
                    st.session_state.uploaded_files.append({
                        'name': uploaded_file.name,
                        'path': str(file_path),
                        'type': file_type,
                        'hash': file_hash,
                        'size_mb': file_size_mb
                    })
                    st.session_state.file_hashes.add(file_hash)
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Excel validation failed: {e}")
                    st.error(f" Invalid Excel file: {uploaded_file.name}")
                    continue
            
            elif file_type == 'application/pdf' or uploaded_file.name.endswith('.pdf'):
                # Add PDF without processing (analyze on demand)
                st.session_state.uploaded_files.append({
                    'name': uploaded_file.name,
                    'path': str(file_path),
                    'type': file_type,
                    'hash': file_hash,
                    'size_mb': file_size_mb
                })
                st.session_state.file_hashes.add(file_hash)
                processed_count += 1
            
            elif file_type == 'text/plain' or uploaded_file.name.endswith('.txt'):
                # Add TXT without processing
                st.session_state.uploaded_files.append({
                    'name': uploaded_file.name,
                    'path': str(file_path),
                    'type': file_type,
                    'hash': file_hash,
                    'size_mb': file_size_mb
                })
                st.session_state.file_hashes.add(file_hash)
                processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()
    
    # Show results
    if processed_count > 0:
        st.success(f" {processed_count} file(s) ready to analyze!")

# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main application - optimized"""
    
    # Load CSS
    load_css()
    
    # Initialize
    init_session()
    
    # ============================================================
    # HEADER
    # ============================================================
    
    st.markdown("""
    <div class="premium-header">
        <h1>DocuMind</h1>
        <p>AI-powered document analysis</p>
        <div class="badge">
            <span class="status-online">Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # SIDEBAR - SIMPLIFIED
    # ============================================================
    
    with st.sidebar:
        st.markdown("### Upload Documents")
        
        # File uploader with better UX
        uploaded_files = st.file_uploader(
            "Drop files or click to upload (PDF, Excel, TXT)",
            type=['pdf', 'txt', 'xlsx', 'xls', 'csv'],
            accept_multiple_files=True,
            key="file_uploader",
            label_visibility="collapsed",
            help="Supports: PDF, Excel (.xlsx/.xls), Text files"
        )
        
        if uploaded_files:
            process_uploaded_files(uploaded_files)
        
        # Active files
        if st.session_state.uploaded_files:
            st.markdown("#### Your Files")
            for file in st.session_state.uploaded_files:
                file_name = file['name']
                file_size = file.get('size_mb', 0)
                
                # Determine icon
                if file_name.endswith(('.xlsx', '.xls', '.csv')):
                    icon = ""
                    file_type_label = "Data"
                elif file_name.endswith('.pdf'):
                    icon = ""
                    file_type_label = "PDF"
                else:
                    icon = ""
                    file_type_label = "Text"
                
                st.markdown(f"""
                <div class="file-tag success">
                    {icon} {file_name} ({file_size:.1f}MB)
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Upload documents to get started")
        
        st.divider()
        
        # Quick actions
        st.markdown("#### Quick Actions")
        if st.button("Clear Chat", use_container_width=True, help="Clear conversation history"):
            st.session_state.messages = []
            st.rerun()
    
    # ============================================================
    # MAIN CHAT AREA
    # ============================================================
    
    if st.session_state.messages:
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            timestamp = message.get("timestamp", datetime.now())
            
            time_str = timestamp.strftime("%I:%M %p") if isinstance(timestamp, datetime) else ""
            
            if role == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="message-content">{content}</div>
                    <div class="message-time">{time_str}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="message-content">{content}</div>
                    <div class="message-time">{time_str}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon"></div>
            <h3>Ready to Analyze</h3>
            <p>Upload files and ask questions to get started.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================================
    # CHAT INPUT - FAST
    # ============================================================
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            prompt = st.text_input(
                "Your question...",
                key="chat_input",
                placeholder="Ask anything about your files...",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button(
                "Ask",
                use_container_width=True,
                type="primary"
            )
        
        if submit_button and prompt:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now()
            })
            
            # Check if files uploaded
            if not st.session_state.uploaded_files:
                # Better response for casual greetings or no files
                casual_greetings = ["hi", "hello", "hey", "what's up", "how are you", "sup"]
                is_casual = any(greeting in prompt.lower() for greeting in casual_greetings)
                
                if is_casual:
                    response = "Hi there! 👋 I'm DocuMind, your AI document analyst. To get started, please upload a file (PDF, Excel, or Text) and then ask me anything about it!"
                else:
                    response = "I'd love to help! Please upload a document first, then ask your question and I'll analyze it for you. 📄"
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now()
                })
                st.rerun()
            
            # Process with files
            with st.spinner("Analyzing..."):
                try:
                    logger.info(f"Query: {prompt[:50]}")
                    
                    # Call agent
                    response = st.session_state.agent.process(
                        prompt,
                        st.session_state.uploaded_files
                    )
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now()
                    })
                    
                except Exception as e:
                    logger.error(f"Error: {str(e)}")
                    error_msg = f"❌ {str(e)[:200]}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now()
                    })
            
            st.rerun()

if __name__ == "__main__":
    main()