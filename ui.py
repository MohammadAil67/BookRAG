import streamlit as st
import time
import pandas as pd
from datetime import datetime, timedelta
from rag_system import RAGSystem  # Changed from rag import
from config import Config  # Changed from rag import
import sys
from io import StringIO
import os

# ==========================================
# 1. SETUP & STATE MANAGEMENT
# ==========================================

# Set page config
st.set_page_config(page_title="AI Tutor", page_icon="📚", layout="wide")

# Initialize RAG System
@st.cache_resource
def initialize_rag(pdf_path=None):
    """Initialize RAG system with optional PDF path"""
    try:
        # Pass PDF path to config
        config = Config(pdf_path=pdf_path) 
        rag = RAGSystem(config)
        return rag, None
    except Exception as e:
        import traceback
        return None, f"{str(e)}\n{traceback.format_exc()}"

# Initialize Session State
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            "id": "1",
            "text": "Hello! I'm your AI tutor. How can I help you learn today? You can select a subject PDF or upload your own document to get started.",
            "sender": "ai",
            "timestamp": datetime.now()
        }
    ]

if 'uploaded_pdfs' not in st.session_state:
    st.session_state.uploaded_pdfs = []

if 'selected_pdf' not in st.session_state:
    st.session_state.selected_pdf = None

if 'view' not in st.session_state:
    st.session_state.view = 'Chat'

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

if 'rag_error' not in st.session_state:
    st.session_state.rag_error = None

# Mock Data Constants
PREDEFINED_PDFS = [
    {"id": "1", "name": "Bangla Shahitto", "type": "predefined", "path": "bangla_shahitto.pdf"},
    {"id": "2", "name": "Physics - Class 10", "type": "predefined", "path": "Physics  9-10 EV book full pdf_compressed.pdf"},
    {"id": "6", "name": "English Literature", "type": "predefined", "path": "Dakhil - 2018 - Class-(9-10) English For Today PDF Web.pdf"},
]

# ==========================================
# 2. HELPER COMPONENTS (MODALS -> SIDEBAR)
# ==========================================

def render_sidebar():
    with st.sidebar:
        st.title("📚 AI Tutor")
        
        # Navigation
        st.markdown("### Navigation")
        selected_view = st.radio(
            "Go to",
            ['Chat', 'Practice', 'Study Plan', 'Progress Tracker', 'System Logs'],
            label_visibility="collapsed"
        )
        
        st.divider()

        # RAG System Status
        with st.expander("🤖 AI System Status", expanded=True):
            if st.session_state.rag_system:
                st.success("✅ RAG System Active")
                if st.session_state.selected_pdf:
                    st.info(f"📄 **Active PDF:**\n{st.session_state.selected_pdf['name']}")
                    st.metric("Chunks Loaded", len(st.session_state.rag_system.chunks))
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Reload System"):
                        st.cache_resource.clear()
                        st.session_state.rag_system = None
                        st.rerun()
                
                with col2:
                    if st.button("🗑️ Clear History"):
                        st.session_state.rag_system.clear_history()
                        st.session_state.messages = [] # Clear UI chat too
                        st.success("History cleared!")
                    
            else:
                st.warning("⚠️ RAG System Not Loaded")
                if st.session_state.rag_error:
                    st.error(f"Error: {st.session_state.rag_error}")

        # Student Info (Converted from Modal)
        with st.expander("👤 Student Profile"):
            st.markdown("""
            <div style="text-align: center;">
                <div style="background-color: #3b82f6; color: white; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 24px; margin: 0 auto;">AS</div>
                <h3>Alex Smith</h3>
                <p style="color: gray;">Grade 10 Student</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**📧 Email:** alex.smith@school.edu")
            st.write("**🏆 Score:** 1,247 points")

        return selected_view

# ==========================================
# 3. VIEW: CHAT INTERFACE (INTEGRATED WITH RAG)
# ==========================================

def search_for_pdf(filename: str) -> str:
    """Search for a PDF file in common locations."""
    from pathlib import Path
    
    if not filename.lower().endswith('.pdf'):
        filename = f"{filename}.pdf"
    
    search_locations = [
        Path.cwd(),
        Path.home() / "Desktop",
        Path.home() / "Documents",
        Path.home() / "Downloads",
    ]
    
    for location in search_locations:
        if location.exists():
            potential_path = location / filename
            if potential_path.exists() and potential_path.is_file():
                return str(potential_path)
    return None

def render_chat_view():
    st.header("💬 AI Tutor Chat")

    # PDF Selection Area
    st.subheader("📚 Select a PDF Document")
    
    selection_method = st.radio(
        "How would you like to select a PDF?",
        ["Predefined PDFs", "Browse Files", "Enter Filename"],
        horizontal=True
    )
    
    st.divider()
    
    selected_pdf_path = None
    pdf_name = None
    
    # Method 1: Predefined PDFs
    if selection_method == "Predefined PDFs":
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.session_state.selected_pdf:
                st.info(f"📄 **Active PDF:** {st.session_state.selected_pdf['name']}")
        with col2:
            pdf_names = [p['name'] for p in PREDEFINED_PDFS]
            selected_name = st.selectbox("Choose PDF", ["None"] + pdf_names, label_visibility="collapsed")
            if selected_name != "None":
                found_pdf = next((p for p in PREDEFINED_PDFS if p['name'] == selected_name), None)
                if found_pdf:
                    selected_pdf_path = found_pdf.get('path')
                    pdf_name = found_pdf['name']
    
    # Method 2: File Browser
    elif selection_method == "Browse Files":
        st.info("💡 **Tip:** Upload or drag & drop your PDF file")
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'], key="pdf_uploader")
        if uploaded_file:
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_pdf_path = temp_path
            pdf_name = uploaded_file.name
            st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    # Method 3: Manual Filename
    elif selection_method == "Enter Filename":
        col1, col2 = st.columns([3, 1])
        with col1:
            filename = st.text_input("Enter PDF filename", placeholder="physics.pdf", key="manual_filename")
        with col2:
            search_button = st.button("🔍 Search", type="primary")
        
        if search_button and filename:
            found_path = search_for_pdf(filename)
            if found_path:
                selected_pdf_path = found_path
                pdf_name = os.path.basename(found_path)
                st.success(f"✅ Found: {found_path}")
            else:
                st.error(f"❌ Could not find '{filename}'")
    
    # Load RAG system if a new PDF is selected
    if selected_pdf_path:
        # Check if we need to reload (different path or not loaded yet)
        current_path = st.session_state.selected_pdf.get('path') if st.session_state.selected_pdf else None
        
        if current_path != selected_pdf_path:
            with st.spinner(f"📄 Loading {pdf_name}..."):
                rag, error = initialize_rag(selected_pdf_path)
                st.session_state.rag_system = rag
                st.session_state.rag_error = error
                
                if rag:
                    st.session_state.selected_pdf = {
                        "name": pdf_name,
                        "path": selected_pdf_path,
                        "type": selection_method
                    }
                    st.session_state.messages = [] # Clear chat on new PDF
                    st.session_state.messages.append({
                        "id": str(time.time()),
                        "text": f"I've loaded '{pdf_name}'. What would you like to learn?",
                        "sender": "ai",
                        "timestamp": datetime.now()
                    })
                    st.rerun()

    # Chat History
    for msg in st.session_state.messages:
        avatar = "🤖" if msg['sender'] == 'ai' else "👤"
        with st.chat_message(msg['sender'], avatar=avatar):
            st.write(msg['text'])

    # Chat Input
    if prompt := st.chat_input("Ask me anything..."):
        if not st.session_state.rag_system:
            st.error("⚠️ Please select a PDF first!")
            return
        
        # User Message
        st.session_state.messages.append({
            "id": str(time.time()),
            "text": prompt,
            "sender": "user",
            "timestamp": datetime.now()
        })
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)

        # AI Response
        with st.chat_message("ai", avatar="🤖"):
            with st.spinner("🧠 Thinking..."):
                try:
                    # Capture console output for debugging logs
                    old_stdout = sys.stdout
                    sys.stdout = captured_output = StringIO()
                    
                    response_text = st.session_state.rag_system.ask(prompt)
                    
                    # Get the captured debug output
                    debug_output = captured_output.getvalue()
                    sys.stdout = old_stdout
                    
                    st.write(response_text)
                    
                except Exception as e:
                    sys.stdout = old_stdout
                    import traceback
                    response_text = f"❌ Error: {str(e)}"
                    st.error(response_text)
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                
        st.session_state.messages.append({
            "id": str(time.time() + 1),
            "text": response_text,
            "sender": "ai",
            "timestamp": datetime.now()
        })

# ==========================================
# 4. VIEW: PRACTICE EXERCISES
# ==========================================

def render_practice_view():
    st.header("📝 Practice Exercises")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ Please load a PDF first to generate practice exercises.")
        return
    
    st.info("Practice exercises will be generated based on the loaded PDF content in future updates.")
    
    # You can add exercise generation here in the future
    # For now, showing topic info
    topic_info = st.session_state.rag_system.get_topic_status()
    if topic_info.get('current_topic'):
        st.write(f"**Current Topic:** {topic_info['current_topic']}")
        if topic_info.get('current_keywords'):
            st.write(f"**Keywords:** {', '.join(topic_info['current_keywords'][:5])}")

# ==========================================
# 5. VIEW: SYSTEM LOGS
# ==========================================

def render_logs_view():
    st.header("🔧 System Logs & Debugging")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ No RAG system loaded.")
        return
    
    rag = st.session_state.rag_system
    
    # System Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Chunks", len(rag.chunks))
    col2.metric("Cached Answers", len(rag.cache.cache))
    col3.metric("History Length", len(rag.chat_history))
    
    # Topic tracking stats
    topic_info = rag.get_topic_status()
    col4.metric("Topics Discussed", len(topic_info.get('all_topics', [])))
    
    st.divider()
    
    # Topic Tracking Info
    with st.expander("🎯 Topic Tracking", expanded=True):
        if topic_info.get('current_topic'):
            st.success(f"**Current Topic:** {topic_info['current_topic']}")
            st.write(f"**Confidence:** {topic_info.get('confidence', 0):.2f}")
            if topic_info.get('current_keywords'):
                st.write(f"**Keywords:** {', '.join(topic_info['current_keywords'])}")
        else:
            st.info("No active topic yet")
        
        if topic_info.get('all_topics'):
            st.write(f"**All Topics:** {', '.join(topic_info['all_topics'])}")
    
    # Configuration
    with st.expander("⚙️ System Configuration"):
        st.json({
            "INITIAL_RETRIEVAL_K": rag.config.INITIAL_RETRIEVAL_K,
            "FINAL_TOP_K": rag.config.FINAL_TOP_K,
            "RERANK_THRESHOLD": rag.config.RERANK_THRESHOLD,
            "PDF_PATH": rag.config.PDF_PATH,
            "MODEL_CACHE_DIR": rag.config.MODEL_CACHE_DIR
        })
    
    # Cached Answers
    with st.expander("💾 Answer Cache Content"):
        if rag.cache.cache:
            st.write(f"Total cached answers: {len(rag.cache.cache)}")
            for i, (key, val) in enumerate(list(rag.cache.cache.items())[:10]):
                st.text(f"Key: {key[:16]}...")
                st.code(val[:200] + "..." if len(val) > 200 else val)
                if i < 9:
                    st.divider()
        else:
            st.info("Cache is empty")

    # Conversation History
    with st.expander("💬 Conversation History"):
        if rag.history.history:
            for i, item in enumerate(rag.history.history):
                st.write(f"**Turn {i+1}**")
                st.write(f"**Q:** {item['question']}")
                st.write(f"**A:** {item['answer'][:300]}..." if len(item['answer']) > 300 else f"**A:** {item['answer']}")
                st.divider()
        else:
            st.info("No conversation history yet")
    
    # Retrieval Stats
    with st.expander("📊 Retrieval System Stats"):
        st.write("**Retriever Stack:**")
        st.write("1. MultiQueryRetriever (generates query variants)")
        st.write("2. TopicAwareRetriever (maintains conversation context)")
        st.write("3. HybridRetriever (BM25 + Vector Search + Cross-Encoder Reranking)")
        
        if hasattr(rag.retriever, 'base_retriever'):
            topic_retriever = rag.retriever.base_retriever
            if hasattr(topic_retriever, 'get_topic_summary'):
                st.write(f"\n**Topic Summary:** {topic_retriever.get_topic_summary()}")

# ==========================================
# 6. MAIN APP ROUTER
# ==========================================

def main():
    selected_view = render_sidebar()

    if selected_view == 'Chat':
        render_chat_view()
    elif selected_view == 'Practice':
        render_practice_view()
    elif selected_view == 'System Logs':
        render_logs_view()
    elif selected_view == 'Study Plan':
        st.header("📅 Study Plan")
        st.info("Study plan features coming soon!")
    elif selected_view == 'Progress Tracker':
        st.header("📈 Progress Tracker")
        st.info("Progress tracking features coming soon!")

if __name__ == "__main__":
    main()