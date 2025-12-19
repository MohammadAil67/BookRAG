import streamlit as st
import time
import pandas as pd
from datetime import datetime, timedelta
from bunga import RAGSystem, Config
import sys
from io import StringIO

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
        config = Config(pdf_path=pdf_path) if pdf_path else Config()
        rag = RAGSystem(config)
        return rag, None
    except Exception as e:
        return None, str(e)

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
    {"id": "3", "name": "Mathematics - Calculus", "type": "predefined", "path": "mathematics_calculus.pdf"},
    {"id": "4", "name": "Chemistry - Organic", "type": "predefined", "path": "chemistry_organic.pdf"},
    {"id": "5", "name": "Biology - Cell Biology", "type": "predefined", "path": "biology_cell.pdf"},
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
                    st.metric("Cached Answers", len(st.session_state.rag_system.cache.cache))
                
                if st.button("🔄 Reload System"):
                    st.cache_resource.clear()
                    st.session_state.rag_system = None
                    st.rerun()
                
                if st.button("🗑️ Clear History"):
                    st.session_state.rag_system.clear_history()
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
            st.write("**📅 Member Since:** Sept 2024")
            st.write("**📖 Active Courses:** 5")
            st.write("**🏆 Score:** 1,247 points")

        # Settings (Converted from Modal)
        with st.expander("⚙️ Settings"):
            st.write("**Theme:** Handled by Streamlit Settings (⋮ > Settings)")
            st.checkbox("🔔 Study reminders", value=True)
            st.selectbox("🌐 Language", ["English", "Spanish", "French", "German"])

        return selected_view

# ==========================================
# 3. VIEW: CHAT INTERFACE (INTEGRATED WITH RAG)
# ==========================================

def render_chat_view():
    st.header("💬 AI Tutor Chat")

    # PDF Selection Area
    st.subheader("📚 Select a PDF Document")
    
    # Three methods: Predefined, File Browser, or Manual Path
    selection_method = st.radio(
        "How would you like to select a PDF?",
        ["Predefined PDFs", "Browse Files", "Enter Filename"],
        horizontal=True
    )
    
    st.divider()
    
    selected_pdf_path = None
    
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
    
    # Method 2: File Browser (using file uploader as proxy)
    elif selection_method == "Browse Files":
        st.info("💡 **Tip:** Upload or drag & drop your PDF file")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            key="pdf_uploader"
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            import tempfile
            import os
            
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            selected_pdf_path = temp_path
            pdf_name = uploaded_file.name
            st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    # Method 3: Manual Filename Entry
    elif selection_method == "Enter Filename":
        st.info("💡 **Tip:** Enter the filename to search in current directory and common locations")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            filename = st.text_input(
                "Enter PDF filename (e.g., 'physics.pdf' or just 'physics')",
                placeholder="physics.pdf",
                key="manual_filename"
            )
        
        with col2:
            search_button = st.button("🔍 Search", type="primary")
        
        if search_button and filename:
            with st.spinner(f"🔍 Searching for '{filename}'..."):
                found_path = search_for_pdf(filename)
                
                if found_path:
                    selected_pdf_path = found_path
                    pdf_name = os.path.basename(found_path)
                    st.success(f"✅ Found: {found_path}")
                else:
                    st.error(f"❌ Could not find '{filename}' in search locations")
                    st.info("Search locations: Current directory, Desktop, Documents, Downloads")
    
    # Show remove button if PDF is active
    if st.session_state.selected_pdf and st.button("❌ Remove Current PDF"):
        st.session_state.selected_pdf = None
        st.session_state.rag_system = None
        st.cache_resource.clear()
        st.session_state.messages.append({
            "id": str(time.time()),
            "text": "PDF removed. Feel free to select another document!",
            "sender": "ai",
            "timestamp": datetime.now()
        })
        st.rerun()
    
    # Load RAG system if a new PDF is selected
    if selected_pdf_path and (
        st.session_state.selected_pdf is None or 
        st.session_state.selected_pdf.get('path') != selected_pdf_path
    ):
        with st.spinner(f"🔄 Loading {pdf_name}..."):
            rag, error = initialize_rag(selected_pdf_path)
            st.session_state.rag_system = rag
            st.session_state.rag_error = error
            
            if rag:
                st.session_state.selected_pdf = {
                    "name": pdf_name,
                    "path": selected_pdf_path,
                    "type": selection_method
                }
                st.session_state.messages.append({
                    "id": str(time.time()),
                    "text": f"Perfect! I've loaded '{pdf_name}'. What would you like to learn?",
                    "sender": "ai",
                    "timestamp": datetime.now()
                })
                st.success(f"✅ Loaded: {pdf_name}")
                st.rerun()
            else:
                st.error(f"❌ Failed to load: {error}")

    st.divider()

    # Chat History
    for msg in st.session_state.messages:
        avatar = "🤖" if msg['sender'] == 'ai' else "👤"
        with st.chat_message(msg['sender'], avatar=avatar):
            st.write(msg['text'])

    # Chat Input
    if prompt := st.chat_input("Ask me anything..."):
        # Check if RAG system is loaded
        if not st.session_state.rag_system:
            st.error("⚠️ Please select a PDF first to enable AI chat!")
            return
        
        # User Message
        st.session_state.messages.append({
            "id": str(time.time()),
            "text": prompt,
            "sender": "user",
            "timestamp": datetime.now()
        })
        
        # Force re-render to show user message immediately
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)

        # AI Response using RAG
        with st.chat_message("ai", avatar="🤖"):
            with st.spinner("🧠 Thinking..."):
                try:
                    # Capture console output
                    old_stdout = sys.stdout
                    sys.stdout = captured_output = StringIO()
                    
                    # Get answer from RAG system
                    response_text = st.session_state.rag_system.ask(prompt)
                    
                    # Restore stdout
                    sys.stdout = old_stdout
                    
                    st.write(response_text)
                    
                except Exception as e:
                    sys.stdout = old_stdout
                    response_text = f"❌ Error: {str(e)}"
                    st.error(response_text)
                
        st.session_state.messages.append({
            "id": str(time.time() + 1),
            "text": response_text,
            "sender": "ai",
            "timestamp": datetime.now()
        })


# ==========================================
# HELPER FUNCTION: PDF SEARCH
# ==========================================

def search_for_pdf(filename: str) -> str:
    """
    Search for a PDF file in common locations.
    Returns the full path if found, None otherwise.
    """
    import os
    from pathlib import Path
    
    # Add .pdf extension if not present
    if not filename.lower().endswith('.pdf'):
        filename = f"{filename}.pdf"
    
    # Define search locations
    search_locations = [
        # Current directory
        Path.cwd(),
        # User's Desktop
        Path.home() / "Desktop",
        # User's Documents
        Path.home() / "Documents",
        # User's Downloads
        Path.home() / "Downloads",
        # Same directory as the script
        Path(__file__).parent if '__file__' in globals() else Path.cwd(),
    ]
    
    # Search in each location
    for location in search_locations:
        if location.exists():
            # Direct match
            potential_path = location / filename
            if potential_path.exists() and potential_path.is_file():
                return str(potential_path)
            
            # Case-insensitive search
            try:
                for file in location.iterdir():
                    if file.is_file() and file.name.lower() == filename.lower():
                        return str(file)
            except PermissionError:
                continue
    
    return None

# ==========================================
# 4. VIEW: PRACTICE EXERCISES
# ==========================================

def render_practice_view():
    st.header("📝 Practice Exercises")
    st.caption("Complete practice questions to reinforce your learning")

    practice_items = [
        {"id": '1', "subject": 'Mathematics', "topic": 'Quadratic Equations', "total": 15, "done": 12, "diff": 'Medium'},
        {"id": '2', "subject": 'Physics', "topic": 'Newton\'s Laws', "total": 20, "done": 20, "diff": 'Hard'},
        {"id": '3', "subject": 'Chemistry', "topic": 'Periodic Table', "total": 10, "done": 5, "diff": 'Easy'},
        {"id": '4', "subject": 'Biology', "topic": 'Cell Structure', "total": 12, "done": 0, "diff": 'Medium'},
    ]

    # Create grid layout
    cols = st.columns(2)
    
    for idx, item in enumerate(practice_items):
        col = cols[idx % 2]
        with col:
            with st.container(border=True):
                # Header
                st.subheader(item['topic'])
                st.markdown(f"**{item['subject']}**")
                
                # Difficulty Badge (using coloring)
                color_map = {"Easy": "green", "Medium": "orange", "Hard": "red"}
                st.markdown(f":{color_map[item['diff']]}[{item['diff']}]")

                # Progress
                progress_pct = item['done'] / item['total']
                st.progress(progress_pct)
                st.write(f"Progress: {item['done']}/{item['total']}")

                # Button
                btn_label = "Review" if item['done'] == item['total'] else "Continue Practice"
                if st.button(btn_label, key=item['id']):
                    st.toast(f"Starting {item['topic']}...")

# ==========================================
# 5. VIEW: STUDY PLAN
# ==========================================

def render_study_plan_view():
    st.header("📅 Study Plan")
    st.caption("Your personalized study schedule")

    study_tasks = [
        {"title": 'Review Calculus', "subject": 'Math', "min": 45, "date": datetime.now().strftime('%Y-%m-%d'), "done": False},
        {"title": 'Physics Lab Report', "subject": 'Physics', "min": 60, "date": datetime.now().strftime('%Y-%m-%d'), "done": True},
        {"title": 'Chemical Bonding', "subject": 'Chemistry', "min": 30, "date": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'), "done": False},
        {"title": 'Biology Chapter 5', "subject": 'Biology', "min": 40, "date": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'), "done": False},
    ]

    # Grouping Logic
    today = datetime.now().strftime('%Y-%m-%d')
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    grouped = {}
    for task in study_tasks:
        if task['date'] not in grouped:
            grouped[task['date']] = []
        grouped[task['date']].append(task)

    for date, tasks in grouped.items():
        label = "Today" if date == today else "Tomorrow" if date == tomorrow else date
        st.subheader(f"📌 {label}")
        
        for i, task in enumerate(tasks):
            # Visual checkmark
            icon = "✅" if task['done'] else "⭕"
            strike = "text-decoration: line-through;" if task['done'] else ""
            
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 4, 2])
                with c1:
                    st.write(f"## {icon}")
                with c2:
                    st.markdown(f"<h4 style='{strike} margin:0'>{task['title']}</h4>", unsafe_allow_html=True)
                    st.caption(f"{task['subject']} • ⏱️ {task['min']} min")
                with c3:
                    if not task['done']:
                        if st.button("Start", key=f"start_{date}_{i}"):
                            st.toast(f"Starting timer for {task['title']}")
                    else:
                        st.button("Review", key=f"rev_{date}_{i}", disabled=True)

# ==========================================
# 6. VIEW: PROGRESS TRACKER
# ==========================================

def render_progress_view():
    st.header("📈 Progress Tracker")
    
    # Top Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Hours", "89h", "+2.5h")
    c2.metric("Completed Tasks", "47", "+4")
    c3.metric("Achievement Points", "1,247", "+150")
    c4.metric("Streak", "12 Days", "🔥")

    st.divider()

    col_charts_1, col_charts_2 = st.columns(2)

    with col_charts_1:
        st.subheader("Subject Proficiency")
        subjects = [
            {"name": "Mathematics", "prog": 0.78},
            {"name": "Physics", "prog": 0.85},
            {"name": "Chemistry", "prog": 0.62},
            {"name": "Biology", "prog": 0.70},
            {"name": "History", "prog": 0.55},
        ]
        for sub in subjects:
            st.text(f"{sub['name']} ({int(sub['prog']*100)}%)")
            st.progress(sub['prog'])

    with col_charts_2:
        st.subheader("Weekly Activity (Hours)")
        # Simple bar chart data
        data = pd.DataFrame({
            "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "Hours": [2.5, 3.0, 1.5, 4.0, 2.0, 3.5, 2.5]
        })
        st.bar_chart(data, x="Day", y="Hours", color="#3b82f6")

    st.subheader("🏆 Recent Achievements")
    ac1, ac2, ac3 = st.columns(3)
    ac1.info("**Week Warrior**\n\n7-day streak completed")
    ac2.success("**Math Master**\n\n100 problems solved")
    ac3.warning("**Quick Learner**\n\nCompleted 5 topics")

# ==========================================
# 7. VIEW: SYSTEM LOGS (NEW - from bunga.py main())
# ==========================================

def render_logs_view():
    st.header("🔧 System Logs & Debugging")
    st.caption("RAG System internals and console output")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ No RAG system loaded. Select a PDF in Chat view first.")
        return
    
    rag = st.session_state.rag_system
    
    # System Stats
    st.subheader("📊 System Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", len(rag.chunks))
    col2.metric("Cached Answers", len(rag.cache.cache))
    col3.metric("Conversation History", len(rag.history.history))
    
    st.divider()
    
    # Configuration
    with st.expander("⚙️ Configuration", expanded=True):
        st.json({
            "PDF_PATH": rag.config.PDF_PATH,
            "TOP_K_CHUNKS": rag.config.TOP_K_CHUNKS,
            "SIMILARITY_THRESHOLD": rag.config.SIMILARITY_THRESHOLD,
            "MAX_CONVERSATION_HISTORY": rag.config.MAX_CONVERSATION_HISTORY,
            "CHUNKS_FILE": rag.config.CHUNKS_FILE,
            "EMBEDDINGS_FILE": rag.config.EMBEDDINGS_FILE,
            "CACHE_FILE": rag.config.CONTEXT_CACHE_FILE
        })
    
    # Recent Conversation
    with st.expander("💬 Conversation History"):
        if rag.history.history:
            for i, exchange in enumerate(rag.history.history):
                st.markdown(f"**Exchange {i+1}** ({exchange['timestamp']})")
                st.info(f"**Q:** {exchange['question']}")
                st.success(f"**A:** {exchange['answer'][:200]}...")
                st.divider()
        else:
            st.info("No conversation history yet")
    
    # Cached Answers
    with st.expander("💾 Answer Cache"):
        if rag.cache.cache:
            st.write(f"Total cached entries: {len(rag.cache.cache)}")
            # Show sample
            sample_keys = list(rag.cache.cache.keys())[:5]
            for key in sample_keys:
                st.code(f"{key}: {rag.cache.cache[key][:100]}...")
        else:
            st.info("No cached answers yet")
    
    # Entity Tracking
    with st.expander("🏷️ Tracked Entities"):
        if rag.history.last_entities:
            st.json(rag.history.last_entities)
        else:
            st.info("No entities tracked yet")
    
    # Test Query
    st.subheader("🧪 Test Query (Debug Mode)")
    test_query = st.text_input("Enter a test question:")
    if st.button("Run Test Query"):
        if test_query:
            with st.spinner("Processing..."):
                # Capture output
                old_stdout = sys.stdout
                sys.stdout = captured = StringIO()
                
                try:
                    answer = rag.ask(test_query)
                    console_output = captured.getvalue()
                    
                    sys.stdout = old_stdout
                    
                    st.success("✅ Query completed")
                    st.markdown("**Answer:**")
                    st.write(answer)
                    
                    st.markdown("**Console Output:**")
                    st.code(console_output)
                    
                except Exception as e:
                    sys.stdout = old_stdout
                    st.error(f"❌ Error: {str(e)}")

# ==========================================
# 8. MAIN APP ROUTER
# ==========================================

def main():
    selected_view = render_sidebar()

    if selected_view == 'Chat':
        render_chat_view()
    elif selected_view == 'Practice':
        render_practice_view()
    elif selected_view == 'Study Plan':
        render_study_plan_view()
    elif selected_view == 'Progress Tracker':
        render_progress_view()
    elif selected_view == 'System Logs':
        render_logs_view()

if __name__ == "__main__":
    main()