import streamlit as st
import time
import pandas as pd
import json  # ← Add this if not already there
import re 
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
    st.header("📝 Practice & Tests")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ Please load a PDF first to generate practice questions.")
        return
    
    # Topic Selection
    st.subheader("📚 Choose Topic")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # Get current topic from tracker
        topic_info = st.session_state.rag_system.get_topic_status()
        current_topic = topic_info.get('current_topic', '')
        
        topic = st.text_input(
            "Enter topic or leave blank for current conversation topic",
            value=current_topic,
            placeholder="e.g., Zahir Raihan, Newton's Laws, etc."
        )
    
    with col2:
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
        num_questions = st.selectbox("Questions", [5, 10, 15, 20])
    
    # Generate Quiz Button
    if st.button("🎲 Generate Quiz", type="primary"):
        with st.spinner("🧠 Generating questions..."):
            # Use RAG system to generate questions
            prompt = f"""Based on the content about {topic if topic else 'the current topic'}, generate {num_questions} {difficulty.lower()} multiple-choice questions.

Format each question as:
Q1. [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [A/B/C/D]
Explanation: [Brief explanation]

Make questions specific to the PDF content, not general knowledge."""

            quiz_text = st.session_state.rag_system.ask(prompt)
            
            # Store quiz in session
            st.session_state.current_quiz = {
                'topic': topic or current_topic,
                'difficulty': difficulty,
                'questions': quiz_text,
                'timestamp': datetime.now(),
                'answers': {}
            }
            
            st.rerun()
    
    # Display Quiz
    if 'current_quiz' in st.session_state and st.session_state.current_quiz:
        quiz = st.session_state.current_quiz
        
        st.divider()
        st.subheader(f"📋 Quiz: {quiz['topic']}")
        st.caption(f"Difficulty: {quiz['difficulty']} | Generated: {quiz['timestamp'].strftime('%I:%M %p')}")
        
        # Parse and display questions
        questions_text = quiz['questions']
        
        # Simple display (you can parse better later)
        st.markdown(questions_text)
        
        st.divider()
        
        # Answer submission area
        st.subheader("✍️ Your Answers")
        st.info("💡 Read questions above, then submit answers here for grading")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            user_answers = st.text_area(
                "Enter your answers (e.g., 1A, 2C, 3B, 4D, 5A)",
                placeholder="1A, 2C, 3B...",
                height=100
            )
        
        with col2:
            if st.button("📊 Grade Quiz", type="primary"):
                with st.spinner("Grading..."):
                    # Use RAG to grade (it knows the correct answers from generation)
                    grade_prompt = f"""Here are the quiz questions and correct answers:
{questions_text}

The student answered: {user_answers}

Grade the quiz and provide:
1. Score (X/Y format)
2. Which questions were correct/incorrect
3. Brief explanation for incorrect answers
4. Encouragement based on performance"""

                    grading_result = st.session_state.rag_system.ask(grade_prompt)
                    
                    st.session_state.quiz_result = {
                        'answers': user_answers,
                        'grading': grading_result,
                        'timestamp': datetime.now()
                    }
                    
                    # Save to progress
                    _save_quiz_result(quiz, user_answers, grading_result)
                    
                    st.rerun()
        
        # Show grading result
        if 'quiz_result' in st.session_state and st.session_state.quiz_result:
            st.divider()
            st.subheader("📊 Results")
            st.markdown(st.session_state.quiz_result['grading'])
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 New Quiz"):
                    del st.session_state.current_quiz
                    del st.session_state.quiz_result
                    st.rerun()
            with col2:
                if st.button("📈 View Progress"):
                    st.session_state.view = 'Progress Tracker'
                    st.rerun()

# ==========================================
# 2. STUDY PLAN VIEW (2-3 hours)
# ==========================================

def render_study_plan_view():
    st.header("📅 Study Plan Generator")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ Please load a PDF first to generate a study plan.")
        return
    
    # Two modes: Auto-detect or Manual
    st.subheader("🎯 Study Plan Type")
    plan_mode = st.radio(
        "How should we create your plan?",
        ["🤖 Auto-Detect Weak Areas", "✍️ Manual Topic Selection"],
        horizontal=True
    )
    
    if plan_mode == "🤖 Auto-Detect Weak Areas":
        st.info("📊 We'll analyze your conversation history and quiz results to identify weak areas")
        
        if st.button("🔍 Analyze & Generate Plan", type="primary"):
            with st.spinner("🧠 Analyzing your learning patterns..."):
                # Get conversation history
                history = st.session_state.rag_system.chat_history
                
                # Get quiz results if any
                quiz_results = _get_quiz_history()
                
                # Use RAG to analyze and create plan
                analysis_prompt = f"""Based on this student's learning activity:

Conversation History (last 10 interactions):
{json.dumps(history[-10:], indent=2)}

Quiz Results:
{json.dumps(quiz_results, indent=2)}

1. Identify 3-5 topics the student is weak at or hasn't covered
2. Create a 7-day study plan with:
   - Daily topics to cover
   - Recommended time (minutes)
   - Specific sections from the PDF to read
   - Practice exercises
3. Prioritize weak areas first
4. Make it realistic and achievable

Format as a clear day-by-day plan."""

                study_plan = st.session_state.rag_system.ask(analysis_prompt)
                
                st.session_state.current_study_plan = {
                    'type': 'auto',
                    'plan': study_plan,
                    'created': datetime.now()
                }
                
                st.rerun()
    
    else:  # Manual mode
        st.subheader("✍️ Customize Your Plan")
        
        col1, col2 = st.columns(2)
        with col1:
            topics = st.text_area(
                "Topics to cover (one per line)",
                placeholder="Zahir Raihan\nLiberation War\nLanguage Movement",
                height=150
            )
        
        with col2:
            duration = st.selectbox("Study Duration", ["3 days", "7 days", "14 days", "30 days"])
            daily_time = st.slider("Daily study time (minutes)", 15, 120, 45, 15)
            focus_areas = st.multiselect(
                "Focus on",
                ["Understanding Concepts", "Memorization", "Problem Solving", "Exam Prep"]
            )
        
        if st.button("📝 Generate Custom Plan", type="primary"):
            with st.spinner("Creating your personalized study plan..."):
                plan_prompt = f"""Create a {duration} study plan for these topics:
{topics}

Requirements:
- Daily study time: {daily_time} minutes
- Focus areas: {', '.join(focus_areas)}
- Use content from the loaded PDF
- Include specific reading sections
- Add practice activities
- Make it achievable and motivating

Format as a clear day-by-day schedule with checkboxes."""

                study_plan = st.session_state.rag_system.ask(plan_prompt)
                
                st.session_state.current_study_plan = {
                    'type': 'manual',
                    'topics': topics,
                    'duration': duration,
                    'plan': study_plan,
                    'created': datetime.now()
                }
                
                st.rerun()
    
    # Display Study Plan
    if 'current_study_plan' in st.session_state and st.session_state.current_study_plan:
        plan = st.session_state.current_study_plan
        
        st.divider()
        st.subheader("📋 Your Study Plan")
        st.caption(f"Created: {plan['created'].strftime('%B %d, %Y at %I:%M %p')}")
        
        # Display the plan
        st.markdown(plan['plan'])
        
        st.divider()
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💾 Save Plan"):
                _save_study_plan(plan)
                st.success("✅ Plan saved to your progress!")
        
        with col2:
            if st.button("📧 Export Plan"):
                # Create downloadable text file
                plan_text = f"Study Plan\n{'='*50}\n{plan['plan']}"
                st.download_button(
                    "⬇️ Download",
                    plan_text,
                    file_name=f"study_plan_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
        
        with col3:
            if st.button("🔄 Generate New Plan"):
                del st.session_state.current_study_plan
                st.rerun()

# ==========================================
# 3. PROGRESS TRACKER VIEW (1-2 hours)
# ==========================================

def render_progress_tracker_view():
    st.header("📈 Progress Tracker")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ No data yet. Start learning to track your progress!")
        return
    
    # Initialize progress storage
    if 'learning_progress' not in st.session_state:
        st.session_state.learning_progress = {
            'conversations': [],
            'quizzes': [],
            'study_plans': [],
            'topics_covered': []
        }
    
    # Summary Stats
    st.subheader("📊 Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_questions = len(st.session_state.rag_system.chat_history)
        st.metric("Questions Asked", total_questions)
    
    with col2:
        quizzes_taken = len(st.session_state.learning_progress.get('quizzes', []))
        st.metric("Quizzes Taken", quizzes_taken)
    
    with col3:
        topics_covered = len(st.session_state.rag_system.topic_tracker.get_all_topics())
        st.metric("Topics Covered", topics_covered)
    
    with col4:
        study_plans = len(st.session_state.learning_progress.get('study_plans', []))
        st.metric("Study Plans", study_plans)
    
    st.divider()
    
    # Detailed Views
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Conversation History",
        "📝 Quiz Results",
        "🎯 Topics Covered",
        "📅 Study Plans"
    ])
    
    with tab1:
        st.subheader("Recent Conversations")
        history = st.session_state.rag_system.chat_history
        
        if history:
            for i, turn in enumerate(reversed(history[-20:])):
                with st.expander(f"Q{len(history)-i}: {turn['user'][:60]}..."):
                    st.write(f"**Question:** {turn['user']}")
                    st.write(f"**Answer:** {turn['ai'][:300]}...")
        else:
            st.info("No conversations yet. Start chatting!")
    
    with tab2:
        st.subheader("Quiz Performance")
        quizzes = st.session_state.learning_progress.get('quizzes', [])
        
        if quizzes:
            for i, quiz in enumerate(reversed(quizzes[-10:])):
                with st.expander(f"Quiz {len(quizzes)-i}: {quiz['topic']} - {quiz['timestamp'].strftime('%b %d')}"):
                    st.write(f"**Topic:** {quiz['topic']}")
                    st.write(f"**Difficulty:** {quiz['difficulty']}")
                    st.write(f"**Score:** {quiz.get('score', 'N/A')}")
                    st.write(f"**Date:** {quiz['timestamp'].strftime('%B %d, %Y at %I:%M %p')}")
        else:
            st.info("No quizzes taken yet. Try the Practice section!")
    
    with tab3:
        st.subheader("Topics You've Explored")
        topics = st.session_state.rag_system.topic_tracker.get_all_topics()
        
        if topics:
            # Get topic details
            topic_info = st.session_state.rag_system.get_topic_status()
            
            st.write(f"**Currently Studying:** {topic_info.get('current_topic', 'N/A')}")
            st.write(f"**All Topics Covered:**")
            
            for topic in topics:
                st.markdown(f"- {topic}")
            
            # Suggest next topics
            if st.button("🔮 Suggest Next Topics"):
                with st.spinner("Analyzing gaps..."):
                    suggest_prompt = f"""Based on these topics the student has covered:
{', '.join(topics)}

And the PDF content available, suggest 3-5 related topics they should explore next.
Make suggestions that build on what they know."""

                    suggestions = st.session_state.rag_system.ask(suggest_prompt)
                    st.markdown("### 💡 Suggested Next Topics")
                    st.markdown(suggestions)
        else:
            st.info("Start chatting to track topics!")
    
    with tab4:
        st.subheader("Your Study Plans")
        plans = st.session_state.learning_progress.get('study_plans', [])
        
        if plans:
            for i, plan in enumerate(reversed(plans)):
                with st.expander(f"Plan {len(plans)-i}: {plan['type'].title()} - {plan['created'].strftime('%b %d')}"):
                    st.markdown(plan['plan'][:500] + "...")
                    if st.button(f"View Full Plan {i}", key=f"view_plan_{i}"):
                        st.markdown(plan['plan'])
        else:
            st.info("No study plans yet. Create one in the Study Plan section!")
    
    st.divider()
    
    # Export Progress
    st.subheader("📤 Export Progress")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate Progress Report"):
            with st.spinner("Creating report..."):
                report_prompt = f"""Create a comprehensive progress report for this student:

Total Questions: {len(st.session_state.rag_system.chat_history)}
Quizzes Taken: {len(st.session_state.learning_progress.get('quizzes', []))}
Topics Covered: {', '.join(st.session_state.rag_system.topic_tracker.get_all_topics())}

Include:
1. Learning summary
2. Strengths and weaknesses
3. Recommendations for improvement
4. Suggested focus areas"""

                report = st.session_state.rag_system.ask(report_prompt)
                
                st.markdown("### 📄 Your Progress Report")
                st.markdown(report)
    
    with col2:
        if st.button("🗑️ Clear All Progress"):
            if st.checkbox("I'm sure I want to clear all progress"):
                st.session_state.learning_progress = {
                    'conversations': [],
                    'quizzes': [],
                    'study_plans': [],
                    'topics_covered': []
                }
                st.session_state.rag_system.clear_history()
                st.success("✅ Progress cleared!")
                st.rerun()

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _save_quiz_result(quiz, answers, grading):
    """Save quiz result to progress"""
    if 'learning_progress' not in st.session_state:
        st.session_state.learning_progress = {'quizzes': []}
    
    st.session_state.learning_progress.setdefault('quizzes', []).append({
        'topic': quiz['topic'],
        'difficulty': quiz['difficulty'],
        'answers': answers,
        'grading': grading,
        'timestamp': datetime.now(),
        'score': _extract_score(grading)
    })

def _extract_score(grading_text):
    """Extract score from grading text (simple regex)"""
    import re
    match = re.search(r'(\d+)/(\d+)', grading_text)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return "N/A"

def _get_quiz_history():
    """Get quiz history for analysis"""
    if 'learning_progress' in st.session_state:
        return st.session_state.learning_progress.get('quizzes', [])[-5:]
    return []

def _save_study_plan(plan):
    """Save study plan to progress"""
    if 'learning_progress' not in st.session_state:
        st.session_state.learning_progress = {'study_plans': []}
    
    st.session_state.learning_progress.setdefault('study_plans', []).append(plan)



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
        render_practice_view()  # NEW
    elif selected_view == 'Study Plan':
        render_study_plan_view()  # NEW
    elif selected_view == 'Progress Tracker':
        render_progress_tracker_view()  # NEW
    elif selected_view == 'System Logs':
        render_logs_view()

if __name__ == "__main__":
    main()