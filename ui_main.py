"""
Main entry point for AI Tutor application
"""
import streamlit as st
from datetime import datetime
from utils.session_state import initialize_session_state
from components.sidebar import render_sidebar
from views.chat_view import render_chat_view
from views.practice_view import render_practice_view
from views.study_plan_view import render_study_plan_view
from views.progress_tracker_view import render_progress_tracker_view
from views.logs_view import render_logs_view

# Set page config
st.set_page_config(
    page_title="AI Tutor", 
    page_icon="📚", 
    layout="wide"
)

def main():
    # Initialize all session state
    initialize_session_state()
    
    # PRIORITY: Handle navigation from study plan FIRST
    if st.session_state.get('navigate_to_chat'):
        st.session_state.view = 'Chat'
    elif st.session_state.get('navigate_to_practice'):
        st.session_state.view = 'Practice'
    
    # Render sidebar and get selected view
    selected_view = render_sidebar()
    
    # Update view from sidebar selection
    if selected_view != st.session_state.view:
        # Clear navigation flags when user manually changes view
        st.session_state.navigate_to_chat = None
        st.session_state.navigate_to_practice = None
        st.session_state.view = selected_view
        st.rerun()
    
    # Route to appropriate view based on session state
    if st.session_state.view == 'Chat':
        render_chat_view()
    elif st.session_state.view == 'Practice':
        render_practice_view()
    elif st.session_state.view == 'Study Plan':
        render_study_plan_view()
    elif st.session_state.view == 'Progress Tracker':
        render_progress_tracker_view()
    elif st.session_state.view == 'System Logs':
        render_logs_view()

if __name__ == "__main__":
    main()