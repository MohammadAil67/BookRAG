"""
Quiz and learning progress helper functions
"""
import streamlit as st
from datetime import datetime
import re

def save_quiz_result(quiz_state, score, percentage):
    """
    Save quiz result to learning progress
    
    Args:
        quiz_state: Current quiz state dict
        score: Number of correct answers
        percentage: Score percentage
    """
    if 'learning_progress' not in st.session_state:
        st.session_state.learning_progress = {'quizzes': []}
    
    st.session_state.learning_progress.setdefault('quizzes', []).append({
        'topic': quiz_state.get('topic', 'Unknown'),
        'difficulty': quiz_state.get('difficulty', 'N/A'),
        'score': score,
        'total': len(quiz_state.get('data', [])),
        'percentage': percentage,
        'timestamp': datetime.now()
    })


def get_quiz_history():
    """
    Get recent quiz history for analysis
    
    Returns:
        list: List of recent quiz results (last 5)
    """
    if 'learning_progress' in st.session_state:
        return st.session_state.learning_progress.get('quizzes', [])[-5:]
    return []


def save_study_plan(plan):
    """
    Save study plan to learning progress
    
    Args:
        plan: Study plan dict
    """
    if 'learning_progress' not in st.session_state:
        st.session_state.learning_progress = {'study_plans': []}
    
    st.session_state.learning_progress.setdefault('study_plans', []).append(plan)


def extract_score(grading_text):
    """
    Extract score from grading text using regex
    
    Args:
        grading_text: Text containing score information
        
    Returns:
        str: Extracted score in "X/Y" format or "N/A"
    """
    match = re.search(r'(\d+)/(\d+)', grading_text)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return "N/A"