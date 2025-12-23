"""
RAG system initialization utilities
"""
import streamlit as st
from rag_system import RAGSystem
from config import Config

@st.cache_resource
def initialize_rag(pdf_path=None):
    """
    Initialize RAG system with optional PDF path
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        tuple: (rag_system, error_message)
    """
    try:
        config = Config(pdf_path=pdf_path) 
        rag = RAGSystem(config)
        return rag, None
    except Exception as e:
        import traceback
        return None, f"{str(e)}\n{traceback.format_exc()}"