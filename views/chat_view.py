"""
Chat view for AI Tutor with Translation Support and Study Plan Integration
"""
import streamlit as st
import time
import os
import sys
from io import StringIO
from datetime import datetime
from utils.rag_utils import initialize_rag
from utils.file_utils import search_for_pdf
from utils.session_state import get_predefined_pdfs, get_ui_text
from utils.translation_utils import get_translator

def render_chat_view():
    """Render the chat interface"""
    st.header(get_ui_text('chat_header'))

    # Check for study plan navigation - DON'T RETURN EARLY
    if st.session_state.get('navigate_to_chat'):
        _handle_study_plan_navigation()
        # Don't show PDF selection when navigating from study plan
        return  # Now we can return because we've handled everything

    # Normal chat flow continues below...
    # PDF Selection Area
    st.subheader(get_ui_text('select_pdf'))
    
    selection_method = st.radio(
        "How would you like to select a PDF?",
        [
            get_ui_text('predefined_pdfs'),
            get_ui_text('browse_files'),
            get_ui_text('enter_filename')
        ],
        horizontal=True
    )
    
    st.divider()
    
    selected_pdf_path = None
    pdf_name = None
    
    # Method 1: Predefined PDFs
    if selection_method == get_ui_text('predefined_pdfs'):
        selected_pdf_path, pdf_name = _handle_predefined_pdfs()
    
    # Method 2: File Browser
    elif selection_method == get_ui_text('browse_files'):
        selected_pdf_path, pdf_name = _handle_file_upload()
    
    # Method 3: Manual Filename
    elif selection_method == get_ui_text('enter_filename'):
        selected_pdf_path, pdf_name = _handle_manual_filename()
    
    # Load RAG system if a new PDF is selected
    if selected_pdf_path:
        _load_rag_system(selected_pdf_path, pdf_name, selection_method)

    # Display chat history
    _display_chat_history()

    # Handle chat input
    _handle_chat_input()


def _handle_study_plan_navigation():
    """Handle navigation from study plan with auto-generated query"""
    nav_data = st.session_state.navigate_to_chat
    
    # Show context banner
    st.info(f"📚 **{get_ui_text('study_material')}**: {nav_data['topic']} - {nav_data['part']}")
    
    if st.button(get_ui_text('back_to_plan'), type="secondary"):
        st.session_state.navigate_to_chat = None
        st.session_state.study_material_processed = False
        st.session_state.view = 'Study Plan'
        st.rerun()
    
    st.divider()
    
    # Check if RAG system is loaded
    if not st.session_state.rag_system:
        st.error(get_ui_text('select_pdf_first'))
        st.info("👆 Please select a PDF above to continue")
        return  # This return is OK because we can't proceed without RAG
    
    # Auto-submit the query
    query = nav_data['query']
    
    # Check if this query was already processed
    if not st.session_state.get('study_material_processed'):
        # Initialize translator if needed
        if st.session_state.translator is None:
            st.session_state.translator = get_translator()
        
        translator = st.session_state.translator
        interface_lang = st.session_state.interface_language
        
        # Detect query language
        query_lang = translator.detect_language(query)
        
        # Add user message
        st.session_state.messages.append({
            "id": str(time.time()),
            "text": query,
            "sender": "user",
            "timestamp": datetime.now()
        })
        
        # Generate AI response
        with st.spinner(get_ui_text('thinking')):
            try:
                # Translate query if needed
                query_for_processing = query
                if query_lang == 'bn':
                    query_for_processing = translator.translate_bn_to_en(query)
                
                # Process with RAG
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()
                
                english_response = st.session_state.rag_system.ask(query_for_processing)
                
                sys.stdout = old_stdout
                
                # Translate response if needed
                final_response = english_response
                if interface_lang == 'bn':
                    final_response = translator.translate_en_to_bn(english_response)
                
                response_text = final_response
                
            except Exception as e:
                sys.stdout = old_stdout
                import traceback
                error_msg = f"{get_ui_text('error')} {str(e)}"
                response_text = error_msg
                st.error(response_text)
        
        # Add AI response
        st.session_state.messages.append({
            "id": str(time.time() + 1),
            "text": response_text,
            "sender": "ai",
            "timestamp": datetime.now()
        })
        
        # Mark as processed
        st.session_state.study_material_processed = True
        st.rerun()
    
    # Display chat history
    _display_chat_history()
    
    # Allow continued conversation
    if prompt := st.chat_input(get_ui_text('ask_anything')):
        # Clear the processed flag and navigation
        st.session_state.study_material_processed = False
        st.session_state.navigate_to_chat = None
        
        # Process new query normally
        _process_user_query(prompt)


def _handle_predefined_pdfs():
    """Handle predefined PDF selection"""
    PREDEFINED_PDFS = get_predefined_pdfs()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.session_state.selected_pdf:
            st.info(f"{get_ui_text('active_pdf')} {st.session_state.selected_pdf['name']}")
    with col2:
        pdf_names = [p['name'] for p in PREDEFINED_PDFS]
        selected_name = st.selectbox(
            get_ui_text('choose_pdf'),
            ["None"] + pdf_names,
            label_visibility="collapsed"
        )
        if selected_name != "None":
            found_pdf = next((p for p in PREDEFINED_PDFS if p['name'] == selected_name), None)
            if found_pdf:
                return found_pdf.get('path'), found_pdf['name']
    return None, None


def _handle_file_upload():
    """Handle file upload"""
    st.info(get_ui_text('upload_tip'))
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'], key="pdf_uploader")
    if uploaded_file:
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"{get_ui_text('uploaded')} {uploaded_file.name}")
        return temp_path, uploaded_file.name
    return None, None


def _handle_manual_filename():
    """Handle manual filename entry"""
    col1, col2 = st.columns([3, 1])
    with col1:
        filename = st.text_input(
            get_ui_text('enter_pdf_filename'),
            placeholder="physics.pdf",
            key="manual_filename"
        )
    with col2:
        search_button = st.button(get_ui_text('search'), type="primary")
    
    if search_button and filename:
        found_path = search_for_pdf(filename)
        if found_path:
            st.success(f"{get_ui_text('found')} {found_path}")
            return found_path, os.path.basename(found_path)
        else:
            st.error(f"{get_ui_text('not_found')} '{filename}'")
    return None, None


def _load_rag_system(selected_pdf_path, pdf_name, selection_method):
    """Load RAG system with selected PDF"""
    current_path = st.session_state.selected_pdf.get('path') if st.session_state.selected_pdf else None
    
    if current_path != selected_pdf_path:
        with st.spinner(f"{get_ui_text('loading')} {pdf_name}..."):
            rag, error = initialize_rag(selected_pdf_path)
            st.session_state.rag_system = rag
            st.session_state.rag_error = error
            
            if rag:
                st.session_state.selected_pdf = {
                    "name": pdf_name,
                    "path": selected_pdf_path,
                    "type": selection_method
                }
                
                # Clear messages and add welcome message in current language
                st.session_state.messages = []
                
                welcome_messages = {
                    'en': f"I've loaded '{pdf_name}'. What would you like to learn?",
                    'bn': f"আমি '{pdf_name}' লোড করেছি। আপনি কী শিখতে চান?"
                }
                
                st.session_state.messages.append({
                    "id": str(time.time()),
                    "text": welcome_messages[st.session_state.interface_language],
                    "sender": "ai",
                    "timestamp": datetime.now()
                })
                st.rerun()


def _display_chat_history():
    """Display chat history"""
    for msg in st.session_state.messages:
        avatar = "🤖" if msg['sender'] == 'ai' else "👤"
        with st.chat_message(msg['sender'], avatar=avatar):
            st.write(msg['text'])


def _handle_chat_input():
    """Handle user chat input with translation logic"""
    if prompt := st.chat_input(get_ui_text('ask_anything')):
        _process_user_query(prompt)


def _process_user_query(prompt):
    """Process user query (extracted for reuse)"""
    if not st.session_state.rag_system:
        st.error(get_ui_text('select_pdf_first'))
        return
    
    # Initialize translator if needed
    if st.session_state.translator is None:
        st.session_state.translator = get_translator()
    
    translator = st.session_state.translator
    interface_lang = st.session_state.interface_language
    
    # Detect query language
    query_lang = translator.detect_language(prompt)
    
    # Add user message (original)
    st.session_state.messages.append({
        "id": str(time.time()),
        "text": prompt,
        "sender": "user",
        "timestamp": datetime.now()
    })
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    # Generate AI response with translation logic
    with st.chat_message("ai", avatar="🤖"):
        with st.spinner(get_ui_text('thinking')):
            try:
                # Step 1: Translate query to English if needed
                query_for_processing = prompt
                
                if query_lang == 'bn':
                    # Bengali query → translate to English
                    with st.status(get_ui_text('translating_query'), expanded=False):
                        query_for_processing = translator.translate_bn_to_en(prompt)
                        st.write(f"**Translated Query:** {query_for_processing}")
                
                # Step 2: Process in English (RAG system always works in English)
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()
                
                english_response = st.session_state.rag_system.ask(query_for_processing)
                
                sys.stdout = old_stdout
                
                # Step 3: Translate response if interface is Bengali
                final_response = english_response
                
                if interface_lang == 'bn':
                    # English response → translate to Bengali
                    with st.status(get_ui_text('translating_response'), expanded=False):
                        final_response = translator.translate_en_to_bn(english_response)
                        st.write(f"**Original (EN):** {english_response[:200]}...")
                
                # Display final response
                st.write(final_response)
                response_text = final_response
                
            except Exception as e:
                sys.stdout = old_stdout
                import traceback
                error_msg = f"{get_ui_text('error')} {str(e)}"
                response_text = error_msg
                st.error(response_text)
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    # Add AI response to history (in interface language)
    st.session_state.messages.append({
        "id": str(time.time() + 1),
        "text": response_text,
        "sender": "ai",
        "timestamp": datetime.now()
    })
    st.rerun()