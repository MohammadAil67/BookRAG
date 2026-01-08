# sidebar.py
"""
Sidebar component for navigation and system status
Enhanced with quiz navigation restrictions
"""
import streamlit as st

def render_sidebar():
    """Render sidebar with navigation and status"""
    with st.sidebar:
        st.title("📚 AI Tutor")
        
        # Language Toggle
        st.markdown("### 🌐 Language / ভাষা")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(
                "🇬🇧 English",
                use_container_width=True,
                type="primary" if st.session_state.interface_language == 'en' else "secondary"
            ):
                if st.session_state.interface_language != 'en':
                    st.session_state.interface_language = 'en'
                    # Update welcome message
                    if st.session_state.messages and st.session_state.messages[0]['id'] == "1":
                        st.session_state.messages[0]['text'] = "Hello! I'm your AI tutor. How can I help you learn today? You can select a subject PDF or upload your own document to get started."
                    st.rerun()
        
        with col2:
            if st.button(
                "🇧🇩 বাংলা",
                use_container_width=True,
                type="primary" if st.session_state.interface_language == 'bn' else "secondary"
            ):
                if st.session_state.interface_language != 'bn':
                    st.session_state.interface_language = 'bn'
                    # Update welcome message
                    if st.session_state.messages and st.session_state.messages[0]['id'] == "1":
                        st.session_state.messages[0]['text'] = "হ্যালো! আমি আপনার এআই শিক্ষক। আজ আমি কিভাবে আপনাকে শিখতে সাহায্য করতে পারি? আপনি একটি বিষয় PDF নির্বাচন করতে পারেন বা শুরু করতে আপনার নিজস্ব নথি আপলোড করতে পারেন।"
                    st.rerun()
        
        # Show current language
        lang_indicator = "🇬🇧 English" if st.session_state.interface_language == 'en' else "🇧🇩 বাংলা"
        st.info(f"**Interface:** {lang_indicator}")
        
        st.divider()

        # Check if quiz is active and not submitted
        quiz_locked = st.session_state.get('quiz_locked', False)
        
        # Navigation
        nav_label = "Navigation" if st.session_state.interface_language == 'en' else "নেভিগেশন"
        st.markdown(f"### {nav_label}")
        
        # Show warning if quiz is active
        if quiz_locked:
            warning_msg = "⚠️ Quiz in progress - navigation locked" if st.session_state.interface_language == 'en' else "⚠️ কুইজ চলছে - নেভিগেশন লক করা"
            st.warning(warning_msg)
        
        nav_options = {
            'en': ['Chat', 'Practice', 'Study Plan', 'Progress Tracker'],
            'bn': ['চ্যাট', 'অনুশীলন', 'পড়াশোনার পরিকল্পনা', 'অগ্রগতি ট্র্যাকার']
        }
        
        # Base English options used for logic/indexing
        english_options = ['Chat', 'Practice', 'Study Plan', 'Progress Tracker']
        
        # Map Bengali back to English keys for routing
        option_map = {
            'চ্যাট': 'Chat',
            'অনুশীলন': 'Practice',
            'পড়াশোনার পরিকল্পনা': 'Study Plan',
            'অগ্রগতি ট্র্যাকার': 'Progress Tracker'
        }
        
        # Calculate the index based on the CURRENT session state view
        try:
            current_index = english_options.index(st.session_state.view)
        except ValueError:
            current_index = 0

        # If quiz is locked, disable navigation by forcing Practice view
        if quiz_locked:
            # Force Practice view
            selected_view = st.radio(
                "Go to",
                nav_options[st.session_state.interface_language],
                index=english_options.index('Practice'),
                label_visibility="collapsed",
                disabled=False  # Keep enabled but we'll override the selection
            )
            
            # Always return Practice when locked
            selected_view = 'Practice'
            
            # Show helper text
            helper_text = "Complete or submit the quiz to unlock navigation" if st.session_state.interface_language == 'en' else "নেভিগেশন আনলক করতে কুইজ সম্পূর্ণ বা জমা দিন"
            st.caption(f"ℹ️ {helper_text}")
        else:
            # Normal navigation
            selected_view = st.radio(
                "Go to",
                nav_options[st.session_state.interface_language],
                index=current_index,
                label_visibility="collapsed"
            )
            
            # Convert Bengali selection back to English for routing
            if st.session_state.interface_language == 'bn':
                selected_view = option_map.get(selected_view, 'Chat')
        
        st.divider()

        # RAG System Status
        status_title = "🤖 AI System Status" if st.session_state.interface_language == 'en' else "🤖 এআই সিস্টেম স্ট্যাটাস"
        with st.expander(status_title, expanded=True):
            if st.session_state.rag_system:
                success_msg = "✅ RAG System Active" if st.session_state.interface_language == 'en' else "✅ RAG সিস্টেম সক্রিয়"
                st.success(success_msg)
                
                if st.session_state.selected_pdf:
                    pdf_label = "📄 **Active PDF:**" if st.session_state.interface_language == 'en' else "📄 **সক্রিয় PDF:**"
                    st.info(f"{pdf_label}\n{st.session_state.selected_pdf['name']}")
                    
                    chunks_label = "Chunks Loaded" if st.session_state.interface_language == 'en' else "চাঙ্ক লোড হয়েছে"
                    st.metric(chunks_label, len(st.session_state.rag_system.chunks))
                
                col1, col2 = st.columns(2)
                with col1:
                    reload_btn = "🔄 Reload" if st.session_state.interface_language == 'en' else "🔄 রিলোড"
                    if st.button(reload_btn, disabled=quiz_locked):
                        st.cache_resource.clear()
                        st.session_state.rag_system = None
                        st.rerun()
                
                with col2:
                    clear_btn = "🗑️ Clear" if st.session_state.interface_language == 'en' else "🗑️ মুছুন"
                    if st.button(clear_btn, disabled=quiz_locked):
                        st.session_state.rag_system.clear_history()
                        st.session_state.messages = []
                        success_clear = "History cleared!" if st.session_state.interface_language == 'en' else "ইতিহাস মুছে ফেলা হয়েছে!"
                        st.success(success_clear)
                    
            else:
                warning_msg = "⚠️ RAG System Not Loaded" if st.session_state.interface_language == 'en' else "⚠️ RAG সিস্টেম লোড হয়নি"
                st.warning(warning_msg)
                if st.session_state.rag_error:
                    error_label = "Error:" if st.session_state.interface_language == 'en' else "ত্রুটি:"
                    st.error(f"{error_label} {st.session_state.rag_error}")

        # Student Info
        profile_title = "👤 Student Profile" if st.session_state.interface_language == 'en' else "👤 শিক্ষার্থীর প্রোফাইল"
        with st.expander(profile_title):
            st.markdown("""
            <div style="text-align: center;">
                <div style="background-color: #3b82f6; color: white; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 24px; margin: 0 auto;">AS</div>
                <h3>Karim</h3>
                <p style="color: gray;">Class 9 Student</p>
            </div>
            """, unsafe_allow_html=True)
            
            email_label = "📧 Email:" if st.session_state.interface_language == 'en' else "📧 ইমেইল:"
            score_label = "🏆 Score:" if st.session_state.interface_language == 'en' else "🏆 স্কোর:"
            
            st.write(f"**{email_label}** karim@example.com")
            st.write(f"**{score_label}** 0 points")

        return selected_view