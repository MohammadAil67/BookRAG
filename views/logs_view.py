"""
System logs and debugging view
"""
import streamlit as st

def render_logs_view():
    """Render system logs and debugging information"""
    st.header("🔧 System Logs & Debugging")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ No RAG system loaded.")
        return
    
    rag = st.session_state.rag_system
    
    # System Stats
    _render_system_stats(rag)
    
    st.divider()
    
    # Topic Tracking Info
    _render_topic_tracking(rag)
    
    # Configuration
    _render_system_config(rag)
    
    # Cached Answers
    _render_cache_content(rag)
    
    # Conversation History
    _render_conversation_log(rag)
    
    # Retrieval Stats
    _render_retrieval_stats(rag)


def _render_system_stats(rag):
    """Render system statistics"""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Chunks", len(rag.chunks))
    col2.metric("Cached Answers", len(rag.cache.cache))
    col3.metric("History Length", len(rag.chat_history))
    
    topic_info = rag.get_topic_status()
    col4.metric("Topics Discussed", len(topic_info.get('all_topics', [])))


def _render_topic_tracking(rag):
    """Render topic tracking information"""
    with st.expander("🎯 Topic Tracking", expanded=True):
        topic_info = rag.get_topic_status()
        
        if topic_info.get('current_topic'):
            st.success(f"**Current Topic:** {topic_info['current_topic']}")
            st.write(f"**Confidence:** {topic_info.get('confidence', 0):.2f}")
            if topic_info.get('current_keywords'):
                st.write(f"**Keywords:** {', '.join(topic_info['current_keywords'])}")
        else:
            st.info("No active topic yet")
        
        if topic_info.get('all_topics'):
            st.write(f"**All Topics:** {', '.join(topic_info['all_topics'])}")


def _render_system_config(rag):
    """Render system configuration"""
    with st.expander("⚙️ System Configuration"):
        st.json({
            "INITIAL_RETRIEVAL_K": rag.config.INITIAL_RETRIEVAL_K,
            "FINAL_TOP_K": rag.config.FINAL_TOP_K,
            "RERANK_THRESHOLD": rag.config.RERANK_THRESHOLD,
            "PDF_PATH": rag.config.PDF_PATH,
            "MODEL_CACHE_DIR": rag.config.MODEL_CACHE_DIR
        })


def _render_cache_content(rag):
    """Render cached answers"""
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


def _render_conversation_log(rag):
    """Render conversation history log"""
    with st.expander("💬 Conversation History"):
        if rag.history.history:
            for i, item in enumerate(rag.history.history):
                st.write(f"**Turn {i+1}**")
                st.write(f"**Q:** {item['question']}")
                answer_preview = item['answer'][:300] + "..." if len(item['answer']) > 300 else item['answer']
                st.write(f"**A:** {answer_preview}")
                st.divider()
        else:
            st.info("No conversation history yet")


def _render_retrieval_stats(rag):
    """Render retrieval system statistics"""
    with st.expander("📊 Retrieval System Stats"):
        st.write("**Retriever Stack:**")
        st.write("1. MultiQueryRetriever (generates query variants)")
        st.write("2. TopicAwareRetriever (maintains conversation context)")
        st.write("3. HybridRetriever (BM25 + Vector Search + Cross-Encoder Reranking)")
        
        if hasattr(rag.retriever, 'base_retriever'):
            topic_retriever = rag.retriever.base_retriever
            if hasattr(topic_retriever, 'get_topic_summary'):
                st.write(f"\n**Topic Summary:** {topic_retriever.get_topic_summary()}")