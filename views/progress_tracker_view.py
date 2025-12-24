"""
Progress tracker view for monitoring learning progress with Translation Support
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.session_state import get_ui_text
from utils.translation_utils import get_translator
from topics import EnhancedTopicTracker

def render_progress_tracker_view():
    """Render progress tracking dashboard"""
    st.header(get_ui_text('progress_header'))
    
    if not st.session_state.rag_system:
        st.warning(get_ui_text('no_data_warning'))
        return
    
    # Summary Stats
    _render_summary_stats()
    
    st.divider()
    
    # Performance Graph
    _render_performance_graph()
    
    st.divider()
    
    # Detailed Views in Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        get_ui_text('tab_conversations'),
        get_ui_text('tab_quiz_results'),
        get_ui_text('tab_topics'),
        get_ui_text('tab_study_plans')
    ])
    
    with tab1:
        _render_conversation_history()
    
    with tab2:
        _render_quiz_history()
    
    with tab3:
        _render_topics_covered()
    
    with tab4:
        _render_study_plans()
    
    st.divider()
    
    # Export and Clear Options
    _render_progress_actions()


def _render_summary_stats():
    """Render summary statistics"""
    st.subheader(get_ui_text('summary_title'))
    
    col1, col2, col3, col4 = st.columns(4)
    
    rag = st.session_state.rag_system
    
    with col1:
        total_questions = len(rag.chat_history)
        st.metric(get_ui_text('questions_asked'), total_questions)
    
    with col2:
        quizzes_taken = len(st.session_state.learning_progress.get('quizzes', []))
        st.metric(get_ui_text('quizzes_taken'), quizzes_taken)
    
    with col3:
        topics_covered = len(rag.topic_tracker.get_all_topics())
        st.metric(get_ui_text('topics_covered'), topics_covered)
    
    with col4:
        study_plans = len(st.session_state.learning_progress.get('study_plans', []))
        st.metric(get_ui_text('study_plans'), study_plans)


def _render_performance_graph():
    """Render quiz performance graph"""
    st.subheader(get_ui_text('performance_graph_title'))
    
    quizzes = st.session_state.learning_progress.get('quizzes', [])
    
    if not quizzes:
        st.info(get_ui_text('no_quiz_data'))
        return
    
    # Prepare data for graph
    dates = []
    scores = []
    topics = []
    
    for quiz in quizzes:
        dates.append(quiz['timestamp'])
        scores.append(quiz.get('percentage', 0))
        topics.append(quiz.get('topic', 'Unknown'))
    
    # Create interactive graph with Plotly
    fig = go.Figure()
    
    # Add line trace
    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode='lines+markers',
        name=get_ui_text('quiz_scores'),
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=10, color='#3b82f6'),
        hovertemplate='<b>%{text}</b><br>' +
                      get_ui_text('score_label') + ': %{y:.1f}%<br>' +
                      get_ui_text('date_label') + ': %{x}<br>' +
                      '<extra></extra>',
        text=topics
    ))
    
    # Add target line at 80%
    fig.add_hline(
        y=80, 
        line_dash="dash", 
        line_color="green",
        annotation_text=get_ui_text('target_score'),
        annotation_position="right"
    )
    
    # Add average line
    if scores:
        avg_score = sum(scores) / len(scores)
        fig.add_hline(
            y=avg_score,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"{get_ui_text('average_label')}: {avg_score:.1f}%",
            annotation_position="left"
        )
    
    # Update layout
    fig.update_layout(
        title=get_ui_text('quiz_performance_over_time'),
        xaxis_title=get_ui_text('date_label'),
        yaxis_title=get_ui_text('score_percentage'),
        yaxis=dict(range=[0, 105]),
        hovermode='closest',
        showlegend=True,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(get_ui_text('best_score'), f"{max(scores):.1f}%")
    with col2:
        st.metric(get_ui_text('average_score'), f"{avg_score:.1f}%")
    with col3:
        st.metric(get_ui_text('latest_score'), f"{scores[-1]:.1f}%")


def _render_conversation_history():
    """Render conversation history tab"""
    st.subheader(get_ui_text('recent_conversations'))
    history = st.session_state.rag_system.chat_history
    
    if history:
        for i, turn in enumerate(reversed(history[-20:])):
            with st.expander(f"Q{len(history)-i}: {turn['user'][:60]}..."):
                st.write(f"**{get_ui_text('question_label')}:** {turn['user']}")
                st.write(f"**{get_ui_text('answer_label')}:** {turn['ai'][:300]}...")
    else:
        st.info(get_ui_text('no_conversations'))


def _render_quiz_history():
    """Render quiz results tab"""
    st.subheader(get_ui_text('quiz_performance'))
    quizzes = st.session_state.learning_progress.get('quizzes', [])
    
    if quizzes:
        for i, quiz in enumerate(reversed(quizzes[-10:])):
            score = quiz.get('score', 0)
            total = quiz.get('total', 0)
            percentage = quiz.get('percentage', 0)
            
            # Emoji based on performance
            if percentage >= 80:
                emoji = "🌟"
            elif percentage >= 60:
                emoji = "👍"
            else:
                emoji = "📚"
            
            with st.expander(f"{emoji} {get_ui_text('quiz_label')} {len(quizzes)-i}: {quiz['topic']} - {quiz['timestamp'].strftime('%b %d')}"):
                st.write(f"**{get_ui_text('topic_label')}:** {quiz['topic']}")
                st.write(f"**{get_ui_text('difficulty')}:** {quiz.get('difficulty', 'N/A')}")
                st.write(f"**{get_ui_text('score_label')}:** {score}/{total}")
                st.write(f"**{get_ui_text('percentage_label')}:** {percentage:.1f}%")
                st.write(f"**{get_ui_text('date_label')}:** {quiz['timestamp'].strftime('%B %d, %Y at %I:%M %p')}")
                
                # Progress bar
                st.progress(percentage / 100)
    else:
        st.info(get_ui_text('no_quizzes'))


def _render_topics_covered():
    """Render topics covered tab"""
    st.subheader(get_ui_text('topics_explored'))
    rag = st.session_state.rag_system

    topic_tracker = rag.topic_tracker
    topics = topic_tracker.get_all_topics()
    
    if topics:
        topic_info = topic_tracker.get_topic_hints()
        current_topic = topic_info.get('current_topic', 'N/A')
        
        st.write(f"**{get_ui_text('currently_studying')}:** {current_topic}")
        st.write(f"**{get_ui_text('all_topics_covered')}:**")
        
        for topic in topics:
            st.markdown(f"- {topic}")
        
        if st.button(get_ui_text('suggest_next_topics')):
            # Initialize translator if needed
            if st.session_state.translator is None:
                st.session_state.translator = get_translator()
            
            translator = st.session_state.translator
            interface_lang = st.session_state.interface_language
            
            with st.spinner(get_ui_text('analyzing_gaps')):
                suggest_prompt = f"""Based on these topics the student has covered:
{', '.join(topics)}

And the PDF content available, suggest 3-5 related topics they should explore next.
Make suggestions that build on what they know."""

                # Get suggestions in English
                suggestions_en = rag.ask(suggest_prompt)
                
                # Translate if interface is Bengali
                if interface_lang == 'bn':
                    suggestions = translator.translate_en_to_bn(suggestions_en)
                else:
                    suggestions = suggestions_en
                
                st.markdown(f"### {get_ui_text('suggested_topics_header')}")
                st.markdown(suggestions)
    else:
        st.info(get_ui_text('no_topics_yet'))


def _render_study_plans():
    """Render study plans tab"""
    st.subheader(get_ui_text('your_study_plans'))
    plans = st.session_state.learning_progress.get('study_plans', [])
    
    if plans:
        for i, plan in enumerate(reversed(plans)):
            with st.expander(f"{get_ui_text('plan_label')} {len(plans)-i}: {plan['type'].title()} - {plan['created'].strftime('%b %d')}"):
                st.markdown(plan['plan'][:500] + "...")
                if st.button(get_ui_text('view_full_plan').format(i), key=f"view_plan_{i}"):
                    st.markdown(plan['plan'])
    else:
        st.info(get_ui_text('no_study_plans'))


def _render_progress_actions():
    """Render export and clear actions"""
    st.subheader(get_ui_text('export_progress'))
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(get_ui_text('generate_report')):
            # Initialize translator if needed
            if st.session_state.translator is None:
                st.session_state.translator = get_translator()
            
            translator = st.session_state.translator
            interface_lang = st.session_state.interface_language
            
            with st.spinner(get_ui_text('creating_report')):
                rag = st.session_state.rag_system
                topics = ', '.join(rag.topic_tracker.get_all_topics())
                
                report_prompt = f"""Create a comprehensive progress report for this student:

Total Questions: {len(rag.chat_history)}
Quizzes Taken: {len(st.session_state.learning_progress.get('quizzes', []))}
Topics Covered: {topics}

Include:
1. Learning summary
2. Strengths and weaknesses
3. Recommendations for improvement
4. Suggested focus areas"""

                # Generate report in English
                report_en = rag.ask(report_prompt)
                
                # Translate if interface is Bengali
                if interface_lang == 'bn':
                    report = translator.translate_en_to_bn(report_en)
                else:
                    report = report_en
                
                st.markdown(f"### {get_ui_text('progress_report_header')}")
                st.markdown(report)
    
    with col2:
        if st.button(get_ui_text('clear_progress')):
            if st.checkbox(get_ui_text('confirm_clear')):
                st.session_state.learning_progress = {
                    'conversations': [],
                    'quizzes': [],
                    'study_plans': [],
                    'topics_covered': []
                }
                st.session_state.rag_system.clear_history()
                st.success(get_ui_text('progress_cleared'))
                st.rerun()