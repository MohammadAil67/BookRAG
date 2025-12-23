"""
Study plan view with interactive progress tracking and translation support
"""
import streamlit as st
import json
from datetime import datetime, timedelta
from utils.quiz_helpers import get_quiz_history, save_study_plan
from utils.session_state import get_ui_text
from utils.translation_utils import get_translator

def render_study_plan_view():
    """Render study plan generator with translation support"""
    st.header(get_ui_text('study_plan_header'))
    
    if not st.session_state.rag_system:
        st.warning(get_ui_text('load_pdf_warning_plan'))
        return
    
    # Initialize translator if needed
    if st.session_state.translator is None:
        st.session_state.translator = get_translator()
    
    # Initialize study plan state if not exists
    if 'study_plan_data' not in st.session_state:
        st.session_state.study_plan_data = {
            'mode': None,  # 'manual' or 'auto'
            'topics': [],  # List of topic objects with parts
            'current_topic_index': 0,
            'created': None,
            'completed_topics': []
        }
    
    # Initialize navigation flags if not exists
    if 'navigate_to_chat' not in st.session_state:
        st.session_state.navigate_to_chat = None
    if 'navigate_to_practice' not in st.session_state:
        st.session_state.navigate_to_practice = None
    
    # Mode Selection
    if not st.session_state.study_plan_data['mode']:
        _render_mode_selection()
    else:
        # Show active plan
        _render_active_plan()


def _render_mode_selection():
    """Render mode selection interface"""
    st.subheader(get_ui_text('plan_type_header'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown(f"### 🤖 {get_ui_text('auto_mode_title')}")
            st.write(get_ui_text('auto_mode_desc'))
            if st.button(get_ui_text('select_auto_mode'), type="primary", use_container_width=True):
                st.session_state.study_plan_data['mode'] = 'auto'
                st.rerun()
    
    with col2:
        with st.container():
            st.markdown(f"### ✏️ {get_ui_text('manual_mode_title')}")
            st.write(get_ui_text('manual_mode_desc'))
            if st.button(get_ui_text('select_manual_mode'), use_container_width=True):
                st.session_state.study_plan_data['mode'] = 'manual'
                st.rerun()


def _render_active_plan():
    """Render the active study plan interface"""
    # Header with mode indicator and reset button
    col1, col2 = st.columns([3, 1])
    with col1:
        mode_label = get_ui_text('auto_mode_title') if st.session_state.study_plan_data['mode'] == 'auto' else get_ui_text('manual_mode_title')
        st.info(f"📋 {get_ui_text('active_plan')}: {mode_label}")
    with col2:
        if st.button(get_ui_text('reset_plan'), type="secondary"):
            st.session_state.study_plan_data = {
                'mode': None,
                'topics': [],
                'current_topic_index': 0,
                'created': None,
                'completed_topics': []
            }
            st.rerun()
    
    st.divider()
    
    # If no topics, show generation interface
    if not st.session_state.study_plan_data['topics']:
        if st.session_state.study_plan_data['mode'] == 'auto':
            _render_auto_generation()
        else:
            _render_manual_creation()
    else:
        # Show plan dashboard
        _render_plan_dashboard()


def _render_auto_generation():
    """Generate study plan automatically based on student profile"""
    st.subheader(get_ui_text('auto_analysis_header'))
    st.info(get_ui_text('auto_analysis_info'))
    
    # Get quiz history for analysis
    quiz_results = get_quiz_history()
    
    # Display current performance overview
    if quiz_results:
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_score = sum([q['score'] for q in quiz_results]) / len(quiz_results)
            st.metric(get_ui_text('avg_performance'), f"{avg_score:.1f}%")
        with col2:
            st.metric(get_ui_text('quizzes_completed'), len(quiz_results))
        with col3:
            weak_topics = [q['topic'] for q in quiz_results if q['score'] < 70]
            st.metric(get_ui_text('weak_areas'), len(set(weak_topics)))
    
    if st.button(get_ui_text('generate_auto_plan'), type="primary"):
        _generate_automatic_plan(quiz_results)


def _generate_automatic_plan(quiz_results):
    """Generate automatic study plan using RAG analysis"""
    translator = st.session_state.translator
    interface_lang = st.session_state.interface_language
    
    with st.spinner(get_ui_text('analyzing_progress')):
        rag = st.session_state.rag_system
        
        # Prepare analysis prompt
        analysis_prompt = f"""Based on this student's learning data, create a personalized study plan.

Quiz Performance:
{json.dumps(quiz_results[-10:], indent=2)}

Student needs:
1. Identify 4-5 topics they are weak at or haven't covered
2. For each topic, break it into 3-4 progressive parts/lessons
3. Prioritize weak areas first
4. Each part should have:
   - Clear learning objectives
   - Estimated study time (15-30 mins)
   - Key concepts to master

Respond in JSON format:
{{
  "topics": [
    {{
      "name": "Topic Name",
      "priority": "high/medium/low",
      "reason": "Why this topic is important",
      "parts": [
        {{
          "part_number": 1,
          "title": "Part Title",
          "objectives": ["objective1", "objective2"],
          "estimated_time": 20,
          "key_concepts": ["concept1", "concept2"]
        }}
      ]
    }}
  ]
}}"""
        
        # Get plan from RAG
        response = rag.ask(analysis_prompt)
        
        # Parse JSON response
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
                
                # Translate to Bengali if needed
                if interface_lang == 'bn':
                    plan_data = _translate_plan_data(plan_data, translator)
                
                # Initialize plan with completion tracking
                topics = []
                for topic in plan_data['topics']:
                    topic_obj = {
                        **topic,
                        'completed': False,
                        'current_part': 0,
                        'parts_status': [
                            {
                                **part,
                                'completed': False,
                                'quiz_score': None,
                                'attempts': 0
                            }
                            for part in topic['parts']
                        ]
                    }
                    topics.append(topic_obj)
                
                st.session_state.study_plan_data['topics'] = topics
                st.session_state.study_plan_data['created'] = datetime.now()
                st.success(get_ui_text('plan_generated'))
                st.rerun()
            else:
                st.error(get_ui_text('plan_generation_failed'))
        except Exception as e:
            st.error(f"{get_ui_text('error')}: {str(e)}")


def _render_manual_creation():
    """Manual study plan creation interface"""
    st.subheader(get_ui_text('manual_creation_header'))
    
    # Topic input
    with st.form("manual_plan_form"):
        topics_input = st.text_area(
            get_ui_text('enter_topics'),
            placeholder=get_ui_text('topics_placeholder'),
            height=150
        )
        
        col1, col2 = st.columns(2)
        with col1:
            duration = st.selectbox(
                get_ui_text('plan_duration'),
                [get_ui_text('3_days'), get_ui_text('7_days'), 
                 get_ui_text('14_days'), get_ui_text('30_days')]
            )
        with col2:
            parts_per_topic = st.slider(
                get_ui_text('parts_per_topic'),
                min_value=2,
                max_value=5,
                value=3
            )
        
        submitted = st.form_submit_button(get_ui_text('create_plan'), type="primary")
        
        if submitted and topics_input:
            _create_manual_plan(topics_input, duration, parts_per_topic)


def _create_manual_plan(topics_input, duration, parts_per_topic):
    """Create manual study plan"""
    translator = st.session_state.translator
    interface_lang = st.session_state.interface_language
    
    with st.spinner(get_ui_text('creating_plan')):
        rag = st.session_state.rag_system
        
        # Translate topics to English if needed
        topics_for_processing = topics_input
        if interface_lang == 'bn':
            topic_lang = translator.detect_language(topics_input)
            if topic_lang == 'bn':
                topics_for_processing = translator.translate_bn_to_en(topics_input)
        
        # Generate structured plan
        topics_list = [t.strip() for t in topics_for_processing.split('\n') if t.strip()]
        
        prompt = f"""Create a detailed study plan for these topics:
{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(topics_list))}

For each topic, create {parts_per_topic} progressive parts/lessons.
Duration: {duration}

For each part, provide:
- Clear title
- Learning objectives (2-3 points)
- Estimated study time (15-30 minutes)
- Key concepts to master

Respond in JSON format:
{{
  "topics": [
    {{
      "name": "Topic Name",
      "priority": "medium",
      "parts": [
        {{
          "part_number": 1,
          "title": "Part Title",
          "objectives": ["objective1", "objective2"],
          "estimated_time": 20,
          "key_concepts": ["concept1", "concept2"]
        }}
      ]
    }}
  ]
}}"""
        
        response = rag.ask(prompt)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
                
                # Translate back to Bengali if needed
                if interface_lang == 'bn':
                    plan_data = _translate_plan_data(plan_data, translator)
                
                # Initialize with tracking
                topics = []
                for topic in plan_data['topics']:
                    topic_obj = {
                        **topic,
                        'completed': False,
                        'current_part': 0,
                        'parts_status': [
                            {
                                **part,
                                'completed': False,
                                'quiz_score': None,
                                'attempts': 0
                            }
                            for part in topic['parts']
                        ]
                    }
                    topics.append(topic_obj)
                
                st.session_state.study_plan_data['topics'] = topics
                st.session_state.study_plan_data['created'] = datetime.now()
                st.success(get_ui_text('plan_created'))
                st.rerun()
        except Exception as e:
            st.error(f"{get_ui_text('error')}: {str(e)}")


def _render_plan_dashboard():
    """Render interactive study plan dashboard"""
    # Overall progress
    total_parts = sum(len(t['parts_status']) for t in st.session_state.study_plan_data['topics'])
    completed_parts = sum(sum(1 for p in t['parts_status'] if p['completed']) for t in st.session_state.study_plan_data['topics'])
    progress_pct = (completed_parts / total_parts * 100) if total_parts > 0 else 0
    
    st.subheader(get_ui_text('overall_progress'))
    st.progress(progress_pct / 100)
    st.write(f"**{completed_parts}/{total_parts}** {get_ui_text('parts_completed')} ({progress_pct:.1f}%)")
    
    st.divider()
    
    # Topics list with expandable sections
    for idx, topic in enumerate(st.session_state.study_plan_data['topics']):
        _render_topic_card(topic, idx)
    
    # Action buttons
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(get_ui_text('save_progress'), type="primary"):
            save_study_plan(st.session_state.study_plan_data)
            st.success(get_ui_text('progress_saved'))
    with col2:
        if st.button(get_ui_text('export_plan')):
            _export_plan(st.session_state.study_plan_data)
    with col3:
        if st.button(get_ui_text('suggest_next')):
            _suggest_next_topic()


def _render_topic_card(topic, topic_idx):
    """Render individual topic card with parts"""
    # Calculate topic progress
    completed_parts = sum(1 for p in topic['parts_status'] if p['completed'])
    total_parts = len(topic['parts_status'])
    topic_progress = (completed_parts / total_parts * 100) if total_parts > 0 else 0
    
    # Topic header
    status_icon = "✅" if topic['completed'] else "📚"
    priority_colors = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
    priority_icon = priority_colors.get(topic.get('priority', 'medium'), '🟡')
    
    with st.expander(
        f"{status_icon} {topic['name']} {priority_icon} - {topic_progress:.0f}% {get_ui_text('complete')}",
        expanded=(not topic['completed'] and topic_idx == st.session_state.study_plan_data['current_topic_index'])
    ):
        # Topic details
        if 'reason' in topic:
            st.info(f"**{get_ui_text('why_important')}:** {topic['reason']}")
        
        # Progress bar
        st.progress(topic_progress / 100)
        st.write(f"**{get_ui_text('progress')}:** {completed_parts}/{total_parts} {get_ui_text('parts_completed')}")
        
        st.divider()
        
        # Parts list
        for part_idx, part in enumerate(topic['parts_status']):
            _render_part_section(topic, topic_idx, part, part_idx)


def _render_part_section(topic, topic_idx, part, part_idx):
    """Render individual part/lesson section"""
    is_locked = part_idx > 0 and not topic['parts_status'][part_idx - 1]['completed']
    is_current = part_idx == topic['current_part'] and not topic['completed']
    
    # Part header
    if part['completed']:
        icon = "✅"
        status = get_ui_text('completed')
    elif is_locked:
        icon = "🔒"
        status = get_ui_text('locked')
    elif is_current:
        icon = "▶️"
        status = get_ui_text('in_progress')
    else:
        icon = "⏸️"
        status = get_ui_text('pending')
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### {icon} {get_ui_text('part')} {part['part_number']}: {part['title']}")
    with col2:
        st.caption(f"{status} | {part['estimated_time']} {get_ui_text('min')}")
    
    if not is_locked:
        # Objectives
        st.markdown(f"**{get_ui_text('objectives')}:**")
        for obj in part['objectives']:
            st.write(f"• {obj}")
        
        # Key concepts
        with st.expander(get_ui_text('key_concepts')):
            for concept in part['key_concepts']:
                st.write(f"• {concept}")
        
        # Action buttons
        if not part['completed']:
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    get_ui_text('study_material'),
                    key=f"study_{topic_idx}_{part_idx}",
                    use_container_width=True
                ):
                    _show_study_material(topic, part)
            with col2:
                if st.button(
                    get_ui_text('take_quiz'),
                    key=f"quiz_{topic_idx}_{part_idx}",
                    type="primary",
                    use_container_width=True
                ):
                    _initiate_part_quiz(topic_idx, part_idx, topic, part)
        else:
            # Show quiz score
            if part['quiz_score'] is not None:
                score_color = "🟢" if part['quiz_score'] >= 70 else "🟡"
                st.success(f"{score_color} {get_ui_text('quiz_score')}: {part['quiz_score']:.1f}% ({part['attempts']} {get_ui_text('attempts')})")
    else:
        st.warning(get_ui_text('complete_previous_part'))
    
    st.divider()


def _show_study_material(topic, part):
    """Navigate to chat with study material request"""
    objectives_text = "\n".join(f"- {obj}" for obj in part['objectives'])
    
    query = f"""I want to learn about: {topic['name']} - {part['title']}

Learning Objectives:
{objectives_text}

Please explain these concepts in detail with examples."""
    
    # Set navigation flag
    st.session_state.navigate_to_chat = {
        'query': query,
        'topic': topic['name'],
        'part': part['title']
    }
    
    # Clear conflicting flags
    st.session_state.navigate_to_practice = None
    st.session_state.study_material_processed = False
    
    # Change view to Chat
    st.session_state.view = 'Chat'
    st.rerun()



def _initiate_part_quiz(topic_idx, part_idx, topic, part):
    """Navigate to practice with quiz generation request"""
    quiz_topic = f"{topic['name']}: {part['title']}"
    
    st.session_state.navigate_to_practice = {
        'topic': quiz_topic,
        'objectives': part['objectives'],
        'difficulty': 'Medium',
        'num_questions': 10,
        'topic_idx': topic_idx,
        'part_idx': part_idx,
        'from_study_plan': True
    }
    
    # Clear conflicting flags and reset quiz state
    st.session_state.navigate_to_chat = None
    st.session_state.regenerate_study_quiz = True
    
    # Reset quiz state to force new generation
    st.session_state.quiz_state = {
        'active': False,
        'data': [],
        'user_answers': {},
        'submitted': False,
        'score': 0
    }
    
    # Change view to Practice
    st.session_state.view = 'Practice'
    st.rerun()
    
def _translate_plan_data(plan_data, translator):
    """Translate plan data to Bengali"""
    try:
        for topic in plan_data['topics']:
            topic['name'] = translator.translate_en_to_bn(topic['name'])
            if 'reason' in topic:
                topic['reason'] = translator.translate_en_to_bn(topic['reason'])
            
            for part in topic['parts']:
                part['title'] = translator.translate_en_to_bn(part['title'])
                part['objectives'] = [translator.translate_en_to_bn(obj) for obj in part['objectives']]
                part['key_concepts'] = [translator.translate_en_to_bn(c) for c in part['key_concepts']]
    except Exception as e:
        print(f"⚠️ Translation error in plan data: {e}")
    
    return plan_data


def _export_plan(plan_data):
    """Export study plan as text file"""
    export_text = f"{get_ui_text('study_plan_export')}\n{'='*60}\n\n"
    
    if plan_data.get('created'):
        export_text += f"{get_ui_text('created')}: {plan_data['created'].strftime('%Y-%m-%d')}\n"
    export_text += f"{get_ui_text('mode')}: {plan_data.get('mode', 'N/A')}\n\n"
    
    for topic in plan_data.get('topics', []):
        export_text += f"\n## {topic['name']}\n"
        export_text += f"{get_ui_text('priority')}: {topic.get('priority', 'N/A')}\n\n"
        
        for part in topic.get('parts_status', []):
            status = "✅" if part.get('completed', False) else "⬜"
            export_text += f"{status} {get_ui_text('part')} {part['part_number']}: {part['title']}\n"
            export_text += f"   {get_ui_text('time')}: {part['estimated_time']} {get_ui_text('min')}\n"
            if part.get('quiz_score'):
                export_text += f"   {get_ui_text('score')}: {part['quiz_score']:.1f}%\n"
            export_text += "\n"
    
    st.download_button(
        get_ui_text('download_plan'),
        export_text,
        file_name=f"study_plan_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )


def _suggest_next_topic():
    """Suggest next topic based on progress"""
    translator = st.session_state.translator
    interface_lang = st.session_state.interface_language
    
    with st.spinner(get_ui_text('analyzing')):
        rag = st.session_state.rag_system
        
        completed = [t['name'] for t in st.session_state.study_plan_data['topics'] if t.get('completed', False)]
        in_progress = [t['name'] for t in st.session_state.study_plan_data['topics'] if not t.get('completed', False)]
        
        # Translate topic names to English for RAG
        completed_en = completed
        in_progress_en = in_progress
        
        if interface_lang == 'bn':
            completed_en = [translator.translate_bn_to_en(t) for t in completed]
            in_progress_en = [translator.translate_bn_to_en(t) for t in in_progress]
        
        prompt = f"""Based on completed topics: {', '.join(completed_en)}
And current topics: {', '.join(in_progress_en)}

Suggest 3 related topics the student should study next to:
1. Reinforce weak areas
2. Build on completed knowledge
3. Fill knowledge gaps

Be specific and relevant to the PDF content."""
        
        suggestion = rag.ask(prompt)
        
        # Translate suggestion if needed
        if interface_lang == 'bn':
            suggestion = translator.translate_en_to_bn(suggestion)
        
        st.info(f"💡 {get_ui_text('suggested_topics')}:")
        st.write(suggestion)