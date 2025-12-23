"""
Practice view for interactive quizzes with Translation Support and Study Plan Integration
"""
import streamlit as st
from datetime import datetime
from utils.quiz_helpers import save_quiz_result
from utils.session_state import get_ui_text
from utils.translation_utils import get_translator

def render_practice_view():
    """Render the practice/quiz interface"""
    st.header(get_ui_text('practice_header'))
    
    if not st.session_state.rag_system:
        st.warning(get_ui_text('load_pdf_warning'))
        st.info("👆 Please load a PDF in the Chat section first")
        return
    
    # Check for study plan navigation - HANDLE IT COMPLETELY
    if st.session_state.get('navigate_to_practice'):
        _handle_study_plan_quiz_generation()
        # Don't show quiz setup when navigating from study plan
        return  # Now we can return because we've handled everything
    
    # Normal quiz flow continues below...
    rag = st.session_state.rag_system

    # Quiz Setup Section
    if not st.session_state.quiz_state['active']:
        _render_quiz_setup(rag)
    
    # Take Quiz Section
    if st.session_state.quiz_state['active']:
        _render_quiz_form()
    
    # Results Section
    if st.session_state.quiz_state['submitted']:
        _render_quiz_results()


def _handle_study_plan_quiz_generation():
    """Handle quiz generation from study plan"""
    nav_data = st.session_state.navigate_to_practice
    
    # Show context banner
    st.info(f"📝 **{get_ui_text('quiz_for')}**: {nav_data['topic']}")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(get_ui_text('study_plan_quiz_info'))
    with col2:
        if st.button(get_ui_text('back_to_plan'), type="secondary"):
            st.session_state.navigate_to_practice = None
            st.session_state.regenerate_study_quiz = False
            st.session_state.view = 'Study Plan'
            st.rerun()
    
    st.divider()
    
    # Check if quiz needs to be generated
    if not st.session_state.quiz_state['active'] or st.session_state.get('regenerate_study_quiz'):
        _generate_study_plan_quiz(nav_data)
        return  # Exit after generation to wait for rerun
    
    # Show quiz (quiz is already generated and active)
    if st.session_state.quiz_state['active'] and not st.session_state.quiz_state['submitted']:
        _render_quiz_form(from_study_plan=True)
    
    # Results with study plan integration
    if st.session_state.quiz_state['submitted']:
        _render_quiz_results(from_study_plan=True)



def _generate_study_plan_quiz(nav_data):
    """Generate quiz for study plan part"""
    translator = st.session_state.translator if st.session_state.translator else get_translator()
    interface_lang = st.session_state.interface_language
    
    # Translate topic to English for processing
    topic_for_processing = nav_data['topic']
    topic_lang = translator.detect_language(topic_for_processing)
    
    if topic_lang == 'bn':
        topic_for_processing = translator.translate_bn_to_en(topic_for_processing)
    
    with st.spinner(get_ui_text('generating_quiz').format(nav_data['difficulty'], nav_data['topic'])):
        rag = st.session_state.rag_system
        
        # Generate quiz
        quiz_data = rag.generate_quiz(
            topic_for_processing, 
            nav_data['difficulty'], 
            nav_data['num_questions']
        )
        
        if quiz_data:
            # Translate quiz if interface is Bengali
            if interface_lang == 'bn':
                quiz_data = _translate_quiz(quiz_data, translator)
            
            st.session_state.quiz_state = {
                'active': True,
                'data': quiz_data,
                'user_answers': {},
                'submitted': False,
                'score': 0,
                'topic': nav_data['topic'],
                'difficulty': nav_data['difficulty'],
                'from_study_plan': True,
                'study_plan_context': {
                    'topic_idx': nav_data['topic_idx'],
                    'part_idx': nav_data['part_idx']
                }
            }
            st.session_state.regenerate_study_quiz = False
            st.success(get_ui_text('quiz_ready'))
            st.rerun()
        else:
            st.error(get_ui_text('quiz_generation_error'))


def _render_quiz_setup(rag):
    """Render quiz configuration section"""
    with st.expander(get_ui_text('quiz_config'), expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Simple topic input without auto-fill
            topic = st.text_input(
                get_ui_text('topic_to_practice'), 
                value="", 
                placeholder=get_ui_text('topic_placeholder')
            )
        
        with col2:
            difficulty = st.selectbox(
                get_ui_text('difficulty'), 
                [get_ui_text('easy'), get_ui_text('medium'), get_ui_text('hard')]
            )
            num_questions = st.number_input(
                get_ui_text('num_questions'), 
                min_value=3, 
                max_value=10, 
                value=5
            )

        if st.button(get_ui_text('start_quiz'), type="primary"):
            if not topic:
                st.error(get_ui_text('enter_topic_error'))
            else:
                # Initialize translator if needed
                if st.session_state.translator is None:
                    st.session_state.translator = get_translator()
                
                translator = st.session_state.translator
                interface_lang = st.session_state.interface_language
                
                # Detect topic language
                topic_lang = translator.detect_language(topic)
                
                # Translate topic to English if needed (for processing)
                topic_for_processing = topic
                if topic_lang == 'bn':
                    topic_for_processing = translator.translate_bn_to_en(topic)
                    print(f"🔄 Translated topic: '{topic}' → '{topic_for_processing}'")
                
                # Map difficulty to English for processing
                difficulty_map = {
                    get_ui_text('easy'): 'Easy',
                    get_ui_text('medium'): 'Medium',
                    get_ui_text('hard'): 'Hard'
                }
                difficulty_en = difficulty_map.get(difficulty, 'Medium')
                
                with st.spinner(get_ui_text('generating_quiz').format(difficulty, topic)):
                    # Generate quiz in English
                    quiz_data = rag.generate_quiz(topic_for_processing, difficulty_en, int(num_questions))
                    
                    if quiz_data:
                        # Translate quiz if interface is Bengali
                        if interface_lang == 'bn':
                            quiz_data = _translate_quiz(quiz_data, translator)
                        
                        st.session_state.quiz_state = {
                            'active': True,
                            'data': quiz_data,
                            'user_answers': {},
                            'submitted': False,
                            'score': 0,
                            'topic': topic,  # Keep original topic
                            'difficulty': difficulty
                        }
                        st.rerun()
                    else:
                        st.error(get_ui_text('quiz_generation_error'))


def _translate_quiz(quiz_data: list, translator) -> list:
    """Translate quiz questions and options to Bengali"""
    translated_quiz = []
    
    for q in quiz_data:
        try:
            translated_q = {
                'id': q['id'],
                'question': translator.translate_en_to_bn(q['question']),
                'options': [translator.translate_en_to_bn(opt) for opt in q['options']],
                'correct_answer': translator.translate_en_to_bn(q['correct_answer']),
                'explanation': translator.translate_en_to_bn(q['explanation'])
            }
            translated_quiz.append(translated_q)
        except Exception as e:
            print(f"⚠️ Translation error for question {q['id']}: {e}")
            # Fall back to original if translation fails
            translated_quiz.append(q)
    
    return translated_quiz


def _render_quiz_form(from_study_plan=False):
    """Render the quiz questions form"""
    st.divider()
    st.subheader(f"{get_ui_text('topic_label')}: {st.session_state.quiz_state.get('topic', 'General')}")
    
    with st.form(key='quiz_form'):
        questions = st.session_state.quiz_state['data']
        
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            
            existing_answer = st.session_state.quiz_state['user_answers'].get(i, None)
            
            user_choice = st.radio(
                get_ui_text('choose_answer'), 
                q['options'], 
                key=f"q_{i}", 
                index=None if existing_answer is None else q['options'].index(existing_answer),
                label_visibility="collapsed"
            )
            st.write("")

        submit_label = get_ui_text('submit_quiz') if not st.session_state.quiz_state['submitted'] else get_ui_text('update_answers')
        submitted = st.form_submit_button(submit_label)
        
        if submitted:
            st.session_state.quiz_state['submitted'] = True
            # Save answers
            for i in range(len(questions)):
                val = st.session_state.get(f"q_{i}")
                st.session_state.quiz_state['user_answers'][i] = val
            st.rerun()


def _render_quiz_results(from_study_plan=False):
    """Render quiz results and explanations"""
    st.divider()
    st.header(get_ui_text('results_header'))
    
    questions = st.session_state.quiz_state['data']
    answers = st.session_state.quiz_state['user_answers']
    score = 0
    
    # Grade each question
    for i, q in enumerate(questions):
        user_ans = answers.get(i)
        correct_ans = q['correct_answer']
        is_correct = (user_ans == correct_ans)
        
        if is_correct: 
            score += 1
            icon = "✅"
        else:
            icon = "❌"

        with st.container():
            col_a, col_b = st.columns([0.05, 0.95])
            with col_a:
                st.write(icon)
            with col_b:
                st.markdown(f"**Q{i+1}:** {q['question']}")
                
                if is_correct:
                    st.success(f"**{get_ui_text('your_answer')}:** {user_ans}")
                else:
                    st.error(f"**{get_ui_text('your_answer')}:** {user_ans}")
                    st.info(f"**{get_ui_text('correct_answer')}:** {correct_ans}")
                
                with st.expander(get_ui_text('explanation_for').format(i+1)):
                    st.write(q['explanation'])
        st.divider()

    # Score Summary
    percentage = (score / len(questions)) * 100
    st.metric(get_ui_text('final_score'), f"{score}/{len(questions)}", f"{percentage:.1f}%")
    
    # Save quiz result
    save_quiz_result(st.session_state.quiz_state, score, percentage)
    
    # Handle study plan completion
    if from_study_plan and st.session_state.quiz_state.get('from_study_plan'):
        _handle_study_plan_completion(percentage)
    
    col1, col2 = st.columns(2)
    with col1:
        if percentage >= 80:
            st.balloons()
            st.success(get_ui_text('great_job'))
        elif percentage >= 50:
            st.warning(get_ui_text('good_effort'))
        else:
            st.error(get_ui_text('keep_studying'))
    
    with col2:
        if from_study_plan:
            if st.button(get_ui_text('back_to_plan'), type="primary"):
                st.session_state.navigate_to_practice = None
                st.session_state.quiz_state['active'] = False
                st.session_state.view = 'Study Plan'
                st.rerun()
        else:
            if st.button(get_ui_text('start_new_quiz')):
                st.session_state.quiz_state['active'] = False
                st.rerun()


def _handle_study_plan_completion(percentage):
    """Update study plan with quiz results"""
    context = st.session_state.quiz_state.get('study_plan_context')
    
    if not context:
        return
    
    topic_idx = context['topic_idx']
    part_idx = context['part_idx']
    
    # Update the part status
    if 'study_plan_data' in st.session_state:
        topics = st.session_state.study_plan_data.get('topics', [])
        
        if topic_idx < len(topics):
            topic = topics[topic_idx]
            parts = topic.get('parts_status', [])
            
            if part_idx < len(parts):
                part = parts[part_idx]
                
                # Update part with quiz results
                part['attempts'] = part.get('attempts', 0) + 1
                
                if percentage >= 70:  # Passing threshold
                    part['completed'] = True
                    part['quiz_score'] = percentage
                    
                    # Move to next part
                    topic['current_part'] = min(part_idx + 1, len(parts) - 1)
                    
                    # Check if topic is complete
                    if all(p.get('completed', False) for p in parts):
                        topic['completed'] = True
                        st.success(f"🎉 {get_ui_text('topic_completed')}: {topic['name']}")
                    else:
                        st.success(get_ui_text('part_completed'))
                else:
                    # Allow retry
                    part['quiz_score'] = percentage
                    st.warning(get_ui_text('retry_recommended'))