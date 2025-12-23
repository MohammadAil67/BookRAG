"""
Session state initialization and management
"""
import streamlit as st
from datetime import datetime

# Constants
PREDEFINED_PDFS = [
    {"id": "1", "name": "Bangla Shahitto", "type": "predefined", "path": "bangla_shahitto.pdf"},
    {"id": "2", "name": "Physics - Class 10", "type": "predefined", "path": "Physics  9-10 EV book full pdf_compressed.pdf"},
    {"id": "6", "name": "English Literature", "type": "predefined", "path": "Dakhil - 2018 - Class-(9-10) English For Today PDF Web.pdf"},
]

def initialize_session_state():
    """Initialize all session state variables"""
    
    # Language settings
    if 'interface_language' not in st.session_state:
        st.session_state.interface_language = 'en'  # 'en' or 'bn'
    
    # Chat messages
    if 'messages' not in st.session_state:
        welcome_message = {
            'en': "Hello! I'm your AI tutor. How can I help you learn today? You can select a subject PDF or upload your own document to get started.",
            'bn': "হ্যালো! আমি আপনার এআই শিক্ষক। আজ আমি কিভাবে আপনাকে শিখতে সাহায্য করতে পারি? আপনি একটি বিষয় PDF নির্বাচন করতে পারেন বা শুরু করতে আপনার নিজস্ব নথি আপলোড করতে পারেন।"
        }
        st.session_state.messages = [
            {
                "id": "1",
                "text": welcome_message[st.session_state.interface_language],
                "sender": "ai",
                "timestamp": datetime.now()
            }
        ]
    
    # PDF management
    if 'uploaded_pdfs' not in st.session_state:
        st.session_state.uploaded_pdfs = []
    
    if 'selected_pdf' not in st.session_state:
        st.session_state.selected_pdf = None
    
    # View state
    if 'view' not in st.session_state:
        st.session_state.view = 'Chat'
    
    # RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    
    if 'rag_error' not in st.session_state:
        st.session_state.rag_error = None
    
    # Translation
    if 'translator' not in st.session_state:
        st.session_state.translator = None
    
    # Quiz state
    if 'quiz_state' not in st.session_state:
        st.session_state.quiz_state = {
            'active': False,
            'data': [],
            'user_answers': {},
            'submitted': False,
            'score': 0
        }
    
    # Learning progress
    if 'learning_progress' not in st.session_state:
        st.session_state.learning_progress = {
            'conversations': [],
            'quizzes': [],
            'study_plans': [],
            'topics_covered': []
        }
    
    # Study plan
    if 'current_study_plan' not in st.session_state:
        st.session_state.current_study_plan = None

def get_predefined_pdfs():
    """Get list of predefined PDFs"""
    return PREDEFINED_PDFS

def get_ui_text(key: str) -> str:
    """
    Get UI text in current interface language
    
    Args:
        key: Text identifier
    
    Returns:
        Localized text
    """
    texts = {
        'chat_header': {
            'en': '💬 AI Tutor Chat',
            'bn': '💬 এআই শিক্ষক চ্যাট'
        },
        'select_pdf': {
            'en': '📚 Select a PDF Document',
            'bn': '📚 একটি PDF নথি নির্বাচন করুন'
        },
        'predefined_pdfs': {
            'en': 'Predefined PDFs',
            'bn': 'পূর্বনির্ধারিত PDFs'
        },
        'browse_files': {
            'en': 'Browse Files',
            'bn': 'ফাইল ব্রাউজ করুন'
        },
        'enter_filename': {
            'en': 'Enter Filename',
            'bn': 'ফাইলের নাম লিখুন'
        },
        'active_pdf': {
            'en': '📄 **Active PDF:**',
            'bn': '📄 **সক্রিয় PDF:**'
        },
        'choose_pdf': {
            'en': 'Choose PDF',
            'bn': 'PDF নির্বাচন করুন'
        },
        'upload_tip': {
            'en': '💡 **Tip:** Upload or drag & drop your PDF file',
            'bn': '💡 **পরামর্শ:** আপনার PDF ফাইল আপলোড করুন বা ড্র্যাগ এবং ড্রপ করুন'
        },
        'uploaded': {
            'en': '✅ Uploaded:',
            'bn': '✅ আপলোড হয়েছে:'
        },
        'enter_pdf_filename': {
            'en': 'Enter PDF filename',
            'bn': 'PDF ফাইলের নাম লিখুন'
        },
        'search': {
            'en': '🔍 Search',
            'bn': '🔍 খুঁজুন'
        },
        'found': {
            'en': '✅ Found:',
            'bn': '✅ পাওয়া গেছে:'
        },
        'not_found': {
            'en': '❌ Could not find',
            'bn': '❌ খুঁজে পাওয়া যায়নি'
        },
        'loading': {
            'en': '📚 Loading',
            'bn': '📚 লোড হচ্ছে'
        },
        'ask_anything': {
            'en': 'Ask me anything...',
            'bn': 'আমাকে যেকোনো কিছু জিজ্ঞাসা করুন...'
        },
        'select_pdf_first': {
            'en': '⚠️ Please select a PDF first!',
            'bn': '⚠️ প্রথমে একটি PDF নির্বাচন করুন!'
        },
        'thinking': {
            'en': '🧠 Thinking...',
            'bn': '🧠 চিন্তা করছি...'
        },
        'error': {
            'en': '❌ Error:',
            'bn': '❌ ত্রুটি:'
        },
        'translating_query': {
            'en': '🔄 Translating query...',
            'bn': '🔄 প্রশ্ন অনুবাদ হচ্ছে...'
        },
        'translating_response': {
            'en': '🔄 Translating response...',
            'bn': '🔄 উত্তর অনুবাদ হচ্ছে...'
        },
        'practice_header': {
            'en': '📝 Interactive Practice Mode',
            'bn': '📝 ইন্টারেক্টিভ অনুশীলন মোড'
        },
        'load_pdf_warning': {
            'en': '⚠️ Please load a PDF first to generate practice questions.',
            'bn': '⚠️ অনুশীলন প্রশ্ন তৈরি করতে প্রথমে একটি PDF লোড করুন।'
        },
        'quiz_config': {
            'en': '⚙️ Quiz Configuration',
            'bn': '⚙️ কুইজ কনফিগারেশন'
        },
        'topic_to_practice': {
            'en': 'Topic to Practice',
            'bn': 'অনুশীলনের বিষয়'
        },
        'topic_placeholder': {
            'en': "e.g. Newton's Third Law",
            'bn': 'যেমন: নিউটনের তৃতীয় সূত্র'
        },
        'difficulty': {
            'en': 'Difficulty',
            'bn': 'অসুবিধা স্তর'
        },
        'easy': {
            'en': 'Easy',
            'bn': 'সহজ'
        },
        'medium': {
            'en': 'Medium',
            'bn': 'মাঝারি'
        },
        'hard': {
            'en': 'Hard',
            'bn': 'কঠিন'
        },
        'num_questions': {
            'en': 'Number of Questions',
            'bn': 'প্রশ্নের সংখ্যা'
        },
        'start_quiz': {
            'en': '🚀 Start Quiz',
            'bn': '🚀 কুইজ শুরু করুন'
        },
        'enter_topic_error': {
            'en': 'Please enter a topic.',
            'bn': 'অনুগ্রহ করে একটি বিষয় লিখুন।'
        },
        'generating_quiz': {
            'en': '🧠 Generating {} questions for "{}"...',
            'bn': '🧠 "{}" এর জন্য {} প্রশ্ন তৈরি হচ্ছে...'
        },
        'quiz_generation_error': {
            'en': 'Could not generate questions. Please try a different topic.',
            'bn': 'প্রশ্ন তৈরি করা যায়নি। অনুগ্রহ করে একটি ভিন্ন বিষয় চেষ্টা করুন।'
        },
        'topic_label': {
            'en': 'Topic',
            'bn': 'বিষয়'
        },
        'choose_answer': {
            'en': 'Choose Answer:',
            'bn': 'উত্তর নির্বাচন করুন:'
        },
        'submit_quiz': {
            'en': '✅ Submit Quiz',
            'bn': '✅ কুইজ জমা দিন'
        },
        'update_answers': {
            'en': '🔄 Update Answers',
            'bn': '🔄 উত্তর আপডেট করুন'
        },
        'results_header': {
            'en': '📊 Results',
            'bn': '📊 ফলাফল'
        },
        'your_answer': {
            'en': 'Your Answer',
            'bn': 'আপনার উত্তর'
        },
        'correct_answer': {
            'en': 'Correct Answer',
            'bn': 'সঠিক উত্তর'
        },
        'explanation_for': {
            'en': '📖 Explanation for Q{}',
            'bn': '📖 Q{} এর ব্যাখ্যা'
        },
        'final_score': {
            'en': 'Final Score',
            'bn': 'চূড়ান্ত স্কোর'
        },
        'great_job': {
            'en': '🌟 Great job! You know this topic well.',
            'bn': '🌟 দুর্দান্ত! আপনি এই বিষয়টি ভালোভাবে জানেন।'
        },
        'good_effort': {
            'en': '👍 Good effort. Review the explanations above.',
            'bn': '👍 ভালো প্রচেষ্টা। উপরের ব্যাখ্যাগুলি পর্যালোচনা করুন।'
        },
        'keep_studying': {
            'en': '💪 Keep studying. Try the "Study Plan" tab for help.',
            'bn': '💪 পড়াশোনা চালিয়ে যান। সাহায্যের জন্য "পড়াশোনার পরিকল্পনা" ট্যাব চেষ্টা করুন।'
        },
        'start_new_quiz': {
            'en': '🔄 Start New Quiz',
            'bn': '🔄 নতুন কুইজ শুরু করুন'
        },
        'progress_header': {
            'en': '📈 Progress Tracker',
            'bn': '📈 অগ্রগতি ট্র্যাকার'
        },
        'no_data_warning': {
            'en': '⚠️ No data yet. Start learning to track your progress!',
            'bn': '⚠️ এখনও কোন ডেটা নেই। আপনার অগ্রগতি ট্র্যাক করতে শিখতে শুরু করুন!'
        },
        'summary_title': {
            'en': '📊 Summary',
            'bn': '📊 সারাংশ'
        },
        'questions_asked': {
            'en': 'Questions Asked',
            'bn': 'জিজ্ঞাসিত প্রশ্ন'
        },
        'quizzes_taken': {
            'en': 'Quizzes Taken',
            'bn': 'কুইজ নেওয়া হয়েছে'
        },
        'topics_covered': {
            'en': 'Topics Covered',
            'bn': 'বিষয় আচ্ছাদিত'
        },
        'study_plans': {
            'en': 'Study Plans',
            'bn': 'পড়াশোনার পরিকল্পনা'
        },
        'performance_graph_title': {
            'en': '📊 Performance Over Time',
            'bn': '📊 সময়ের সাথে কর্মক্ষমতা'
        },
        'no_quiz_data': {
            'en': 'No quiz data available. Take some quizzes to see your progress!',
            'bn': 'কোন কুইজ ডেটা উপলব্ধ নেই। আপনার অগ্রগতি দেখতে কিছু কুইজ নিন!'
        },
        'quiz_scores': {
            'en': 'Quiz Scores',
            'bn': 'কুইজ স্কোর'
        },
        'score_label': {
            'en': 'Score',
            'bn': 'স্কোর'
        },
        'date_label': {
            'en': 'Date',
            'bn': 'তারিখ'
        },
        'target_score': {
            'en': 'Target: 80%',
            'bn': 'লক্ষ্য: ৮০%'
        },
        'average_label': {
            'en': 'Average',
            'bn': 'গড়'
        },
        'quiz_performance_over_time': {
            'en': 'Quiz Performance Over Time',
            'bn': 'সময়ের সাথে কুইজ কর্মক্ষমতা'
        },
        'score_percentage': {
            'en': 'Score (%)',
            'bn': 'স্কোর (%)'
        },
        'best_score': {
            'en': 'Best Score',
            'bn': 'সেরা স্কোর'
        },
        'average_score': {
            'en': 'Average Score',
            'bn': 'গড় স্কোর'
        },
        'latest_score': {
            'en': 'Latest Score',
            'bn': 'সর্বশেষ স্কোর'
        },
        'tab_conversations': {
            'en': '💬 Conversation History',
            'bn': '💬 কথোপকথনের ইতিহাস'
        },
        'tab_quiz_results': {
            'en': '📝 Quiz Results',
            'bn': '📝 কুইজ ফলাফল'
        },
        'tab_topics': {
            'en': '🎯 Topics Covered',
            'bn': '🎯 বিষয় আচ্ছাদিত'
        },
        'tab_study_plans': {
            'en': '📅 Study Plans',
            'bn': '📅 পড়াশোনার পরিকল্পনা'
        },
        'recent_conversations': {
            'en': 'Recent Conversations',
            'bn': 'সাম্প্রতিক কথোপকথন'
        },
        'question_label': {
            'en': 'Question',
            'bn': 'প্রশ্ন'
        },
        'answer_label': {
            'en': 'Answer',
            'bn': 'উত্তর'
        },
        'no_conversations': {
            'en': 'No conversations yet. Start chatting!',
            'bn': 'এখনও কোন কথোপকথন নেই। চ্যাট শুরু করুন!'
        },
        'quiz_performance': {
            'en': 'Quiz Performance',
            'bn': 'কুইজ কর্মক্ষমতা'
        },
        'quiz_label': {
            'en': 'Quiz',
            'bn': 'কুইজ'
        },
        'percentage_label': {
            'en': 'Percentage',
            'bn': 'শতাংশ'
        },
        'no_quizzes': {
            'en': 'No quizzes taken yet. Try the Practice section!',
            'bn': 'এখনও কোন কুইজ নেওয়া হয়নি। অনুশীলন বিভাগ চেষ্টা করুন!'
        },
        'topics_explored': {
            'en': 'Topics You\'ve Explored',
            'bn': 'আপনি যে বিষয়গুলি অন্বেষণ করেছেন'
        },
        'currently_studying': {
            'en': 'Currently Studying',
            'bn': 'বর্তমানে অধ্যয়ন করছেন'
        },
        'all_topics_covered': {
            'en': 'All Topics Covered',
            'bn': 'সমস্ত বিষয় আচ্ছাদিত'
        },
        'suggest_next_topics': {
            'en': '🔮 Suggest Next Topics',
            'bn': '🔮 পরবর্তী বিষয় পরামর্শ দিন'
        },
        'analyzing_gaps': {
            'en': 'Analyzing gaps...',
            'bn': 'ফাঁক বিশ্লেষণ করা হচ্ছে...'
        },
        'suggested_topics_header': {
            'en': '💡 Suggested Next Topics',
            'bn': '💡 পরামর্শকৃত পরবর্তী বিষয়'
        },
        'no_topics_yet': {
            'en': 'Start chatting to track topics!',
            'bn': 'বিষয় ট্র্যাক করতে চ্যাট শুরু করুন!'
        },
        'your_study_plans': {
            'en': 'Your Study Plans',
            'bn': 'আপনার পড়াশোনার পরিকল্পনা'
        },
        'plan_label': {
            'en': 'Plan',
            'bn': 'পরিকল্পনা'
        },
        'view_full_plan': {
            'en': 'View Full Plan {}',
            'bn': 'সম্পূর্ণ পরিকল্পনা {} দেখুন'
        },
        'no_study_plans': {
            'en': 'No study plans yet. Create one in the Study Plan section!',
            'bn': 'এখনও কোন পড়াশোনার পরিকল্পনা নেই। স্টাডি প্ল্যান বিভাগে একটি তৈরি করুন!'
        },
        'export_progress': {
            'en': '📤 Export Progress',
            'bn': '📤 অগ্রগতি রপ্তানি করুন'
        },
        'generate_report': {
            'en': '📊 Generate Progress Report',
            'bn': '📊 অগ্রগতি রিপোর্ট তৈরি করুন'
        },
        'creating_report': {
            'en': 'Creating report...',
            'bn': 'রিপোর্ট তৈরি হচ্ছে...'
        },
        'progress_report_header': {
            'en': '📄 Your Progress Report',
            'bn': '📄 আপনার অগ্রগতি রিপোর্ট'
        },
        'clear_progress': {
            'en': '🗑️ Clear All Progress',
            'bn': '🗑️ সমস্ত অগ্রগতি মুছুন'
        },
        'confirm_clear': {
            'en': 'I\'m sure I want to clear all progress',
            'bn': 'আমি নিশ্চিত যে আমি সমস্ত অগ্রগতি মুছতে চাই'
        },
        'progress_cleared': {
            'en': '✅ Progress cleared!',
            'bn': '✅ অগ্রগতি মুছে ফেলা হয়েছে!'
        },
        'study_plan_header': {
            'en': '📅 Study Plan Generator',
            'bn': '📅 অধ্যয়ন পরিকল্পনা জেনারেটর'
        },
        'load_pdf_warning_plan': {
            'en': '⚠️ Please load a PDF first to generate a study plan.',
            'bn': '⚠️ অধ্যয়ন পরিকল্পনা তৈরি করতে প্রথমে একটি PDF লোড করুন।'
        },
        
        # Mode Selection
        'plan_type_header': {
            'en': '🎯 Study Plan Type',
            'bn': '🎯 অধ্যয়ন পরিকল্পনার ধরন'
        },
        'auto_mode_title': {
            'en': 'Auto-Detect Weak Areas',
            'bn': 'দুর্বল এলাকা স্বয়ংক্রিয়-সনাক্তকরণ'
        },
        'auto_mode_desc': {
            'en': 'AI analyzes your quiz performance and conversation history to identify weak areas and create a personalized plan.',
            'bn': 'AI আপনার কুইজ পারফরম্যান্স এবং কথোপকথনের ইতিহাস বিশ্লেষণ করে দুর্বল এলাকা চিহ্নিত করে এবং একটি ব্যক্তিগতকৃত পরিকল্পনা তৈরি করে।'
        },
        'manual_mode_title': {
            'en': 'Manual Topic Selection',
            'bn': 'ম্যানুয়াল বিষয় নির্বাচন'
        },
        'manual_mode_desc': {
            'en': 'Choose your own topics and customize your study plan based on your preferences and curriculum.',
            'bn': 'আপনার পছন্দ এবং পাঠ্যক্রমের উপর ভিত্তি করে আপনার নিজের বিষয় চয়ন করুন এবং আপনার অধ্যয়ন পরিকল্পনা কাস্টমাইজ করুন।'
        },
        'select_auto_mode': {
            'en': '🤖 Select Auto Mode',
            'bn': '🤖 স্বয়ংক্রিয় মোড নির্বাচন করুন'
        },
        'select_manual_mode': {
            'en': '✏️ Select Manual Mode',
            'bn': '✏️ ম্যানুয়াল মোড নির্বাচন করুন'
        },
        
        # Active Plan
        'active_plan': {
            'en': 'Active Plan',
            'bn': 'সক্রিয় পরিকল্পনা'
        },
        'reset_plan': {
            'en': '🔄 Reset Plan',
            'bn': '🔄 পরিকল্পনা রিসেট করুন'
        },
        
        # Auto Generation
        'auto_analysis_header': {
            'en': '📊 Performance Analysis',
            'bn': '📊 কর্মক্ষমতা বিশ্লেষণ'
        },
        'auto_analysis_info': {
            'en': '📈 We\'ll analyze your conversation history and quiz results to identify weak areas and generate a personalized study plan.',
            'bn': '📈 আমরা আপনার কথোপকথনের ইতিহাস এবং কুইজ ফলাফল বিশ্লেষণ করব দুর্বল এলাকা চিহ্নিত করতে এবং একটি ব্যক্তিগতকৃত অধ্যয়ন পরিকল্পনা তৈরি করতে।'
        },
        'avg_performance': {
            'en': 'Average Performance',
            'bn': 'গড় কর্মক্ষমতা'
        },
        'quizzes_completed': {
            'en': 'Quizzes Completed',
            'bn': 'কুইজ সম্পন্ন'
        },
        'weak_areas': {
            'en': 'Weak Areas',
            'bn': 'দুর্বল এলাকা'
        },
        'generate_auto_plan': {
            'en': '🔍 Analyze & Generate Plan',
            'bn': '🔍 বিশ্লেষণ এবং পরিকল্পনা তৈরি করুন'
        },
        'analyzing_progress': {
            'en': '🧠 Analyzing your learning patterns...',
            'bn': '🧠 আপনার শেখার প্যাটার্ন বিশ্লেষণ করা হচ্ছে...'
        },
        'plan_generated': {
            'en': '✅ Personalized study plan generated!',
            'bn': '✅ ব্যক্তিগতকৃত অধ্যয়ন পরিকল্পনা তৈরি হয়েছে!'
        },
        'plan_generation_failed': {
            'en': '❌ Could not generate plan. Please try again.',
            'bn': '❌ পরিকল্পনা তৈরি করা যায়নি। অনুগ্রহ করে আবার চেষ্টা করুন।'
        },
        
        # Manual Creation
        'manual_creation_header': {
            'en': '✏️ Create Your Custom Plan',
            'bn': '✏️ আপনার কাস্টম পরিকল্পনা তৈরি করুন'
        },
        'enter_topics': {
            'en': 'Enter topics (one per line)',
            'bn': 'বিষয়গুলি লিখুন (প্রতি লাইনে একটি)'
        },
        'topics_placeholder': {
            'en': 'Zahir Raihan\nLiberation War\nLanguage Movement',
            'bn': 'জহির রায়হান\nমুক্তিযুদ্ধ\nভাষা আন্দোলন'
        },
        'plan_duration': {
            'en': 'Plan Duration',
            'bn': 'পরিকল্পনার সময়কাল'
        },
        '3_days': {
            'en': '3 days',
            'bn': '৩ দিন'
        },
        '7_days': {
            'en': '7 days',
            'bn': '৭ দিন'
        },
        '14_days': {
            'en': '14 days',
            'bn': '১৪ দিন'
        },
        '30_days': {
            'en': '30 days',
            'bn': '৩০ দিন'
        },
        'parts_per_topic': {
            'en': 'Parts per Topic',
            'bn': 'প্রতি বিষয়ে অংশ'
        },
        'create_plan': {
            'en': '🎯 Create Plan',
            'bn': '🎯 পরিকল্পনা তৈরি করুন'
        },
        'creating_plan': {
            'en': '🔨 Creating your personalized plan...',
            'bn': '🔨 আপনার ব্যক্তিগতকৃত পরিকল্পনা তৈরি করা হচ্ছে...'
        },
        'plan_created': {
            'en': '✅ Study plan created successfully!',
            'bn': '✅ অধ্যয়ন পরিকল্পনা সফলভাবে তৈরি হয়েছে!'
        },
        
        # Dashboard
        'overall_progress': {
            'en': '📊 Overall Progress',
            'bn': '📊 সামগ্রিক অগ্রগতি'
        },
        'parts_completed': {
            'en': 'parts completed',
            'bn': 'অংশ সম্পন্ন'
        },
        'complete': {
            'en': 'Complete',
            'bn': 'সম্পূর্ণ'
        },
        'why_important': {
            'en': 'Why Important',
            'bn': 'কেন গুরুত্বপূর্ণ'
        },
        'progress': {
            'en': 'Progress',
            'bn': 'অগ্রগতি'
        },
        
        # Part Status
        'part': {
            'en': 'Part',
            'bn': 'অংশ'
        },
        'completed': {
            'en': 'Completed',
            'bn': 'সম্পন্ন'
        },
        'locked': {
            'en': 'Locked',
            'bn': 'লক করা'
        },
        'in_progress': {
            'en': 'In Progress',
            'bn': 'চলমান'
        },
        'pending': {
            'en': 'Pending',
            'bn': 'অপেক্ষমান'
        },
        'min': {
            'en': 'min',
            'bn': 'মিনিট'
        },
        'objectives': {
            'en': 'Learning Objectives',
            'bn': 'শেখার উদ্দেশ্য'
        },
        'key_concepts': {
            'en': '🔑 Key Concepts',
            'bn': '🔑 মূল ধারণা'
        },
        'study_material': {
            'en': '📖 Study Material',
            'bn': '📖 অধ্যয়ন উপকরণ'
        },
        'take_quiz': {
            'en': '✅ Take Quiz',
            'bn': '✅ কুইজ নিন'
        },
        'quiz_score': {
            'en': 'Quiz Score',
            'bn': 'কুইজ স্কোর'
        },
        'attempts': {
            'en': 'attempts',
            'bn': 'প্রচেষ্টা'
        },
        'complete_previous_part': {
            'en': '🔒 Complete the previous part to unlock this section',
            'bn': '🔒 এই বিভাগটি আনলক করতে পূর্ববর্তী অংশটি সম্পূর্ণ করুন'
        },
        
        # Actions
        'save_progress': {
            'en': '💾 Save Progress',
            'bn': '💾 অগ্রগতি সংরক্ষণ করুন'
        },
        'progress_saved': {
            'en': '✅ Progress saved successfully!',
            'bn': '✅ অগ্রগতি সফলভাবে সংরক্ষিত হয়েছে!'
        },
        'export_plan': {
            'en': '📤 Export Plan',
            'bn': '📤 পরিকল্পনা রপ্তানি করুন'
        },
        'suggest_next': {
            'en': '💡 Suggest Next Topics',
            'bn': '💡 পরবর্তী বিষয় পরামর্শ দিন'
        },
        'analyzing': {
            'en': '🔍 Analyzing...',
            'bn': '🔍 বিশ্লেষণ করা হচ্ছে...'
        },
        'suggested_topics': {
            'en': 'Suggested Next Topics',
            'bn': 'পরামর্শকৃত পরবর্তী বিষয়'
        },
        
        # Export
        'study_plan_export': {
            'en': 'Study Plan Export',
            'bn': 'অধ্যয়ন পরিকল্পনা রপ্তানি'
        },
        'created': {
            'en': 'Created',
            'bn': 'তৈরি করা হয়েছে'
        },
        'mode': {
            'en': 'Mode',
            'bn': 'মোড'
        },
        'priority': {
            'en': 'Priority',
            'bn': 'অগ্রাধিকার'
        },
        'time': {
            'en': 'Time',
            'bn': 'সময়'
        },
        'score': {
            'en': 'Score',
            'bn': 'স্কোর'
        },
        'download_plan': {
            'en': '📥 Download Plan',
            'bn': '📥 পরিকল্পনা ডাউনলোড করুন'
        },
        'study_plan_header': {
        'en': '📅 Study Plan Generator',
        'bn': '📅 অধ্যয়ন পরিকল্পনা জেনারেটর'
        },
        'load_pdf_warning_plan': {
            'en': '⚠️ Please load a PDF first to generate a study plan.',
            'bn': '⚠️ অধ্যয়ন পরিকল্পনা তৈরি করতে প্রথমে একটি PDF লোড করুন।'
        },
        
        # Navigation
        'back_to_plan': {
            'en': '← Back to Study Plan',
            'bn': '← অধ্যয়ন পরিকল্পনায় ফিরে যান'
        },
        'quiz_for': {
            'en': 'Quiz for',
            'bn': 'কুইজ'
        },
        'study_plan_quiz_info': {
            'en': '10 questions • Medium difficulty • From your study plan',
            'bn': '১০টি প্রশ্ন • মাঝারি অসুবিধা • আপনার অধ্যয়ন পরিকল্পনা থেকে'
        },
        'quiz_ready': {
            'en': '✅ Quiz is ready! Answer all questions below.',
            'bn': '✅ কুইজ প্রস্তুত! নিচের সব প্রশ্নের উত্তর দিন।'
        },
        'topic_completed': {
            'en': 'Topic Completed',
            'bn': 'বিষয় সম্পন্ন'
        },
        'part_completed': {
            'en': '✅ Part completed! Next part unlocked.',
            'bn': '✅ অংশ সম্পন্ন! পরবর্তী অংশ আনলক হয়েছে।'
        },
        'retry_recommended': {
            'en': '📚 Score below 70%. Review the material and try again.',
            'bn': '📚 স্কোর ৭০% এর নিচে। উপাদান পর্যালোচনা করুন এবং আবার চেষ্টা করুন।'
        },
        
        # Mode Selection
        'plan_type_header': {
            'en': '🎯 Study Plan Type',
            'bn': '🎯 অধ্যয়ন পরিকল্পনার ধরন'
        },
        'auto_mode_title': {
            'en': 'Auto-Detect Weak Areas',
            'bn': 'দুর্বল এলাকা স্বয়ংক্রিয়-সনাক্তকরণ'
        },
        'auto_mode_desc': {
            'en': 'AI analyzes your quiz performance and conversation history to identify weak areas and create a personalized plan.',
            'bn': 'AI আপনার কুইজ পারফরম্যান্স এবং কথোপকথনের ইতিহাস বিশ্লেষণ করে দুর্বল এলাকা চিহ্নিত করে এবং একটি ব্যক্তিগতকৃত পরিকল্পনা তৈরি করে।'
        },
        'manual_mode_title': {
            'en': 'Manual Topic Selection',
            'bn': 'ম্যানুয়াল বিষয় নির্বাচন'
        },
        'manual_mode_desc': {
            'en': 'Choose your own topics and customize your study plan based on your preferences and curriculum.',
            'bn': 'আপনার পছন্দ এবং পাঠ্যক্রমের উপর ভিত্তি করে আপনার নিজের বিষয় চয়ন করুন এবং আপনার অধ্যয়ন পরিকল্পনা কাস্টমাইজ করুন।'
        },
        'select_auto_mode': {
            'en': '🤖 Select Auto Mode',
            'bn': '🤖 স্বয়ংক্রিয় মোড নির্বাচন করুন'
        },
        'select_manual_mode': {
            'en': '✏️ Select Manual Mode',
            'bn': '✏️ ম্যানুয়াল মোড নির্বাচন করুন'
        },
        
        # Active Plan
        'active_plan': {
            'en': 'Active Plan',
            'bn': 'সক্রিয় পরিকল্পনা'
        },
        'reset_plan': {
            'en': '🔄 Reset Plan',
            'bn': '🔄 পরিকল্পনা রিসেট করুন'
        },
        
        # Auto Generation
        'auto_analysis_header': {
            'en': '📊 Performance Analysis',
            'bn': '📊 কর্মক্ষমতা বিশ্লেষণ'
        },
        'auto_analysis_info': {
            'en': '📈 We\'ll analyze your conversation history and quiz results to identify weak areas and generate a personalized study plan.',
            'bn': '📈 আমরা আপনার কথোপকথনের ইতিহাস এবং কুইজ ফলাফল বিশ্লেষণ করব দুর্বল এলাকা চিহ্নিত করতে এবং একটি ব্যক্তিগতকৃত অধ্যয়ন পরিকল্পনা তৈরি করতে।'
        },
        'avg_performance': {
            'en': 'Average Performance',
            'bn': 'গড় কর্মক্ষমতা'
        },
        'quizzes_completed': {
            'en': 'Quizzes Completed',
            'bn': 'কুইজ সম্পন্ন'
        },
        'weak_areas': {
            'en': 'Weak Areas',
            'bn': 'দুর্বল এলাকা'
        },
        'generate_auto_plan': {
            'en': '🔍 Analyze & Generate Plan',
            'bn': '🔍 বিশ্লেষণ এবং পরিকল্পনা তৈরি করুন'
        },
        'analyzing_progress': {
            'en': '🧠 Analyzing your learning patterns...',
            'bn': '🧠 আপনার শেখার প্যাটার্ন বিশ্লেষণ করা হচ্ছে...'
        },
        'plan_generated': {
            'en': '✅ Personalized study plan generated!',
            'bn': '✅ ব্যক্তিগতকৃত অধ্যয়ন পরিকল্পনা তৈরি হয়েছে!'
        },
        'plan_generation_failed': {
            'en': '❌ Could not generate plan. Please try again.',
            'bn': '❌ পরিকল্পনা তৈরি করা যায়নি। অনুগ্রহ করে আবার চেষ্টা করুন।'
        },
        
        # Manual Creation
        'manual_creation_header': {
            'en': '✏️ Create Your Custom Plan',
            'bn': '✏️ আপনার কাস্টম পরিকল্পনা তৈরি করুন'
        },
        'enter_topics': {
            'en': 'Enter topics (one per line)',
            'bn': 'বিষয়গুলি লিখুন (প্রতি লাইনে একটি)'
        },
        'topics_placeholder': {
            'en': 'Zahir Raihan\nLiberation War\nLanguage Movement',
            'bn': 'জহির রায়হান\nমুক্তিযুদ্ধ\nভাষা আন্দোলন'
        },
        'plan_duration': {
            'en': 'Plan Duration',
            'bn': 'পরিকল্পনার সময়কাল'
        },
        '3_days': {
            'en': '3 days',
            'bn': '৩ দিন'
        },
        '7_days': {
            'en': '7 days',
            'bn': '৭ দিন'
        },
        '14_days': {
            'en': '14 days',
            'bn': '১৪ দিন'
        },
        '30_days': {
            'en': '30 days',
            'bn': '৩০ দিন'
        },
        'parts_per_topic': {
            'en': 'Parts per Topic',
            'bn': 'প্রতি বিষয়ে অংশ'
        },
        'create_plan': {
            'en': '🎯 Create Plan',
            'bn': '🎯 পরিকল্পনা তৈরি করুন'
        },
        'creating_plan': {
            'en': '🔨 Creating your personalized plan...',
            'bn': '🔨 আপনার ব্যক্তিগতকৃত পরিকল্পনা তৈরি করা হচ্ছে...'
        },
        'plan_created': {
            'en': '✅ Study plan created successfully!',
            'bn': '✅ অধ্যয়ন পরিকল্পনা সফলভাবে তৈরি হয়েছে!'
        },
        
        # Dashboard
        'overall_progress': {
            'en': '📊 Overall Progress',
            'bn': '📊 সামগ্রিক অগ্রগতি'
        },
        'parts_completed': {
            'en': 'parts completed',
            'bn': 'অংশ সম্পন্ন'
        },
        'complete': {
            'en': 'Complete',
            'bn': 'সম্পূর্ণ'
        },
        'why_important': {
            'en': 'Why Important',
            'bn': 'কেন গুরুত্বপূর্ণ'
        },
        'progress': {
            'en': 'Progress',
            'bn': 'অগ্রগতি'
        },
        
        # Part Status
        'part': {
            'en': 'Part',
            'bn': 'অংশ'
        },
        'completed': {
            'en': 'Completed',
            'bn': 'সম্পন্ন'
        },
        'locked': {
            'en': 'Locked',
            'bn': 'লক করা'
        },
        'in_progress': {
            'en': 'In Progress',
            'bn': 'চলমান'
        },
        'pending': {
            'en': 'Pending',
            'bn': 'অপেক্ষমান'
        },
        'min': {
            'en': 'min',
            'bn': 'মিনিট'
        },
        'objectives': {
            'en': 'Learning Objectives',
            'bn': 'শেখার উদ্দেশ্য'
        },
        'key_concepts': {
            'en': '🔑 Key Concepts',
            'bn': '🔑 মূল ধারণা'
        },
        'study_material': {
            'en': '📖 Study Material',
            'bn': '📖 অধ্যয়ন উপকরণ'
        },
        'take_quiz': {
            'en': '✅ Take Quiz',
            'bn': '✅ কুইজ নিন'
        },
        'quiz_score': {
            'en': 'Quiz Score',
            'bn': 'কুইজ স্কোর'
        },
        'attempts': {
            'en': 'attempts',
            'bn': 'প্রচেষ্টা'
        },
        'complete_previous_part': {
            'en': '🔒 Complete the previous part to unlock this section',
            'bn': '🔒 এই বিভাগটি আনলক করতে পূর্ববর্তী অংশটি সম্পূর্ণ করুন'
        },
        
        # Actions
        'save_progress': {
            'en': '💾 Save Progress',
            'bn': '💾 অগ্রগতি সংরক্ষণ করুন'
        },
        'progress_saved': {
            'en': '✅ Progress saved successfully!',
            'bn': '✅ অগ্রগতি সফলভাবে সংরক্ষিত হয়েছে!'
        },
        'export_plan': {
            'en': '📤 Export Plan',
            'bn': '📤 পরিকল্পনা রপ্তানি করুন'
        },
        'suggest_next': {
            'en': '💡 Suggest Next Topics',
            'bn': '💡 পরবর্তী বিষয় পরামর্শ দিন'
        },
        'analyzing': {
            'en': '🔍 Analyzing...',
            'bn': '🔍 বিশ্লেষণ করা হচ্ছে...'
        },
        'suggested_topics': {
            'en': 'Suggested Next Topics',
            'bn': 'পরামর্শকৃত পরবর্তী বিষয়'
        },
        
        # Export
        'study_plan_export': {
            'en': 'Study Plan Export',
            'bn': 'অধ্যয়ন পরিকল্পনা রপ্তানি'
        },
        'created': {
            'en': 'Created',
            'bn': 'তৈরি করা হয়েছে'
        },
        'mode': {
            'en': 'Mode',
            'bn': 'মোড'
        },
        'priority': {
            'en': 'Priority',
            'bn': 'অগ্রাধিকার'
        },
        'time': {
            'en': 'Time',
            'bn': 'সময়'
        },
        'score': {
            'en': 'Score',
            'bn': 'স্কোর'
        },
        'download_plan': {
            'en': '📥 Download Plan',
            'bn': '📥 পরিকল্পনা ডাউনলোড করুন'
        },
        
        # Practice/Quiz existing
        'load_pdf_warning': {
            'en': '⚠️ Please load a PDF first to generate practice questions.',
            'bn': '⚠️ অনুশীলন প্রশ্ন তৈরি করতে প্রথমে একটি PDF লোড করুন।'
        },
        'topic_to_practice': {
            'en': 'Topic to Practice',
            'bn': 'অনুশীলনের বিষয়'
        }
    }
    
    lang = st.session_state.interface_language
    return texts.get(key, {}).get(lang, key)