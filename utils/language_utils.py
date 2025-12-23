"""
Language detection and translation utilities with BanglaT5
"""
import re
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LanguageManager:
    """Manages language detection and translation using BanglaT5"""
    
    def __init__(self):
        self.translations = self._load_translations()
        self._translator_model = None
        self._translator_tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_translator(self):
        """Lazy load BanglaT5 model (only when needed for translation)"""
        if self._translator_model is None:
            print("[TRANSLATION] Loading BanglaT5 model...")
            model_name = "csebuetnlp/banglat5_nmt_en_bn"
            self._translator_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._translator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._translator_model.to(self._device)
            print(f"[TRANSLATION] Model loaded on {self._device}")
    
    def detect_language(self, text: str) -> str:
        """
        Detect if text is in Bangla or English
        
        Args:
            text: Input text to detect
            
        Returns:
            str: 'bn' for Bangla, 'en' for English
        """
        # Bangla Unicode range: \u0980-\u09FF
        bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u0980-\u09FF]', text))
        
        if total_chars == 0:
            return 'en'
        
        bangla_ratio = bangla_chars / total_chars
        return 'bn' if bangla_ratio > 0.3 else 'en'
    
    def translate_bangla_to_english(self, text: str) -> str:
        """
        Translate Bangla text to English using BanglaT5
        
        Args:
            text: Bangla text
            
        Returns:
            str: English translation
        """
        try:
            self._load_translator()
            
            # BanglaT5 expects format: "translate Bangla to English: <text>"
            input_text = f"translate Bangla to English: {text}"
            
            inputs = self._translator_tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self._device)
            
            with torch.no_grad():
                outputs = self._translator_model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            translated = self._translator_tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            print(f"[TRANSLATION] Bangla -> English: {text[:50]}... -> {translated[:50]}...")
            return translated.strip()
            
        except Exception as e:
            print(f"[ERROR] Translation failed: {e}")
            return text  # Return original if translation fails
    
    def translate_english_to_bangla(self, text: str) -> str:
        """
        Translate English text to Bangla using BanglaT5
        
        Args:
            text: English text
            
        Returns:
            str: Bangla translation
        """
        try:
            self._load_translator()
            
            # BanglaT5 expects format: "translate English to Bangla: <text>"
            input_text = f"translate English to Bangla: {text}"
            
            inputs = self._translator_tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self._device)
            
            with torch.no_grad():
                outputs = self._translator_model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            translated = self._translator_tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            print(f"[TRANSLATION] English -> Bangla: {text[:50]}... -> {translated[:50]}...")
            return translated.strip()
            
        except Exception as e:
            print(f"[ERROR] Translation failed: {e}")
            return text  # Return original if translation fails
    
    def translate_query(self, query: str, query_lang: str, interface_lang: str) -> str:
        """
        Translate query based on language combination
        
        Args:
            query: Query text
            query_lang: Detected query language ('bn' or 'en')
            interface_lang: Interface language ('bn' or 'en')
            
        Returns:
            str: Translated query (always English for RAG processing)
        """
        # If query is already in English, no translation needed
        if query_lang == 'en':
            return query
        
        # If query is in Bangla, translate to English for RAG
        if query_lang == 'bn':
            return self.translate_bangla_to_english(query)
        
        return query
    
    def translate_response(self, response: str, interface_lang: str) -> str:
        """
        Translate LLM response to interface language
        
        Args:
            response: English response from LLM
            interface_lang: Target interface language ('bn' or 'en')
            
        Returns:
            str: Translated response
        """
        # If interface is English, no translation needed
        if interface_lang == 'en':
            return response
        
        # If interface is Bangla, translate response
        if interface_lang == 'bn':
            return self.translate_english_to_bangla(response)
        
        return response
    
    def get_ui_text(self, key: str, lang: str = 'en') -> str:
        """
        Get UI text in specified language
        
        Args:
            key: Translation key
            lang: Language code ('en' or 'bn')
            
        Returns:
            str: Translated text
        """
        return self.translations.get(key, {}).get(lang, key)
    
    def _load_translations(self) -> dict:
        """Load all UI translations"""
        return {
            # Header texts
            'app_title': {'en': 'AI Tutor', 'bn': 'এআই টিউটর'},
            'chat_header': {'en': '💬 AI Tutor Chat', 'bn': '💬 এআই টিউটর চ্যাট'},
            'practice_header': {'en': '📝 Interactive Practice Mode', 'bn': '📝 ইন্টারঅ্যাক্টিভ অনুশীলন মোড'},
            'study_plan_header': {'en': '📅 Study Plan Generator', 'bn': '📅 অধ্যয়ন পরিকল্পনা জেনারেটর'},
            'progress_header': {'en': '📈 Progress Tracker', 'bn': '📈 অগ্রগতি ট্র্যাকার'},
            'logs_header': {'en': '🔧 System Logs & Debugging', 'bn': '🔧 সিস্টেম লগ এবং ডিবাগিং'},
            
            # Navigation
            'nav_chat': {'en': 'Chat', 'bn': 'চ্যাট'},
            'nav_practice': {'en': 'Practice', 'bn': 'অনুশীলন'},
            'nav_study_plan': {'en': 'Study Plan', 'bn': 'অধ্যয়ন পরিকল্পনা'},
            'nav_progress': {'en': 'Progress Tracker', 'bn': 'অগ্রগতি ট্র্যাকার'},
            'nav_logs': {'en': 'System Logs', 'bn': 'সিস্টেম লগ'},
            
            # PDF Selection
            'select_pdf': {'en': '📚 Select a PDF Document', 'bn': '📚 একটি পিডিএফ নথি চয়ন করুন'},
            'selection_method': {'en': 'How would you like to select a PDF?', 'bn': 'আপনি কীভাবে একটি পিডিএফ নির্বাচন করতে চান?'},
            'predefined_pdfs': {'en': 'Predefined PDFs', 'bn': 'পূর্বনির্ধারিত পিডিএফ'},
            'browse_files': {'en': 'Browse Files', 'bn': 'ফাইল ব্রাউজ করুন'},
            'enter_filename': {'en': 'Enter Filename', 'bn': 'ফাইলনাম লিখুন'},
            'active_pdf': {'en': '📄 Active PDF:', 'bn': '📄 সক্রিয় পিডিএফ:'},
            'choose_pdf': {'en': 'Choose PDF', 'bn': 'পিডিএফ চয়ন করুন'},
            'upload_tip': {'en': '💡 Tip: Upload or drag & drop your PDF file', 'bn': '💡 টিপ: আপনার পিডিএফ ফাইল আপলোড বা ড্র্যাগ করুন'},
            'choose_file': {'en': 'Choose a PDF file', 'bn': 'একটি পিডিএফ ফাইল চয়ন করুন'},
            'uploaded': {'en': '✅ Uploaded:', 'bn': '✅ আপলোড হয়েছে:'},
            'enter_pdf_filename': {'en': 'Enter PDF filename', 'bn': 'পিডিএফ ফাইলনাম লিখুন'},
            'search': {'en': '🔍 Search', 'bn': '🔍 অনুসন্ধান'},
            'found': {'en': '✅ Found:', 'bn': '✅ পাওয়া গেছে:'},
            'not_found': {'en': '❌ Could not find', 'bn': '❌ খুঁজে পাওয়া যায়নি'},
            'loading': {'en': '📚 Loading', 'bn': '📚 লোড হচ্ছে'},
            
            # Chat
            'chat_placeholder': {'en': 'Ask me anything...', 'bn': 'আমাকে যেকোনো কিছু জিজ্ঞাসা করুন...'},
            'select_pdf_first': {'en': '⚠️ Please select a PDF first!', 'bn': '⚠️ দয়া করে প্রথমে একটি পিডিএফ নির্বাচন করুন!'},
            'thinking': {'en': '🧠 Thinking...', 'bn': '🧠 চিন্তা করছি...'},
            'translating': {'en': '🌐 Translating...', 'bn': '🌐 অনুবাদ করা হচ্ছে...'},
            'error': {'en': '❌ Error:', 'bn': '❌ ত্রুটি:'},
            'loaded_pdf': {'en': "I've loaded", 'bn': 'আমি লোড করেছি'},
            'what_learn': {'en': 'What would you like to learn?', 'bn': 'আপনি কী শিখতে চান?'},
            
            # Practice/Quiz
            'quiz_config': {'en': '⚙️ Quiz Configuration', 'bn': '⚙️ কুইজ কনফিগারেশন'},
            'topic_practice': {'en': 'Topic to Practice', 'bn': 'অনুশীলন করার বিষয়'},
            'difficulty': {'en': 'Difficulty', 'bn': 'অসুবিধা স্তর'},
            'easy': {'en': 'Easy', 'bn': 'সহজ'},
            'medium': {'en': 'Medium', 'bn': 'মাধ্যম'},
            'hard': {'en': 'Hard', 'bn': 'কঠিন'},
            'num_questions': {'en': 'Number of Questions', 'bn': 'প্রশ্নের সংখ্যা'},
            'start_quiz': {'en': '🚀 Start Quiz', 'bn': '🚀 কুইজ শুরু করুন'},
            'enter_topic': {'en': 'Please enter a topic.', 'bn': 'দয়া করে একটি বিষয় লিখুন।'},
            'generating': {'en': '🧠 Generating', 'bn': '🧠 তৈরি করা হচ্ছে'},
            'questions_for': {'en': 'questions for', 'bn': 'এর জন্য প্রশ্ন'},
            'topic': {'en': 'Topic:', 'bn': 'বিষয়:'},
            'choose_answer': {'en': 'Choose Answer:', 'bn': 'উত্তর চয়ন করুন:'},
            'submit_quiz': {'en': '✅ Submit Quiz', 'bn': '✅ কুইজ জমা দিন'},
            'update_answers': {'en': '🔄 Update Answers', 'bn': '🔄 উত্তর আপডেট করুন'},
            'results': {'en': '📊 Results', 'bn': '📊 ফলাফল'},
            'your_answer': {'en': 'Your Answer:', 'bn': 'আপনার উত্তর:'},
            'correct_answer': {'en': 'Correct Answer:', 'bn': 'সঠিক উত্তর:'},
            'explanation': {'en': 'Explanation for', 'bn': 'এর ব্যাখ্যা'},
            'final_score': {'en': 'Final Score', 'bn': 'চূড়ান্ত স্কোর'},
            'great_job': {'en': '🌟 Great job! You know this topic well.', 'bn': '🌟 দারুণ! আপনি এই বিষয়টি ভালো জানেন।'},
            'good_effort': {'en': '👍 Good effort. Review the explanations above.', 'bn': '👍 ভালো প্রচেষ্টা। উপরের ব্যাখ্যাগুলি পর্যালোচনা করুন।'},
            'keep_studying': {'en': "💪 Keep studying. Try the 'Study Plan' tab for help.", 'bn': '💪 পড়াশোনা চালিয়ে যান। সাহায্যের জন্য "অধ্যয়ন পরিকল্পনা" ট্যাব চেষ্টা করুন।'},
            'start_new_quiz': {'en': '🔄 Start New Quiz', 'bn': '🔄 নতুন কুইজ শুরু করুন'},
            'no_pdf_quiz': {'en': '⚠️ Please load a PDF first to generate practice questions.', 'bn': '⚠️ অনুশীলন প্রশ্ন তৈরি করতে প্রথমে একটি পিডিএফ লোড করুন।'},
            
            # Language Toggle
            'switch_language': {'en': '🌐 Switch to Bangla', 'bn': '🌐 Switch to English'},
            'current_lang': {'en': 'Language: English', 'bn': 'ভাষা: বাংলা'},
        }

# Global instance
language_manager = LanguageManager()