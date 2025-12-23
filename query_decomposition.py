import re
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass

@dataclass
class DecompositionResult:
    """Result of query decomposition"""
    original_query: str
    sub_queries: List[str]
    is_decomposed: bool
    decomposition_type: str  # 'comparison', 'multi_part', 'analytical', 'none'
    language: str  # 'en' or 'bn'

class TextNormalizer:
    """Language-aware text normalization utilities"""
    
    @staticmethod
    def is_bengali(text: str) -> bool:
        """Check if text contains Bengali characters"""
        bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        return bengali_chars > len(text) * 0.2
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text based on detected language"""
        if TextNormalizer.is_bengali(text):
            return text  # No case conversion for Bengali
        return text.lower()  # Lowercase for English

class QueryDecomposer:
    """
    Intelligent query decomposition system
    Breaks complex queries into simpler sub-queries for better retrieval
    Supports both English and Bengali queries
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.normalizer = TextNormalizer()
        
        # English keywords
        self.comparison_keywords = {
            'en': [
                'compare', 'contrast', 'difference', 'distinguish',
                'similar', 'alike', 'versus', 'vs', 'vs.', 
                'both', 'relationship between', 'connection between'
            ],
            'bn': [
                'তুলনা', 'পার্থক্য', 'তুলনা করুন', 'পার্থক্য কী',
                'মিল', 'সাদৃশ্য', 'বনাম', 'মধ্যে পার্থক্য',
                'সম্পর্ক', 'উভয়', 'দুটি', 'মধ্যে সম্পর্ক',
                'সংযোগ', 'ভিন্নতা', 'একইরকম'
            ]
        }
        
        self.analytical_keywords = {
            'en': [
                'analyze', 'evaluate', 'assess', 'discuss',
                'examine', 'explore', 'investigate', 'explain'
            ],
            'bn': [
                'বিশ্লেষণ', 'মূল্যায়ন', 'আলোচনা', 'ব্যাখ্যা',
                'পরীক্ষা', 'অনুসন্ধান', 'জানুন', 'বুঝুন',
                'বর্ণনা করুন', 'ব্যাখ্যা করুন'
            ]
        }
        
        # Question words for validation
        self.question_words = {
            'en': ['what', 'how', 'why', 'when', 'where', 'who', 'which'],
            'bn': ['কী', 'কে', 'কোথায়', 'কখন', 'কীভাবে', 'কেন', 'কোন', 'কাকে']
        }
        
        # Conjunction patterns
        self.conjunctions = {
            'en': [' and ', ' or '],
            'bn': [' এবং ', ' ও ', ' আর ', ' অথবা ', ' কিংবা ']
        }
    
    def detect_language(self, text: str) -> str:
        """Detect query language"""
        return 'bn' if self.normalizer.is_bengali(text) else 'en'
    
    def should_decompose(self, query: str) -> Tuple[bool, str]:
        """
        Determine if query needs decomposition
        Returns: (should_decompose: bool, decomposition_type: str)
        """
        lang = self.detect_language(query)
        query_normalized = self.normalizer.normalize(query)
        
        # Check for comparison keywords
        for keyword in self.comparison_keywords[lang]:
            if keyword in query_normalized:
                return True, 'comparison'
        
        # Check for analytical keywords
        for keyword in self.analytical_keywords[lang]:
            if keyword in query_normalized:
                return True, 'analytical'
        
        # Check if it's a multi-part query
        if self._is_multipart_query(query, lang):
            return True, 'multi_part'
        
        # Check for multiple explicit questions
        if self._has_multiple_questions(query, lang):
            return True, 'multiple_questions'
        
        return False, 'none'
    
    def _is_multipart_query(self, query: str, lang: str) -> bool:
        """Check if query has multiple parts connected by conjunctions"""
        query_normalized = self.normalizer.normalize(query)
        
        # Must have conjunction
        has_conjunction = any(conj in query_normalized for conj in self.conjunctions[lang])
        if not has_conjunction:
            return False
        
        # Check for multiple question indicators
        if lang == 'en':
            patterns = [
                r'(what|how|why|when|where|who).+(and|or).+(what|how|why|when|where|who)',
                r'(explain|describe|tell).+and.+(explain|describe|tell)',
            ]
            return any(re.search(p, query_normalized, re.IGNORECASE) for p in patterns)
        else:  # Bengali
            patterns = [
                r'(কী|কে|কীভাবে|কেন|কখন|কোথায়).+(এবং|ও|আর).+(কী|কে|কীভাবে|কেন|কখন|কোথায়)',
                r'(ব্যাখ্যা|বর্ণনা|বলুন).+(এবং|ও|আর).+(ব্যাখ্যা|বর্ণনা|বলুন)',
            ]
            return any(re.search(p, query) for p in patterns)
    
    def _has_multiple_questions(self, query: str, lang: str) -> bool:
        """Check if query contains multiple explicit questions"""
        if lang == 'en':
            # Count question marks
            return query.count('?') > 1
        else:  # Bengali
            # Bengali can use ? or just question words
            question_count = query.count('?')
            if question_count > 1:
                return True
            
            # Count question word occurrences
            q_word_count = sum(1 for qw in self.question_words['bn'] if qw in query)
            return q_word_count > 2  # Multiple question words suggest multiple questions
    
    def decompose(self, query: str) -> DecompositionResult:
        """
        Main decomposition method
        Returns DecompositionResult with sub-queries
        """
        lang = self.detect_language(query)
        should_decompose, decomp_type = self.should_decompose(query)
        
        if not should_decompose:
            return DecompositionResult(
                original_query=query,
                sub_queries=[query],
                is_decomposed=False,
                decomposition_type='none',
                language=lang
            )
        
        # Try rule-based decomposition first
        sub_queries = self._rule_based_decompose(query, decomp_type, lang)
        
        # Fallback to LLM if rule-based fails
        if not sub_queries or len(sub_queries) < 2:
            sub_queries = self._llm_decompose(query, lang)
        
        # If still no good decomposition, return original
        if not sub_queries or len(sub_queries) < 2:
            return DecompositionResult(
                original_query=query,
                sub_queries=[query],
                is_decomposed=False,
                decomposition_type='none',
                language=lang
            )
        
        return DecompositionResult(
            original_query=query,
            sub_queries=sub_queries,
            is_decomposed=True,
            decomposition_type=decomp_type,
            language=lang
        )
    
    def _rule_based_decompose(self, query: str, decomp_type: str, lang: str) -> List[str]:
        """Rule-based decomposition for common patterns"""
        
        if decomp_type == 'comparison':
            return self._decompose_comparison(query, lang)
        elif decomp_type == 'multi_part':
            return self._decompose_multipart(query, lang)
        elif decomp_type == 'multiple_questions':
            return self._decompose_multiple_questions(query, lang)
        
        return []
    
    def _decompose_comparison(self, query: str, lang: str) -> List[str]:
        """Decompose comparison queries"""
        if lang == 'en':
            # English patterns
            patterns = [
                (r'compare\s+(.+?)\s+and\s+(.+?)(?:\?|$)', 
                 "What is {0}?", "What is {1}?", "What are the differences between {0} and {1}?"),
                (r'difference between\s+(.+?)\s+and\s+(.+?)(?:\?|$)',
                 "What is {0}?", "What is {1}?", "How do {0} and {1} differ?"),
            ]
            
            for pattern, *templates in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    entities = match.groups()
                    return [t.format(*entities) for t in templates]
        
        else:  # Bengali
            patterns = [
                (r'(.+?)\s+এবং\s+(.+?)\s+তুলনা',
                 "{0} কী?", "{1} কী?", "{0} এবং {1} এর মধ্যে পার্থক্য কী?"),
                (r'(.+?)\s+ও\s+(.+?)\s+মধ্যে পার্থক্য',
                 "{0} কী?", "{1} কী?", "{0} এবং {1} কীভাবে আলাদা?"),
            ]
            
            for pattern, *templates in patterns:
                match = re.search(pattern, query)
                if match:
                    entities = match.groups()
                    return [t.format(*entities) for t in templates]
        
        return []
    
    def _decompose_multipart(self, query: str, lang: str) -> List[str]:
        """Decompose multi-part questions"""
        if lang == 'en':
            # Split on 'and' while preserving question structure
            parts = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)
        else:  # Bengali
            # Split on Bengali conjunctions
            for conj in self.conjunctions['bn']:
                if conj in query:
                    parts = query.split(conj)
                    break
            else:
                return []
        
        if len(parts) < 2:
            return []
        
        sub_queries = []
        for part in parts:
            part = part.strip()
            
            # Ensure it ends with question mark
            if not part.endswith('?'):
                part += '?'
            
            # Capitalize first letter (for English)
            if lang == 'en':
                part = part[0].upper() + part[1:]
            
            sub_queries.append(part)
        
        return sub_queries
    
    def _decompose_multiple_questions(self, query: str, lang: str) -> List[str]:
        """Split multiple explicit questions"""
        # Split by '?' and clean up
        questions = [q.strip() + '?' for q in query.split('?') if q.strip()]
        
        return questions if len(questions) > 1 else []
    
    def _llm_decompose(self, query: str, lang: str) -> List[str]:
        """Use LLM for complex decomposition"""
        
        if lang == 'en':
            prompt = f"""Break down this complex question into simpler sub-questions.

Input: "Compare photosynthesis and cellular respiration"
Output:
1. What is photosynthesis and how does it work?
2. What is cellular respiration and how does it work?
3. What are the similarities between photosynthesis and cellular respiration?
4. What are the differences between photosynthesis and cellular respiration?

Now decompose this question:
Original question: {query}

Sub-questions:"""
        
        else:  # Bengali
            prompt = f"""এই জটিল প্রশ্নটি সরল উপ-প্রশ্নে ভাগ করুন।

উদাহরণ :
Input : "সালোকসংশ্লেষণ এবং কোষীয় শ্বসনের তুলনা করুন"
Output:
১. সালোকসংশ্লেষণ কী এবং এটি কীভাবে কাজ করে?
২. কোষীয় শ্বসন কী এবং এটি কীভাবে কাজ করে?
৩. সালোকসংশ্লেষণ এবং কোষীয় শ্বসনের মধ্যে মিল কী?
৪. সালোকসংশ্লেষণ এবং কোষীয় শ্বসনের মধ্যে পার্থক্য কী?

এখন এই প্রশ্নটি বিভাজন করুন:
মূল প্রশ্ন: {query}

উপ-প্রশ্ন:"""
        
        try:
            response = self.llm.generate(prompt, temperature=0.3)
            
            # Parse numbered list
            sub_queries = []
            for line in response.strip().split('\n'):
                # Remove numbering: "1. " or "১. " (Bengali numbers)
                clean = re.sub(r'^[\d০-৯]+[\.):\s]+', '', line.strip())
                
                # Validate: must be substantial question
                if clean and len(clean) > 15:
                    # Check if it contains question words or ends with ?
                    has_q_word = any(qw in clean for qw in self.question_words[lang])
                    
                    if '?' in clean or has_q_word:
                        # Ensure ends with ?
                        if not clean.endswith('?'):
                            clean += '?'
                        
                        sub_queries.append(clean)
            
            # Limit to 4 sub-queries
            return sub_queries[:4] if sub_queries else []
        
        except Exception as e:
            print(f"  ⚠️ LLM decomposition failed: {e}")
            return []


class SmartContextApplier:
    """
    Intelligently applies topic context to queries
    Works alongside query decomposition for simple follow-up queries
    Supports both English and Bengali
    """
    
    def __init__(self, topic_tracker=None):
        self.topic_tracker = topic_tracker
        self.normalizer = TextNormalizer()
        
        # Pronouns to detect
        self.pronouns = {
            'en': ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'their'],
            'bn': ['সে', 'তিনি', 'তারা', 'তাকে', 'তাদের', 'তার', 'এটি', 'ওটি']
        }
        
        # Vague terms
        self.vague_terms = {
            'en': ['that', 'this', 'those', 'these', 'it'],
            'bn': ['এটি', 'ওটি', 'এগুলো', 'ওগুলো', 'সেটি']
        }
    
    def detect_language(self, text: str) -> str:
        """Detect query language"""
        return 'bn' if self.normalizer.is_bengali(text) else 'en'
    
    def should_apply_context(self, query: str, history: List[Dict]) -> bool:
        """Determine if context should be applied"""
        if not history or not self.topic_tracker:
            return False
        
        lang = self.detect_language(query)
        query_normalized = self.normalizer.normalize(query)
        
        # Check for pronouns
        has_pronoun = any(pron in query_normalized.split() for pron in self.pronouns[lang])
        
        # Check for vague terms
        has_vague = any(term in query_normalized.split() for term in self.vague_terms[lang])
        
        # Short query check
        is_short = len(query.split()) < 8
        
        return (has_pronoun or has_vague) and is_short
    
    def apply_context(self, sub_query: str, history: List[Dict], topic_info: str) -> str:
        """Apply topic context to vague queries"""
        if not topic_info or not self.should_apply_context(sub_query, history):
            return sub_query
        
        lang = self.detect_language(sub_query)
        
        if lang == 'en':
            # English context application
            return f"{sub_query} (about {topic_info})"
        else:  # Bengali
            # Bengali context application
            return f"{sub_query} ({topic_info} সম্পর্কে)"
    
    def get_smart_context(self, query: str) -> str:
        """Get current topic context"""
        if not self.topic_tracker:
            return ""
        
        current_topic = self.topic_tracker.get_current_topic()
        if current_topic:
            return current_topic.name
        
        return ""