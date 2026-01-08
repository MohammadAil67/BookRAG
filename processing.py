import re
from typing import List, Dict
from collections import Counter

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
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect text language"""
        return 'bn' if TextNormalizer.is_bengali(text) else 'en'


class PromptBuilder:
    SYSTEM_PROMPT = """You are an expert AI tutor.
    
RULES:
1. Use ONLY the provided Context to answer.
2. If the answer is not in the Context, say "I cannot find the answer in the provided text."
3. Cite your sources implicitly (e.g., "According to the text...").
4. Be detailed.
"""
    
    @staticmethod
    def build(query: str, chunks: List[str], history: List[Dict]) -> str:
        context_str = "\n---\n".join(chunks)
        history_str = ""
        if history:
            history_str = "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in history[-3:]])
        
        return f"""{PromptBuilder.SYSTEM_PROMPT}

Conversation History:
{history_str}

Context Information:
{context_str}

User Question: {query}

Answer:"""


class AdvancedQueryRefiner:
    """
    Tier 2 Query Refinement with Multi-Strategy Approach
    Handles pronouns, vague queries, and context dependencies
    Supports both English and Bengali
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.last_entities = []  # Track mentioned entities
        self.normalizer = TextNormalizer()
        
        # Pronouns by language
        self.pronouns = {
            'en': {
                'he', 'she', 'it', 'they', 'this', 'that', 'these', 'those',
                'his', 'her', 'its', 'their', 'him', 'them', 'we', 'us', 'our'
            },
            'bn': {
                'সে', 'তিনি', 'তারা', 'তাকে', 'তাদের', 'তার', 
                'এটি', 'ওটি', 'এগুলো', 'ওগুলো', 'সেটি', 'ওই',
                'আমরা', 'আমাদের', 'তোমরা', 'তোমাদের'
            }
        }
        
        # Vague command words
        self.vague_commands = {
            'en': [
                'elaborate', 'explain', 'tell', 'show', 'give',
                'do', 'what', 'how', 'describe', 'more', 'who', 'why'
            ],
            'bn': [
                'ব্যাখ্যা', 'বলুন', 'জানান', 'দেখান', 'দিন',
                'করুন', 'কী', 'কীভাবে', 'বর্ণনা', 'আরো', 'কে', 'কেন',
                'বুঝান', 'বিস্তারিত', 'সম্পর্কে'
            ]
        }
        
        # Follow-up indicators
        self.followup_words = {
            'en': {'also', 'too', 'additionally', 'furthermore', 'moreover', 'previous', 'earlier'},
            'bn': {'এছাড়াও', 'আরও', 'তাছাড়া', 'পূর্বে', 'আগে', 'ও', 'এবং'}
        }
        
        # Meta-question patterns
        self.meta_patterns = {
            'en': ["talking about", "referred to", "mentioned"],
            'bn': ["সম্পর্কে বলছেন", "উল্লেখ করা", "বলেছেন"]
        }
    
    def detect_language(self, text: str) -> str:
        """Detect query language"""
        return self.normalizer.detect_language(text)
    
    def refine(self, query: str, history: List[Dict], topic_context: str = "") -> str:
        """
        Multi-stage refinement:
        1. Quick heuristic checks
        2. Entity extraction from history
        3. LLM-based rewriting
        """
        
        # Stage 1: Check if refinement needed
        needs_refinement = self._needs_refinement(query, history)
        
        if not needs_refinement:
            return query
        
        # Stage 2: Extract context from history
        context_info = self._extract_context(history)
        
        # Stage 3: LLM-based rewriting with strong prompt
        refined = self._llm_rewrite(query, history, context_info, topic_context)
        
        # Stage 4: Validation
        if self._validate_refinement(refined, query):
            print(f"  ✨ Refined: '{query}' → '{refined}'")
            return refined
        
        return query
    
    def _needs_refinement(self, query: str, history: List[Dict]) -> bool:
        """Determine if query needs refinement (language-aware)"""
        
        if not history:
            return False
        
        lang = self.detect_language(query)
        query_normalized = self.normalizer.normalize(query)
        words = query_normalized.split()
        
        # Check 1: Contains pronouns
        has_pronouns = any(word in self.pronouns[lang] for word in words)
        
        # Check 2: Very short/vague
        is_short = len(words) <= 6
        
        # Check 3: Vague command-style start
        is_vague_command = any(query_normalized.startswith(cmd) for cmd in self.vague_commands[lang])
        
        # Check 4: Follow-up indicators
        is_followup = any(word in self.followup_words[lang] for word in words)
        
        # Check 5: Incomplete questions
        is_incomplete = query.endswith('?') and len(words) <= 4
        
        # Check 6: Meta-questions
        is_meta_question = any(pattern in query_normalized for pattern in self.meta_patterns[lang])
        
        # Refine if ANY condition is true
        return (has_pronouns or is_short or is_vague_command or 
                is_followup or is_incomplete or is_meta_question)
    
    def _extract_context(self, history: List[Dict]) -> Dict:
        """Extract key entities and topics from history (language-aware)"""
        
        context = {
            'entities': [],
            'topics': [],
            'last_subject': None
        }
        
        if not history:
            return context
        
        # Analyze last 3 turns
        recent = history[-3:]
        
        for entry in recent:
            user_text = entry['user']
            ai_text = entry['ai'][:500]  # Limit length
            
            lang = self.detect_language(user_text + ' ' + ai_text)
            
            if lang == 'en':
                # English: capitalized phrases
                entities = re.findall(
                    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                    user_text + ' ' + ai_text
                )
                context['entities'].extend(entities)
                
                # Extract key nouns (simple approach)
                words = ai_text.split()
                nouns = [w for w in words if len(w) > 4 and w[0].isupper()]
                context['topics'].extend(nouns)
            
            else:  # Bengali
                # Bengali: Extract longer words (4+ chars)
                words_bn = re.findall(r'[\u0980-\u09FF]{4,}', user_text + ' ' + ai_text)
                
                # Use frequency to identify likely entities/topics
                word_counts = Counter(words_bn)
                frequent_words = [word for word, count in word_counts.items() if count >= 2]
                
                context['entities'].extend(frequent_words)
                context['topics'].extend(frequent_words)
        
        # Deduplicate and get most common
        if context['entities']:
            entity_counts = Counter(context['entities'])
            context['last_subject'] = entity_counts.most_common(1)[0][0]
            context['entities'] = [e for e, _ in entity_counts.most_common(5)]
        
        return context
    
    def _llm_rewrite(self, query: str, history: List[Dict], 
                     context_info: Dict, topic_context: str = "") -> str:
        """Use LLM to rewrite query with strong instructions (multilingual)"""
        
        lang = self.detect_language(query)
        
        # Build history string (last 2 turns)
        hist_str = ""
        if history:
            recent = history[-2:]
            hist_str = "\n".join([
                f"User: {h['user']}\nAssistant: {h['ai'][:800]}..."
                for h in recent
            ])
        
        # Build context hints
        hints = ""
        if topic_context:
            if lang == 'en':
                hints += f"\nCurrent Topic Context: {topic_context}"
            else:
                hints += f"\nবর্তমান বিষয়: {topic_context}"
        
        if context_info['last_subject']:
            if lang == 'en':
                hints += f"\nMain subject: {context_info['last_subject']}"
            else:
                hints += f"\nমূল বিষয়: {context_info['last_subject']}"
        
        # Language-specific prompt
        if lang == 'en':
            prompt = f"""You are a query rewriting expert.

Conversation History:
{hist_str}
{hints}

Current Question: "{query}"

CRITICAL RULES:
1. **New Entities:** If the Current Question contains a Specific Name (e.g., "Zainul Abedin"), USE IT. Do NOT replace it with the name from history.
2. **Context:** Only add context if the question is vague (e.g., "what did he do?", "explain it").
3. **Pronouns:** Replace pronouns (he, she, it, they, we) with the specific name or subject they refer to.
4. **Clarity:** Make the question self-contained and clear.
5. **Brevity:** Keep it concise (1 sentence).

Output ONLY the rewritten question. No explanation.

Rewritten Question:"""
        
        else:  # Bengali
            prompt = f"""আপনি একটি প্রশ্ন পুনর্লিখন বিশেষজ্ঞ।

কথোপকথনের ইতিহাস:
{hist_str}
{hints}

বর্তমান প্রশ্ন: "{query}"

গুরুত্বপূর্ণ নিয়ম:
১. **নতুন নাম:** প্রশ্নে যদি নির্দিষ্ট নাম থাকে, তা ব্যবহার করুন।
২. **প্রসঙ্গ:** শুধুমাত্র অস্পষ্ট প্রশ্নে প্রসঙ্গ যোগ করুন (যেমন "তিনি কী করেছিলেন?")।
৩. **সর্বনাম:** সর্বনাম (সে, তিনি, এটি, তারা) প্রতিস্থাপন করুন নির্দিষ্ট নাম দিয়ে।
৪. **স্পষ্টতা:** প্রশ্নটি স্বয়ংসম্পূর্ণ এবং স্পষ্ট করুন।
৫. **সংক্ষিপ্ততা:** একটি বাক্যে রাখুন।

শুধুমাত্র পুনর্লিখিত প্রশ্ন লিখুন। ব্যাখ্যা নয়।

পুনর্লিখিত প্রশ্ন:"""
        
        try:
            response = self.llm.generate(prompt, temperature=0.3)
            
            # Clean response
            refined = response.strip()
            refined = refined.strip('"').strip("'").strip()
            
            # Remove common prefixes (language-aware)
            if lang == 'en':
                prefixes = ['rewritten question:', 'question:', 'refined:']
            else:
                prefixes = ['পুনর্লিখিত প্রশ্ন:', 'প্রশ্ন:', 'উত্তর:']
            
            for prefix in prefixes:
                if self.normalizer.normalize(refined).startswith(self.normalizer.normalize(prefix)):
                    refined = refined[len(prefix):].strip()
            
            return refined
            
        except Exception as e:
            print(f"  ⚠️ Refinement failed: {e}")
            return query
    
    def _validate_refinement(self, refined: str, original: str) -> bool:
        """Validate that refinement is reasonable (language-aware)"""
        
        lang = self.detect_language(original)
        
        # Check 1: Not too long
        if len(refined.split()) > 30:
            return False
        
        # Check 2: Not too short
        if len(refined.split()) < 3:
            return False
        
        # Check 3: Actually different
        if self.normalizer.normalize(refined) == self.normalizer.normalize(original):
            return False
        
        # Check 4: Preserve question format
        if original.endswith('?') and not refined.endswith('?'):
            # Define question words by language
            if lang == 'en':
                question_words = ['what', 'when', 'where', 'who', 'how', 'why', 'which']
            else:
                question_words = ['কী', 'কে', 'কোথায়', 'কখন', 'কীভাবে', 'কেন', 'কোন']
            
            refined_normalized = self.normalizer.normalize(refined)
            if not any(refined_normalized.startswith(qw) for qw in question_words):
                return False
        
        return True


class MultiQueryGenerator:
    """
    Tier 2 Enhancement: Generate multiple query variants
    Improves recall by searching with different phrasings
    Supports both English and Bengali
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.normalizer = TextNormalizer()
    
    def detect_language(self, text: str) -> str:
        """Detect query language"""
        return self.normalizer.detect_language(text)
    
    def generate_variants(self, query: str, num_variants: int = 2) -> List[str]:
        """
        Generate alternative query phrasings (multilingual)
        Returns: [original_query, variant1, variant2, ...]
        """
        
        # Quick check: don't generate variants for very simple queries
        if len(query.split()) <= 3:
            return [query]
        
        lang = self.detect_language(query)
        
        if lang == 'en':
            prompt = f"""Given this question: "{query}"

Generate {num_variants} alternative ways to ask the same question that might retrieve different relevant information.

RULES:
1. Use different vocabulary (synonyms)
2. Rephrase the structure
3. Keep the core meaning identical
4. Each variant should be 1 sentence
5. Be specific, not vague

EXAMPLES:
Question: "What were Zahir Raihan's contributions to cinema?"
Variant 1: "What films did Zahir Raihan create and how did they impact Bangladesh?"
Variant 2: "How did Zahir Raihan influence the film industry in Bangladesh?"

Question: "When did Zahir Raihan die?"
Variant 1: "What happened to Zahir Raihan and when?"
Variant 2: "What was the date of Zahir Raihan's death or disappearance?"

Now generate {num_variants} variants for: "{query}"

Respond ONLY with the variants, one per line, numbered:
1. [variant 1]
2. [variant 2]"""
        
        else:  # Bengali
            prompt = f"""এই প্রশ্নের জন্য: "{query}"

{num_variants}টি বিকল্প উপায়ে একই প্রশ্ন তৈরি করুন যা বিভিন্ন প্রাসঙ্গিক তথ্য পুনরুদ্ধার করতে পারে।

নিয়ম:
১. ভিন্ন শব্দভাণ্ডার ব্যবহার করুন (প্রতিশব্দ)
২. গঠন পরিবর্তন করুন
৩. মূল অর্থ অভিন্ন রাখুন
৪. প্রতিটি বৈকল্পিক ১ বাক্য হতে হবে
৫. নির্দিষ্ট হন, অস্পষ্ট নয়

উদাহরণ:
প্রশ্ন: "জহির রায়হানের চলচ্চিত্রে অবদান কী ছিল?"
বৈকল্পিক ১: "জহির রায়হান কোন চলচ্চিত্র তৈরি করেছেন এবং তারা বাংলাদেশে কীভাবে প্রভাব ফেলেছে?"
বৈকল্পিক ২: "জহির রায়হান চলচ্চিত্র শিল্পে কীভাবে প্রভাব ফেলেছেন?"

এখন "{query}" এর জন্য {num_variants}টি বৈকল্পিক তৈরি করুন।

শুধুমাত্র বৈকল্পিকগুলো লিখুন, প্রতিটি আলাদা লাইনে, সংখ্যা দিয়ে:
১. [বৈকল্পিক ১]
২. [বৈকল্পিক ২]"""
        
        try:
            response = self.llm.generate(prompt, temperature=0.7)
            
            # Parse variants
            variants = self._parse_variants(response, lang)
            
            # Validate variants
            valid_variants = [v for v in variants if self._validate_variant(v, query)]
            
            # Return original + valid variants
            return [query] + valid_variants[:num_variants]
            
        except Exception as e:
            print(f"  ⚠️ Variant generation failed: {e}")
            return [query]
    
    def _parse_variants(self, response: str, lang: str) -> List[str]:
        """Parse LLM response into list of variants (language-aware)"""
        
        variants = []
        
        # Try numbered format
        if lang == 'en':
            numbered = re.findall(r'\d+\.\s*(.+?)(?:\n|$)', response)
        else:  # Bengali - support Bengali numerals too
            numbered = re.findall(r'[\d০-৯]+\.\s*(.+?)(?:\n|$)', response)
        
        if numbered:
            variants.extend(numbered)
        
        # Try line-by-line fallback
        if not variants:
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            variants.extend(lines)
        
        # Clean up
        variants = [v.strip().strip('"').strip("'") for v in variants]
        variants = [v for v in variants if len(v) > 10]  # Filter too short
        
        return variants
    
    def _validate_variant(self, variant: str, original: str) -> bool:
        """Check if variant is valid (language-aware)"""
        
        # Not too long
        if len(variant.split()) > 30:
            return False
        
        # Not identical to original
        if self.normalizer.normalize(variant) == self.normalizer.normalize(original):
            return False
        
        # Contains some overlap with original (not completely different)
        orig_words = set(self.normalizer.normalize(original).split())
        var_words = set(self.normalizer.normalize(variant).split())
        
        # Extract content words (> 3 chars)
        content_words_orig = {w for w in orig_words if len(w) > 3}
        content_words_var = {w for w in var_words if len(w) > 3}
        
        if content_words_orig and content_words_var:
            overlap = len(content_words_orig & content_words_var)
            ratio = overlap / len(content_words_orig)
            
            if ratio < 0.2:  # Too different
                return False
        
        return True