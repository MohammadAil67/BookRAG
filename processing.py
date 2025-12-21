import re
from typing import List, Tuple, Dict
from collections import Counter
from models import GroqLLM

class PromptBuilder:
    SYSTEM_PROMPT = """You are an expert AI tutor.
    
RULES:
1. Use ONLY the provided Context to answer.
2. If the answer is not in the Context, say "I cannot find the answer in the provided text."
3. Cite your sources implicitly (e.g., "According to the text...").
4. Be concise but comprehensive.
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
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.last_entities = []  # Track mentioned entities
        
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
        """Determine if query needs refinement"""
        
        if not history:
            return False
        
        query_lower = query.lower()
        words = query_lower.split()
        
        # Check 1: Contains pronouns (ADDED 'we', 'us', 'our')
        pronouns = {'he', 'she', 'it', 'they', 'this', 'that', 'these', 'those', 
                   'his', 'her', 'its', 'their', 'him', 'them', 'we', 'us', 'our'}
        has_pronouns = any(word in pronouns for word in words)
        
        # Check 2: Very short/vague (INCREASED to <= 6)
        is_short = len(words) <= 6
        
        # Check 3: Command-style (ADDED 'who', 'why')
        vague_commands = ['elaborate', 'explain', 'tell', 'show', 'give', 
                         'do', 'what', 'how', 'describe', 'more', 'who', 'why']
        is_vague_command = any(query_lower.startswith(cmd) for cmd in vague_commands)
        
        # Check 4: Follow-up indicators
        followup_words = {'also', 'too', 'additionally', 'furthermore', 'moreover', 'previous', 'earlier'}
        is_followup = any(word in followup_words for word in words)
        
        # Check 5: Incomplete questions
        is_incomplete = query.endswith('?') and len(words) <= 4
        
        # Check 6: "Talking about" patterns (NEW)
        is_meta_question = "talking about" in query_lower or "referred to" in query_lower
        
        # Refine if ANY condition is true
        return (has_pronouns or is_short or is_vague_command or 
                is_followup or is_incomplete or is_meta_question)
    
    def _extract_context(self, history: List[Dict]) -> Dict:
        """Extract key entities and topics from history"""
        
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
            # Extract capitalized phrases (named entities)
            user_text = entry['user']
            ai_text = entry['ai'][:500]  # Limit length
            
            # Find proper nouns and multi-word names
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', 
                                user_text + ' ' + ai_text)
            context['entities'].extend(entities)
            
            # Extract key nouns (simple approach)
            words = ai_text.split()
            nouns = [w for w in words if len(w) > 4 and w[0].isupper()]
            context['topics'].extend(nouns)
        
        # Deduplicate and get most common
        if context['entities']:
            entity_counts = Counter(context['entities'])
            context['last_subject'] = entity_counts.most_common(1)[0][0]
            context['entities'] = [e for e, _ in entity_counts.most_common(5)]
        
        return context
    
    def _llm_rewrite(self, query: str, history: List[Dict], 
                     context_info: Dict, topic_context: str = "") -> str:
        """Use LLM to rewrite query with strong instructions"""
        
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
            hints += f"\nCurrent Topic Context: {topic_context}"
        if context_info['last_subject']:
            hints += f"\nMain subject: {context_info['last_subject']}"
        if context_info['entities']:
            hints += f"\nMentioned: {', '.join(context_info['entities'][:3])}"
        
        # Inside processing.py -> AdvancedQueryRefiner -> _llm_rewrite

        prompt = f"""You are a query rewriting expert.

Conversation History:
{hist_str}
{hints}

Current Question: "{query}"

CRITICAL RULES:
1. **New Entities:** If the Current Question contains a Specific Name (e.g., "Zainul Abedin"), USE IT. Do NOT replace it with the name from history.
2. **Context:** Only add context if the question is vague (e.g., "what did he do?", "explain it").
3. **Pronouns:** Replace pronouns (he, she, it, they, we) with the specific name or subject they refer to.
4. **Meta-Questions:** If the user asks "who are we talking about?", rewrite it to clarify the previous subject (e.g., "Who is the subject of the previous discussion?").

EXAMPLES:
- History: [Zahir Raihan] | Question: "when did he die" -> "when did Zahir Raihan die"
- History: [Zahir Raihan] | Question: "Explain Zainul Abedin" -> "Explain Zainul Abedin" (New Name Kept)
- History: [Comparison]   | Question: "who are we talking about" -> "Who were the subjects of the previous comparison?"

Rewritten Question (respond with ONLY the refined question text):"""

        try:
            response = self.llm.generate(prompt, temperature=0.2)
            
            # Clean response
            refined = response.strip()
            refined = refined.strip('"').strip("'").strip()
            
            # Remove common prefixes
            prefixes = ['rewritten question:', 'question:', 'refined:']
            for prefix in prefixes:
                if refined.lower().startswith(prefix):
                    refined = refined[len(prefix):].strip()
            
            return refined
            
        except Exception as e:
            print(f"  ⚠️ Refinement failed: {e}")
            return query
    
    def _validate_refinement(self, refined: str, original: str) -> bool:
        """Validate that refinement is reasonable"""
        
        # Check 1: Not too long
        if len(refined.split()) > 30:
            return False
        
        # Check 2: Not too short
        if len(refined.split()) < 3:
            return False
        
        # Check 3: Actually different
        if refined.lower() == original.lower():
            return False
        
        # Check 4: Contains question words (for questions)
        if original.endswith('?') and not refined.endswith('?'):
            # If original was a question, refined should be too
            question_words = ['what', 'when', 'where', 'who', 'how', 'why', 'which']
            if not any(refined.lower().startswith(qw) for qw in question_words):
                return False
        
        return True
        
class MultiQueryGenerator:
    """
    Tier 2 Enhancement: Generate multiple query variants
    Improves recall by searching with different phrasings
    """
    
    def __init__(self, llm):
        self.llm = llm
        
    def generate_variants(self, query: str, num_variants: int = 2) -> List[str]:
        """
        Generate alternative query phrasings
        Returns: [original_query, variant1, variant2, ...]
        """
        
        # Quick check: don't generate variants for very simple queries
        if len(query.split()) <= 3:
            return [query]
        
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

        try:
            response = self.llm.generate(prompt, temperature=0.7)
            
            # Parse variants
            variants = self._parse_variants(response)
            
            # Validate variants
            valid_variants = [v for v in variants if self._validate_variant(v, query)]
            
            # Return original + valid variants
            return [query] + valid_variants[:num_variants]
            
        except Exception as e:
            print(f"  ⚠️ Variant generation failed: {e}")
            return [query]
    
    def _parse_variants(self, response: str) -> List[str]:
        """Parse LLM response into list of variants"""
        
        variants = []
        
        # Try numbered format first: "1. variant"
        numbered = re.findall(r'\d+\.\s*(.+?)(?:\n|$)', response)
        if numbered:
            variants.extend(numbered)
        
        # Try line-by-line
        if not variants:
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            variants.extend(lines)
        
        # Clean up
        variants = [v.strip().strip('"').strip("'") for v in variants]
        variants = [v for v in variants if len(v) > 10]  # Filter too short
        
        return variants
    
    def _validate_variant(self, variant: str, original: str) -> bool:
        """Check if variant is valid"""
        
        # Not too long
        if len(variant.split()) > 30:
            return False
        
        # Not identical to original
        if variant.lower() == original.lower():
            return False
        
        # Contains some overlap with original (not completely different)
        orig_words = set(original.lower().split())
        var_words = set(variant.lower().split())
        
        # Should share at least 20% of content words
        content_words_orig = {w for w in orig_words if len(w) > 3}
        content_words_var = {w for w in var_words if len(w) > 3}
        
        if content_words_orig and content_words_var:
            overlap = len(content_words_orig & content_words_var)
            ratio = overlap / len(content_words_orig)
            
            if ratio < 0.2:  # Too different
                return False
        
        return True