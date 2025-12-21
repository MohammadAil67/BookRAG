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

class QueryDecomposer:
    """
    Intelligent query decomposition system
    Breaks complex queries into simpler sub-queries for better retrieval
    """
    
    def __init__(self, llm):
        self.llm = llm
        
        # Patterns for different decomposition types
        self.comparison_keywords = [
            'compare', 'contrast', 'difference', 'distinguish',
            'similar', 'alike', 'versus', 'vs', 'vs.', 
            'both', 'relationship between', 'connection between'
        ]
        
        self.analytical_keywords = [
            'analyze', 'evaluate', 'assess', 'discuss',
            'examine', 'explore', 'investigate'
        ]
    
    def should_decompose(self, query: str) -> Tuple[bool, str]:
        """
        Determine if query needs decomposition
        Returns: (should_decompose: bool, decomposition_type: str)
        """
        query_lower = query.lower()
        
        # Type 1: Comparison queries
        if self._is_comparison_query(query_lower):
            return True, 'comparison'
        
        # Type 2: Multi-part questions (connected by 'and'/'or')
        if self._is_multipart_query(query):
            return True, 'multi_part'
        
        # Type 3: Analytical queries
        if self._is_analytical_query(query_lower):
            return True, 'analytical'
        
        # Type 4: Multiple explicit questions
        if query.count('?') > 1:
            return True, 'multi_question'
        
        return False, 'none'
    
    def _is_comparison_query(self, query_lower: str) -> bool:
        """Check if query is asking for comparison"""
        return any(keyword in query_lower for keyword in self.comparison_keywords)
    
    def _is_multipart_query(self, query: str) -> bool:
        """Check if query has multiple parts connected by conjunctions"""
        # Look for patterns like "What is X and what is Y?"
        # or "Tell me about X and Y"
        
        # Must have 'and' or 'or'
        if ' and ' not in query.lower() and ' or ' not in query.lower():
            return False
        
        # Check if it's a substantial multi-part question
        # (not just "X and Y" as a compound subject)
        patterns = [
            r'(what|how|why|when|where|who).+(and|or).+(what|how|why|when|where|who)',
            r'(explain|describe|tell).+and.+(explain|describe|tell)',
        ]
        
        return any(re.search(pattern, query.lower()) for pattern in patterns)
    
    def _is_analytical_query(self, query_lower: str) -> bool:
        """Check if query requires analytical breakdown"""
        return any(keyword in query_lower for keyword in self.analytical_keywords)
    
    def decompose(self, query: str) -> DecompositionResult:
        """
        Main decomposition method
        Returns DecompositionResult with sub-queries
        """
        should_decomp, decomp_type = self.should_decompose(query)
        
        if not should_decomp:
            return DecompositionResult(
                original_query=query,
                sub_queries=[query],
                is_decomposed=False,
                decomposition_type='none'
            )
        
        # Try rule-based decomposition first (faster)
        sub_queries = self._rule_based_decompose(query, decomp_type)
        
        # If rule-based fails or produces poor results, use LLM
        if not sub_queries or len(sub_queries) == 1:
            sub_queries = self._llm_decompose(query)
        
        return DecompositionResult(
            original_query=query,
            sub_queries=sub_queries if sub_queries else [query],
            is_decomposed=len(sub_queries) > 1,
            decomposition_type=decomp_type
        )
    
    def _rule_based_decompose(self, query: str, decomp_type: str) -> List[str]:
        """Rule-based decomposition for common patterns"""
        
        if decomp_type == 'comparison':
            return self._decompose_comparison(query)
        
        elif decomp_type == 'multi_part':
            return self._decompose_multipart(query)
        
        elif decomp_type == 'multi_question':
            return self._decompose_multiple_questions(query)
        
        return []
    
    def _decompose_comparison(self, query: str) -> List[str]:
        """Decompose comparison queries"""
        
        query_lower = query.lower()
        
        # Pattern: "Compare X and Y"
        compare_pattern = r'compare\s+(.+?)\s+(?:and|with|to)\s+(.+?)(?:\s+in terms of\s+(.+?))?(?:\?|$)'
        match = re.search(compare_pattern, query_lower)
        
        if match:
            entity1 = match.group(1).strip()
            entity2 = match.group(2).strip()
            aspect = match.group(3).strip() if match.group(3) else None
            
            # Clean up trailing punctuation
            entity2 = re.sub(r'[?.!]+$', '', entity2).strip()
            
            sub_queries = [
                f"What is {entity1}?",
                f"What are the key characteristics of {entity1}?",
                f"What is {entity2}?",
                f"What are the key characteristics of {entity2}?",
            ]
            
            if aspect:
                sub_queries.append(f"How does {entity1} relate to {aspect}?")
                sub_queries.append(f"How does {entity2} relate to {aspect}?")
            
            return sub_queries
        
        # Pattern: "Difference between X and Y"
        diff_pattern = r'(?:difference|distinguish|contrast).*between\s+(.+?)\s+and\s+(.+?)(?:\?|$)'
        match = re.search(diff_pattern, query_lower)
        
        if match:
            entity1 = match.group(1).strip()
            entity2 = match.group(2).strip()
            entity2 = re.sub(r'[?.!]+$', '', entity2).strip()
            
            return [
                f"What is {entity1}?",
                f"What is {entity2}?",
                f"What are the key differences between {entity1} and {entity2}?"
            ]
        
        # Pattern: "Similarities between X and Y" or "How are X and Y similar?"
        sim_pattern = r'(?:similar|alike|resemble).*(?:between)?\s+(.+?)\s+and\s+(.+?)(?:\?|$)'
        match = re.search(sim_pattern, query_lower)
        
        if match:
            entity1 = match.group(1).strip()
            entity2 = match.group(2).strip()
            entity2 = re.sub(r'[?.!]+$', '', entity2).strip()
            
            return [
                f"What is {entity1}?",
                f"What is {entity2}?",
                f"What similarities exist between {entity1} and {entity2}?"
            ]
        
        return []
    
    def _decompose_multipart(self, query: str) -> List[str]:
        """Decompose multi-part questions"""
        
        # Split on 'and' while preserving question structure
        # Example: "What is X and how does Y work?" → ["What is X?", "How does Y work?"]
        
        parts = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)
        
        if len(parts) < 2:
            return []
        
        sub_queries = []
        for part in parts:
            part = part.strip()
            
            # Ensure it ends with question mark
            if not part.endswith('?'):
                part += '?'
            
            # Capitalize first letter
            part = part[0].upper() + part[1:]
            
            sub_queries.append(part)
        
        return sub_queries
    
    def _decompose_multiple_questions(self, query: str) -> List[str]:
        """Split multiple explicit questions"""
        
        # Split by '?' and clean up
        questions = [q.strip() + '?' for q in query.split('?') if q.strip()]
        
        return questions if len(questions) > 1 else []
    
    def _llm_decompose(self, query: str) -> List[str]:
        """Use LLM for complex decomposition"""
        
        prompt = f"""Break down this complex question into 2-4 simpler, independent sub-questions.
Each sub-question should be clear, specific, and answerable on its own.

Rules:
1. Each sub-question should focus on one aspect
2. Sub-questions should be self-contained (no pronouns like "it", "they")
3. Maintain the original intent of the question
4. Return ONLY the numbered sub-questions, no explanations

Examples:

Input: "Compare photosynthesis and cellular respiration"
Output:
1. What is photosynthesis and how does it work?
2. What is cellular respiration and how does it work?
3. What are the similarities between photosynthesis and cellular respiration?
4. What are the differences between photosynthesis and cellular respiration?

Input: "How did the Liberation War affect Bangladesh's culture and economy?"
Output:
1. What was the Liberation War of Bangladesh?
2. How did the Liberation War impact Bangladesh's culture?
3. How did the Liberation War affect Bangladesh's economy?

Now decompose this question:
Original question: {query}

Sub-questions:"""

        try:
            response = self.llm.generate(prompt, max_tokens=250)
            
            # Parse numbered list
            sub_queries = []
            for line in response.strip().split('\n'):
                # Remove numbering: "1. " or "1) " or "1: "
                clean = re.sub(r'^\d+[\.):\s]+', '', line.strip())
                
                # Validate: must be substantial question
                if clean and len(clean) > 15 and ('?' in clean or 
                    any(qw in clean.lower() for qw in ['what', 'how', 'why', 'when', 'where', 'who'])):
                    
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
    """
    
    def __init__(self, topic_tracker):
        self.topic_tracker = topic_tracker
    
    def should_apply_context(self, query: str) -> bool:
        """Determine if context should be applied"""
        
        current_topic = self.topic_tracker.get_current_topic()
        
        # No topic or low confidence
        if not current_topic or current_topic.confidence < 0.6:
            return False
        
        # Check if query is vague (needs context)
        if not self._query_is_vague(query):
            return False
        
        # Check if it's a topic switch (shouldn't apply context)
        if self._is_topic_switch(query, current_topic):
            return False
        
        return True
    
    def _query_is_vague(self, query: str) -> bool:
        """Check if query contains pronouns or vague references"""
        
        vague_patterns = [
            r'\b(it|this|that|they|he|she|him|her|his|their|its)\b',
            r'\bthe (process|event|person|concept|method|system|theory|idea)\b',
            r'\b(how does|what does|why does)\b',
            r'\b(more|further|additional)\s+(information|details|about)\b',
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in vague_patterns)
    
    def _is_topic_switch(self, query: str, current_topic) -> bool:
        """Detect if query is switching to a new topic"""
        
        # Pattern 1: Explicit topic switch indicators
        switch_patterns = [
            r'^(who is|what is|tell me about|explain|describe)\s+[A-Z]',
            r'\b(now|instead|different|another|next)\s+',
            r'(switch|change)\s+(to|topic)',
            r'^[A-Z][a-z]+\s+[A-Z]',  # Starts with capitalized name
        ]
        
        for pattern in switch_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        # Pattern 2: Check for new named entities not in current topic
        new_entities = self._extract_entities(query)
        
        if new_entities:
            # Get entities from current topic
            topic_name_words = set(current_topic.name.lower().split())
            topic_keywords = set(kw.lower() for kw in current_topic.keywords)
            topic_context = topic_name_words | topic_keywords
            
            # Check if any new entity overlaps with current topic
            new_entity_words = set()
            for entity in new_entities:
                new_entity_words.update(entity.lower().split())
            
            # If no overlap, it's a topic switch
            if not (new_entity_words & topic_context):
                return True
        
        return False
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (capitalized phrases)"""
        
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities = re.findall(pattern, text)
        
        # Filter stopwords
        stopwords = {
            'According', 'The', 'This', 'That', 'These', 'Those',
            'There', 'Here', 'What', 'When', 'Where', 'Who', 'How',
            'Why', 'Which', 'Answer'
        }
        
        return [e for e in entities if e not in stopwords]
    
    def get_smart_context(self, query: str) -> str:
        """Get context only if appropriate for this query"""
        
        if self.should_apply_context(query):
            return self.topic_tracker.get_topic_context()
        
        return ""