import os
# --- CRITICAL FIX: Set Cache Path Globally ---
os.environ['HF_HOME'] = os.path.join(os.getcwd(), "model_cache")
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(os.getcwd(), "model_cache")

import time
import hashlib
import json
import re
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any, Set
from collections import Counter, defaultdict
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import shutil
from pathlib import Path
import pickle
from dataclasses import dataclass, field

# ==========================================
# 1. SYSTEM UTILITIES & CONFIG
# ==========================================

class SystemUtils:
    @staticmethod
    def find_tesseract() -> Optional[str]:
        if tesseract_path := os.getenv("TESSERACT_PATH"): return tesseract_path
        common_paths = [r'C:\Program Files\Tesseract-OCR\tesseract.exe', r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe']
        for path in common_paths:
            if Path(path).exists(): return path
        if tesseract_cmd := shutil.which("tesseract"): return tesseract_cmd
        return None

    @staticmethod
    def find_poppler() -> Optional[str]:
        if poppler_path := os.getenv("POPPLER_PATH"): return poppler_path
        common_paths = [r'C:\poppler\Library\bin', r'C:\ProgramData\chocolatey\lib\poppler\Library\bin']
        for path in common_paths:
            if Path(path).exists(): return path
        return None

class Config:
    def __init__(self, pdf_path: str = None, groq_api_key: str = None):
        self.PDF_PATH = pdf_path or os.getenv("PDF_PATH", "Dakhil - 2018 - Class-(9-10) English For Today PDF Web.pdf")
        self.GROQ_API_KEY = groq_api_key or os.getenv("GROQ_API_KEY", "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")
        
        # --- PATHS ---
        self.CACHE_DIR = "./cache"
        self.MODEL_CACHE_DIR = os.path.join(os.getcwd(), "model_cache")
        pdf_name = os.path.splitext(os.path.basename(self.PDF_PATH))[0]
        self.CHUNKS_FILE = f"{pdf_name}_chunks.pkl"
        self.EMBEDDINGS_FILE = f"{pdf_name}_bge_embeddings.pkl"
        
        # --- TUNING PARAMETERS ---
        self.INITIAL_RETRIEVAL_K = 15
        self.FINAL_TOP_K = 5
        self.RERANK_THRESHOLD = -2.0
        
        # --- LEGACY SUPPORT (For UI Compatibility) ---
        self.TOP_K_CHUNKS = self.FINAL_TOP_K
        self.SIMILARITY_THRESHOLD = 0.0 
        self.MAX_CONVERSATION_HISTORY = 10
        self.CONTEXT_CACHE_FILE = os.path.join(self.CACHE_DIR, "answer_cache.json")

# ==========================================
# 2. CACHE & HISTORY
# ==========================================

class SimpleCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "answer_cache.json")
        self.cache = {} 
        self.answer_cache = self.cache 
        self.load()
    
    def load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache.update(json.load(f))
            except: pass
            
    def save(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except: pass
        
    def get(self, key):
        return self.cache.get(key)
        
    def set(self, key, value):
        self.cache[key] = value
        self.save()

@dataclass
class HistoryObject:
    history: List[Dict] = field(default_factory=list)
    last_entities: Dict = field(default_factory=dict)

# ==========================================
# 3. CORE COMPONENTS
# ==========================================

class GroqLLM:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.3-70b-versatile"
    
    def generate(self, prompt: str, temperature: float = 0.5) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM Error: {e}"

class Embedder:
    def __init__(self, config: Config):
        self.model = SentenceTransformer('BAAI/bge-m3')
        self.config = config

    def get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

class Reranker:
    def __init__(self, config: Config):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank(self, query: str, chunks: List[str], top_k: int) -> List[Tuple[int, float]]:
        if not chunks: return []
        
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.model.predict(pairs)
        
        results = []
        for i, score in enumerate(scores):
            results.append((i, float(score)))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

# ==========================================
# 4. HYBRID RETRIEVAL ENGINE
# ==========================================

class HybridRetriever:
    def __init__(self, chunks: List[str], embeddings: np.ndarray, embedder: Embedder, reranker: Reranker, config: Config):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder
        self.reranker = reranker
        self.config = config
        
        print("⚙️ Initializing BM25 index...")
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve(self, query: str) -> List[int]:
        k = self.config.INITIAL_RETRIEVAL_K
        
        # 1. Vector Search
        query_emb = self.embedder.get_embedding(query)
        vector_sims = cosine_similarity(query_emb.reshape(1, -1), self.embeddings)[0]
        vector_indices = np.argsort(vector_sims)[-k:][::-1]
        
        # 2. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[-k:][::-1]
        
        # 3. Merge Candidates
        candidate_indices = list(set(vector_indices) | set(bm25_indices))
        candidate_chunks = [self.chunks[i] for i in candidate_indices]
        
        # 4. Rerank
        print(f"  🔍 Reranking {len(candidate_indices)} candidates...")
        reranked_results = self.reranker.rerank(query, candidate_chunks, self.config.FINAL_TOP_K)
        
        final_indices = [candidate_indices[local_idx] for local_idx, score in reranked_results]
        
        if reranked_results:
            print(f"  🏆 Top Score: {reranked_results[0][1]:.4f}")
            
        return final_indices

# --- FIX #2: Topic-Aware Contextual Retrieval ---
class TopicAwareRetriever:
    """
    Tier 2 Retrieval: Maintains topic coherence across conversation
    Filters out off-topic chunks even if they have keyword matches
    """
    
    def __init__(self, base_retriever, embedder, config):
        self.base_retriever = base_retriever
        self.embedder = embedder
        self.config = config
        
        # Track conversation topics
        self.conversation_embedding = None
        self.topic_window = []  # Last N queries
        
    def retrieve(self, query: str, history: List[Dict]) -> List[int]:
        """
        Enhanced retrieval with topic filtering:
        1. Get candidates from base retriever
        2. Build conversation topic representation
        3. Filter candidates by topic similarity
        4. Return top-K coherent chunks
        """
        
        # Step 1: Get initial candidates (cast wider net)
        initial_k = self.config.INITIAL_RETRIEVAL_K * 2  # Double the usual
        
        # Temporarily increase retrieval
        old_k = self.config.INITIAL_RETRIEVAL_K
        self.config.INITIAL_RETRIEVAL_K = initial_k
        
        candidate_indices = self.base_retriever.retrieve(query)
        
        self.config.INITIAL_RETRIEVAL_K = old_k  # Restore
        
        # Step 2: Build topic context
        topic_context = self._build_topic_context(query, history)
        
        # Step 3: Filter by topic coherence
        if history:  # Only filter if there's conversation history
            filtered_indices = self._filter_by_topic(
                candidate_indices,
                topic_context,
                query
            )
        else:
            filtered_indices = candidate_indices
        
        # Step 4: Update conversation state
        self._update_topic_state(query)
        
        return filtered_indices[:self.config.FINAL_TOP_K]
    
    def _build_topic_context(self, query: str, history: List[Dict]) -> str:
        """Build a representation of conversation topic"""
        
        if not history:
            return query
        
        # Strategy: Combine recent queries + key parts of answers
        recent_queries = [h['user'] for h in history[-3:]]
        
        # Extract first sentence of recent answers (main points)
        recent_answer_intros = []
        for h in history[-2:]:
            sentences = h['ai'].split('.')
            if sentences:
                recent_answer_intros.append(sentences[0])
        
        # Combine everything
        topic_parts = recent_queries + recent_answer_intros + [query]
        topic_context = ' '.join(topic_parts)
        
        return topic_context
    
    # --- FIX #3: Specificity Calculation ---
    def _calculate_query_specificity(self, query: str) -> float:
        """
        Returns: 0.0 (very vague) to 1.0 (very specific)
        """
        specificity = 0.0
        words = query.lower().split()
        
        # Factor 1: Length (longer = more specific)
        if len(words) > 8:
            specificity += 0.3
        elif len(words) > 5:
            specificity += 0.15
        
        # Factor 2: Contains proper nouns (capitalized words)
        # Note: We rely on simple capitalization heuristic
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+', query))
        if proper_nouns > 0:
            specificity += 0.3
        
        # Factor 3: Contains specific question words
        specific_qwords = ['when', 'where', 'which', 'how many', 'what year']
        if any(qw in query.lower() for qw in specific_qwords):
            specificity += 0.2
        
        # Factor 4: NOT vague/generic (Penalty)
        vague_patterns = [
            r'^(who|what|tell|explain|elaborate|more|it|this|that)\b',
            r'\b(about it|on this|more details)\b'
        ]
        if any(re.search(pattern, query.lower()) for pattern in vague_patterns):
            specificity -= 0.3
        
        # Clamp to [0.1, 1.0] to prevent zero division or negative weights
        return max(0.1, min(1.0, specificity))

    def _filter_by_topic(self, indices: List[int], topic_context: str, 
                         query: str) -> List[int]:
        """Filter chunks to keep only topic-relevant ones using Dynamic Weighting"""
        
        if not indices:
            return indices
        
        # Get chunks
        chunks = [self.base_retriever.chunks[i] for i in indices]
        
        # Get embeddings
        topic_emb = self.embedder.get_embedding(topic_context)
        query_emb = self.embedder.get_embedding(query)
        
        # Encode chunks (batch for efficiency)
        chunk_embs = np.array([
            self.embedder.get_embedding(chunk) 
            for chunk in chunks
        ])
        
        # Calculate topic similarity
        topic_sims = cosine_similarity(
            topic_emb.reshape(1, -1), 
            chunk_embs
        )[0]
        
        # Calculate query similarity (for ranking)
        query_sims = cosine_similarity(
            query_emb.reshape(1, -1),
            chunk_embs
        )[0]
        
        # --- FIX #3: Dynamic Weighting Implementation ---
        query_specificity = self._calculate_query_specificity(query)
        
        # Dynamic weighting:
        # - Vague (specificity ~0.1): 75% Topic, 25% Query
        # - Specific (specificity ~0.9): 30% Topic, 70% Query
        topic_weight = 0.3 + (0.45 * (1 - query_specificity)) 
        query_weight = 1 - topic_weight
        
        print(f"  ⚖️ Dynamic Weights | Specificity: {query_specificity:.2f} | Topic: {topic_weight:.2f} | Query: {query_weight:.2f}")

        combined_scores = topic_weight * topic_sims + query_weight * query_sims
        # ------------------------------------------------
        
        # Dynamic threshold based on score distribution
        threshold = self._calculate_threshold(combined_scores)
        
        # Filter and sort
        filtered_pairs = [
            (idx, score) 
            for idx, score in zip(indices, combined_scores)
            if score >= threshold
        ]
        
        # Sort by combined score
        filtered_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract indices
        filtered_indices = [idx for idx, _ in filtered_pairs]
        
        # Debug info
        kept_ratio = len(filtered_indices) / len(indices) if indices else 0
        print(f"  🎯 Topic Filter: Kept {len(filtered_indices)}/{len(indices)} "
              f"chunks ({kept_ratio:.1%})")
        
        # Fallback: if we filtered too aggressively, keep top-K by score
        if len(filtered_indices) < 3:
            print(f"  ⚠️ Too aggressive filtering, using top-K instead")
            sorted_pairs = sorted(
                zip(indices, combined_scores), 
                key=lambda x: x[1], 
                reverse=True
            )
            filtered_indices = [idx for idx, _ in sorted_pairs[:self.config.FINAL_TOP_K]]
        
        return filtered_indices
    
    def _calculate_threshold(self, scores: np.ndarray) -> float:
        """Calculate dynamic threshold based on score distribution"""
        
        if len(scores) == 0:
            return 0.0
        
        # Use percentile-based threshold
        # Keep chunks above 40th percentile
        threshold = np.percentile(scores, 40)
        
        # But ensure minimum threshold for quality
        min_threshold = 0.3
        
        return max(threshold, min_threshold)
    
    def _update_topic_state(self, query: str):
        """Update conversation topic tracking"""
        
        # Maintain sliding window of recent queries
        self.topic_window.append(query)
        if len(self.topic_window) > 5:
            self.topic_window.pop(0)
        
        # Update conversation embedding (average of recent queries)
        if self.topic_window:
            embeddings = [
                self.embedder.get_embedding(q) 
                for q in self.topic_window
            ]
            self.conversation_embedding = np.mean(embeddings, axis=0)
    
    def get_topic_summary(self) -> str:
        """Get current conversation topic (for debugging/UI)"""
        if self.topic_window:
            return " | ".join(self.topic_window[-3:])
        return "No topic yet"
    
# --- FIX #3: Conversation Topic Tracker ---
@dataclass
class Topic:
    """Represents a conversation topic"""
    name: str
    keywords: List[str]
    confidence: float
    first_mentioned: int  # Turn number
    last_mentioned: int

class ConversationTopicTracker:
    """
    Tier 2 Topic Tracking: Understands what conversation is about
    Helps with context maintenance and query refinement
    """
    
    def __init__(self):
        self.topics: List[Topic] = []
        self.turn_number = 0
        self.entity_history = defaultdict(int)  # Track entity mentions
        
    def update(self, query: str, answer: str):
        """Update topic tracking after each turn"""
        
        self.turn_number += 1
        
        # Extract entities and keywords
        entities = self._extract_entities(query + " " + answer)
        keywords = self._extract_keywords(query + " " + answer)
        
        # Update entity history
        for entity in entities:
            self.entity_history[entity] += 1
        
        # Determine if this is a new topic or continuation
        current_topic = self._identify_topic(entities, keywords)
        
        if current_topic:
            self._update_or_add_topic(current_topic, entities, keywords)
    
    def get_current_topic(self) -> Optional[Topic]:
        """Get the most relevant current topic"""
        
        if not self.topics:
            return None
        
        # Get topics mentioned in last 3 turns
        recent_topics = [
            t for t in self.topics 
            if self.turn_number - t.last_mentioned <= 3
        ]
        
        if not recent_topics:
            return None
        
        # Return most confident recent topic
        return max(recent_topics, key=lambda t: t.confidence)
    
    def get_topic_context(self) -> str:
        """Get topic context string for query refinement"""
        
        current = self.get_current_topic()
        if not current:
            return ""
        
        # Build context string
        context_parts = [current.name]
        
        # Add top keywords
        if current.keywords:
            context_parts.extend(current.keywords[:3])
        
        return " ".join(context_parts)
    
    def get_all_topics(self) -> List[str]:
        """Get all topics discussed (for analytics)"""
        return [t.name for t in self.topics]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (proper nouns)"""
        
        # Find capitalized phrases (simple NER)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities = re.findall(pattern, text)
        
        # Filter out common false positives
        stopwords = {'According', 'The', 'This', 'That', 'These', 'Those',
                    'There', 'Here', 'What', 'When', 'Where', 'Who', 'How'}
        entities = [e for e in entities if e not in stopwords]
        
        # Deduplicate
        return list(set(entities))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords"""
        
        # Simple keyword extraction: content words > 4 chars
        words = re.findall(r'\b\w{5,}\b', text.lower())
        
        # Remove common words
        common = {'about', 'would', 'could', 'should', 'their', 'there',
                 'which', 'these', 'those', 'where', 'other', 'being'}
        words = [w for w in words if w not in common]
        
        # Get most frequent
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(10)]
    
    def _identify_topic(self, entities: List[str], keywords: List[str]) -> Optional[str]:
        """Identify main topic from entities and keywords"""
        
        # Priority 1: Named entities (people, places, things)
        if entities:
            # Most frequently mentioned entity
            entity_scores = {
                e: self.entity_history.get(e, 0) 
                for e in entities
            }
            if entity_scores:
                return max(entity_scores, key=entity_scores.get)
        
        # Priority 2: Recurring keywords
        if keywords:
            keyword_scores = Counter(keywords)
            if keyword_scores:
                return keyword_scores.most_common(1)[0][0]
        
        return None
    
    def _update_or_add_topic(self, topic_name: str, entities: List[str], 
                            keywords: List[str]):
        """Update existing topic or add new one"""
        
        # Check if topic already exists
        existing = None
        for topic in self.topics:
            # Fuzzy match (contains or is contained)
            if (topic_name.lower() in topic.name.lower() or 
                topic.name.lower() in topic_name.lower()):
                existing = topic
                break
        
        if existing:
            # Update existing topic
            existing.last_mentioned = self.turn_number
            existing.keywords = list(set(existing.keywords + keywords[:5]))
            # Increase confidence
            existing.confidence = min(1.0, existing.confidence + 0.1)
        else:
            # Add new topic
            new_topic = Topic(
                name=topic_name,
                keywords=keywords[:5],
                confidence=0.5,
                first_mentioned=self.turn_number,
                last_mentioned=self.turn_number
            )
            self.topics.append(new_topic)
        
        # Decay confidence of old topics
        for topic in self.topics:
            if topic.last_mentioned < self.turn_number - 3:
                topic.confidence *= 0.8
    
    def get_topic_hints(self) -> Dict:
        """Get topic information for debugging/UI"""
        
        current = self.get_current_topic()
        
        return {
            'current_topic': current.name if current else None,
            'current_keywords': current.keywords if current else [],
            'confidence': current.confidence if current else 0.0,
            'all_topics': self.get_all_topics(),
            'turn_number': self.turn_number
        }

# ==========================================
# 5. PROMPT & HELPERS
# ==========================================

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
        
        # Check 1: Contains pronouns
        pronouns = {'he', 'she', 'it', 'they', 'this', 'that', 'these', 'those', 
                   'his', 'her', 'its', 'their', 'him', 'them'}
        has_pronouns = any(word in pronouns for word in words)
        
        # Check 2: Very short/vague
        is_short = len(words) <= 4
        
        # Check 3: Command-style (needs context)
        vague_commands = ['elaborate', 'explain', 'tell', 'show', 'give', 
                         'do', 'what', 'how', 'describe', 'more']
        is_vague_command = any(query_lower.startswith(cmd) for cmd in vague_commands)
        
        # Check 4: Follow-up indicators
        followup_words = {'also', 'too', 'additionally', 'furthermore', 'moreover'}
        is_followup = any(word in followup_words for word in words)
        
        # Check 5: Incomplete questions
        is_incomplete = query.endswith('?') and len(words) <= 3
        
        # Refine if ANY condition is true
        return (has_pronouns or is_short or is_vague_command or 
                is_followup or is_incomplete)
    
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
        
        prompt = f"""You are a query rewriting expert. Your job is to rewrite vague or context-dependent questions into clear, standalone questions.

Conversation History:
{hist_str}
{hints}

Current Question: "{query}"

RULES:
1. Replace ALL pronouns (he, she, it, they, this, that) with specific names/subjects
2. Add missing context from conversation history
3. Make the question completely self-contained
4. Keep it concise (under 20 words)
5. Preserve the original intent

EXAMPLES:
- "when did he die" → "when did Zahir Raihan die"
- "elaborate please" → "elaborate on Zahir Raihan's role in the Liberation War"
- "do the exercise" → "show the exercise questions from the Zahir Raihan lesson"
- "what about it" → "what about Zahir Raihan's film career"
- "more details" → "provide more details about Zahir Raihan's political activism"

Rewritten Question (respond with ONLY the rewritten question, no explanation):"""

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

class SelfReflector:
    def __init__(self, llm: GroqLLM):
        self.llm = llm
        
    def verify(self, answer: str, chunks: List[str]) -> Tuple[bool, str]:
        context = "\n".join(chunks)
        prompt = f"""Context:
{context[:3000]}

Generated Answer:
{answer}

Task: Verify if the Answer is supported by the Context.
1. Does the Context contain the facts in the Answer?
2. Are there hallucinations?

Respond JSON: {{"supported": true/false, "reason": "..."}}"""

        try:
            res = self.llm.generate(prompt, temperature=0.1)
            if "true" in res.lower() and "false" not in res.lower():
                return True, "Supported"
            if "supported" in res.lower() and "false" in res.lower():
                return False, "Not supported by text"
            return True, "Passed"
        except:
            return True, "Verification failed"
        
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

class MultiQueryRetriever:
    """
    Enhanced retriever that uses multiple query variants.
    Wraps the TopicAwareRetriever.
    """
    
    def __init__(self, base_retriever, query_generator, reranker, config, chunks):
        self.base_retriever = base_retriever
        self.query_generator = query_generator
        self.reranker = reranker
        self.config = config
        self.chunks = chunks # Explicitly passed to avoid deep attribute access
        
    def retrieve(self, query: str, history: List[Dict]) -> List[int]:
        """
        Retrieve using multiple query variants:
        1. Generate query variants
        2. Retrieve with each variant
        3. Merge and deduplicate results
        4. Rerank all candidates with original query
        """
        
        # Generate variants
        queries = self.query_generator.generate_variants(query, num_variants=2)
        
        if len(queries) > 1:
            print(f"  🔄 Using {len(queries)} query variants")
            for i, q in enumerate(queries):
                if i > 0: print(f"     {i}. {q[:60]}...")
        
        # Retrieve with each variant
        all_indices = set()
        
        for variant_query in queries:
            # Pass history because base_retriever is TopicAwareRetriever
            indices = self.base_retriever.retrieve(variant_query, history)
            all_indices.update(indices)
        
        # Convert to list
        candidate_indices = list(all_indices)
        
        print(f"  📊 Multi-query retrieved {len(candidate_indices)} unique chunks")
        
        # If we got too many, rerank everything with original query
        # or if we have results, we always want to rerank the merged set against the original query
        if candidate_indices:
            candidate_chunks = [
                self.chunks[i] 
                for i in candidate_indices
            ]
            
            # Rerank with ORIGINAL query (most important)
            # We use a slightly higher top_k here to ensure we don't cut off good results too early
            rerank_k = min(len(candidate_indices), self.config.FINAL_TOP_K + 2)
            
            reranked = self.reranker.rerank(
                query, 
                candidate_chunks, 
                rerank_k
            )
            
            # Map back to original indices
            final_indices = [
                candidate_indices[local_idx] 
                for local_idx, score in reranked
            ]
            
            return final_indices[:self.config.FINAL_TOP_K]
        
        return []

# ==========================================
# 6. MAIN RAG SYSTEM
# ==========================================

class RAGSystem:
    def __init__(self, config: Config):
        print("🚀 Initializing 9.6/10 RAG System (Multi-Query Enabled)...")
        self.config = config
        self.llm = GroqLLM(config.GROQ_API_KEY)
        
        # 1. Load/Process PDF
        self.chunks, self.embeddings = self._load_data()
        
        # 2. Init Models
        self.embedder = Embedder(config)
        self.reranker = Reranker(config)
        
        # 3. Init Base Retrieval Stack
        base_retriever = HybridRetriever(
            self.chunks, self.embeddings, self.embedder, self.reranker, config
        )
        
        # Topic Aware Wrapper
        self.topic_retriever = TopicAwareRetriever(
            base_retriever=base_retriever,
            embedder=self.embedder,
            config=self.config
        )
        
        # --- FIX #4: Multi-Query Integration ---
        self.query_generator = MultiQueryGenerator(self.llm)
        
        # The main retriever is now the MultiQueryRetriever, which wraps TopicAware
        self.retriever = MultiQueryRetriever(
            base_retriever=self.topic_retriever,
            query_generator=self.query_generator,
            reranker=self.reranker,
            config=self.config,
            chunks=self.chunks
        )
        # ---------------------------------------
        
        # 4. Helpers
        self.refiner = AdvancedQueryRefiner(self.llm)
        self.verifier = SelfReflector(self.llm)
        self.topic_tracker = ConversationTopicTracker()
        
        # 5. State & Cache
        self.chat_history: List[Dict] = []
        self.history = HistoryObject() 
        self.cache = SimpleCache(config.CACHE_DIR)
        
        print("✅ System Ready")

    def _load_data(self):
        """Loads chunks and embeddings from cache or processes PDF if missing."""
        if os.path.exists(self.config.CHUNKS_FILE) and os.path.exists(self.config.EMBEDDINGS_FILE):
             try:
                 with open(self.config.CHUNKS_FILE, 'rb') as f: chunks = pickle.load(f)
                 with open(self.config.EMBEDDINGS_FILE, 'rb') as f: embeddings = pickle.load(f)
                 return chunks, embeddings
             except Exception as e:
                 print(f"⚠️ Error loading cache: {e}. Re-processing...")

        try:
            from PDFprocessing import PDFProcess 
            text = PDFProcess.process_pdf(self.config.PDF_PATH, SystemUtils.find_poppler())
            chunks = PDFProcess.create_chunks(text)
        except ImportError:
            print("⚠️ PDFprocessing module not found. Using dummy data.")
            text = "Dummy text content for testing purposes."
            chunks = [text]

        print("⚙️ Generating embeddings (this may take a moment)...")
        temp_embedder = SentenceTransformer('BAAI/bge-m3')
        embeddings = temp_embedder.encode(chunks, show_progress_bar=True)
        
        with open(self.config.CHUNKS_FILE, 'wb') as f: pickle.dump(chunks, f)
        with open(self.config.EMBEDDINGS_FILE, 'wb') as f: pickle.dump(embeddings, f)
        
        return chunks, embeddings

    # --- HELPER METHODS ---
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract Named Entities"""
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities = re.findall(pattern, text)
        stopwords = {
            'According', 'The', 'This', 'That', 'These', 'Those', 'There', 'Here', 
            'What', 'When', 'Where', 'Who', 'How', 'Answer', 
            'Liberation', 'Language', 'War'
        }
        return set([e for e in entities if e not in stopwords])

    def _check_topic_coherence(self, chunks: List[str], history: List[Dict]) -> float:
        """Check if retrieved chunks are about the same topic as recent conversation"""
        recent_text = " ".join([h['ai'] for h in history[-2:]])
        if not recent_text: return 1.0

        conversation_entities = self._extract_entities(recent_text)
        chunks_text = " ".join(chunks)
        chunk_entities = self._extract_entities(chunks_text)

        if not conversation_entities: return 1.0
        
        overlap = len(conversation_entities & chunk_entities)
        union = len(conversation_entities | chunk_entities)
        jaccard = overlap / union if union > 0 else 0.0

        conv_embedding = self.embedder.get_embedding(recent_text[:1000])
        chunks_embedding = self.embedder.get_embedding(chunks_text[:1000])
        
        semantic_sim = cosine_similarity(
            conv_embedding.reshape(1, -1),
            chunks_embedding.reshape(1, -1)
        )[0][0]

        return 0.5 * jaccard + 0.5 * semantic_sim

    def _detect_topic_drift(self, new_answer: str, query: str) -> bool:
        """Detect if new answer has drifted to a completely different topic"""
        if not self.chat_history:
            return False
        
        current_topic = self.topic_tracker.get_current_topic()
        if not current_topic:
            return False
        
        new_entities = self._extract_entities(new_answer)
        topic_keywords = set(current_topic.keywords + [current_topic.name])
        answer_words = set(re.findall(r'\w+', new_answer.lower()))
        topic_keywords_lower = {k.lower() for k in topic_keywords}
        
        keyword_overlap = len(topic_keywords_lower & answer_words)
        
        if new_entities:
            current_entity = current_topic.name
            if not any(current_entity.lower() in entity.lower() or 
                       entity.lower() in current_entity.lower() 
                       for entity in new_entities):
                if keyword_overlap == 0:
                    return True
        
        recent_context = " ".join([h['ai'] for h in self.chat_history[-2:]])
        context_emb = self.embedder.get_embedding(recent_context[:1000])
        answer_emb = self.embedder.get_embedding(new_answer[:1000])
        
        similarity = cosine_similarity(
            context_emb.reshape(1, -1),
            answer_emb.reshape(1, -1)
        )[0][0]
        
        if similarity < 0.3:
            return True
        
        return False

    # --- MAIN PIPELINE ---
    def ask(self, query: str) -> str:
        
        # 1. Get Topic Context & Refine Query
        topic_context = self.topic_tracker.get_topic_context()
        refined_query = self.refiner.refine(query, self.chat_history, topic_context)
        
        # 2. Multi-Query Retrieval (Wraps TopicAware & Hybrid)
        # Note: We pass chat_history because the wrapped TopicAwareRetriever needs it
        chunk_indices = self.retriever.retrieve(refined_query, self.chat_history)
        retrieved_chunks = [self.chunks[i] for i in chunk_indices]
        
        # 2a. Coherence Validation (Fix #2) - Still useful as a final check
        if self.chat_history and retrieved_chunks:
            coherence_score = self._check_topic_coherence(retrieved_chunks, self.chat_history)
            if coherence_score < 0.45:
                print(f"  ⚠️ Low Topic Coherence ({coherence_score:.2f}). Re-retrieving...")
                current_topic = self.topic_tracker.get_current_topic()
                if current_topic:
                    enhanced_query = f"{refined_query} regarding {current_topic.name}"
                    print(f"  🔄 Contextual Retry: '{enhanced_query}'")
                    # Recursive call or retry logic
                    chunk_indices = self.retriever.retrieve(enhanced_query, self.chat_history)
                    retrieved_chunks = [self.chunks[i] for i in chunk_indices]

        if not retrieved_chunks:
            return "I couldn't find any relevant information in the document."

        # 3. Generate Initial Answer
        prompt = PromptBuilder.build(refined_query, retrieved_chunks, self.chat_history)
        answer = self.llm.generate(prompt)
        
        # 4. Verify (Self-Reflection)
        is_valid, reason = self.verifier.verify(answer, retrieved_chunks)
        if not is_valid:
            print(f"  ⚠️ Answer rejected by Verifier: {reason}")
            prompt += "\n\nCRITICAL: The previous answer was rejected. Answer ONLY using the context."
            answer = self.llm.generate(prompt)

        # 5. Detect Topic Drift (Fix #4)
        if self.chat_history:
            drift_detected = self._detect_topic_drift(answer, query)
            if drift_detected:
                print(f"  🚨 TOPIC DRIFT DETECTED - Rejecting answer")
                
                current_topic = self.topic_tracker.get_current_topic()
                if current_topic:
                    # Strategy: Force retrieval AND generation to stay on-topic
                    constrained_query = f"{refined_query} specifically about {current_topic.name}"
                    
                    # Re-retrieve with constraint
                    chunk_indices = self.retriever.retrieve(constrained_query, self.chat_history)
                    retrieved_chunks = [self.chunks[i] for i in chunk_indices]
                    
                    # Re-generate
                    prompt = PromptBuilder.build(constrained_query, retrieved_chunks, self.chat_history)
                    answer = self.llm.generate(prompt)

        # 6. Finalize & Update State
        self._update_history(query, answer)
        
        # IMPORTANT: Only update topic tracker with the FINAL, validated answer
        self.topic_tracker.update(query, answer)
        
        return answer

    def _update_history(self, question, answer):
        self.chat_history.append({"user": question, "ai": answer})
        self.history.history.append({"question": question, "answer": answer})

    def clear_history(self):
        self.chat_history = []
        self.history.history = []
        self.topic_tracker = ConversationTopicTracker()

    def get_topic_status(self) -> Dict:
        return self.topic_tracker.get_topic_hints()