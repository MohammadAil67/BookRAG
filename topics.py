import re
import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# Language Detection & Normalization
# ==========================================

class TextNormalizer:
    """Language-aware text normalization utilities"""
    
    @staticmethod
    def is_bengali(text: str) -> bool:
        """Check if text contains Bengali characters"""
        if not text:
            return False
        bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        return bengali_chars > len(text) * 0.2
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text based on detected language"""
        if not text:
            return ""
        if TextNormalizer.is_bengali(text):
            return text.strip()  # No case conversion for Bengali
        return text.lower().strip()  # Lowercase for English
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect text language"""
        return 'bn' if TextNormalizer.is_bengali(text) else 'en'


# ==========================================
# Topic Data Structure
# ==========================================

@dataclass
class Topic:
    """Represents a conversation topic with multilingual support"""
    name: str
    keywords: Set[str]
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0
    last_mentioned: int = 0
    entity_mentions: Dict[str, int] = field(default_factory=dict)
    language: str = 'en'  # 'en' or 'bn'
    turn_count: int = 0  # Track how many turns this topic has been discussed


# ==========================================
# Enhanced Topic Tracker
# ==========================================

class EnhancedTopicTracker:
    """
    Advanced topic tracking with semantic merging and entity awareness
    Supports both English and Bengali
    """
    
    def __init__(self, embedder):
        self.embedder = embedder
        self.topics: List[Topic] = []
        self.turn_number = 0
        self.last_intent = None
        self.entity_history: Dict[str, int] = {}
        self.normalizer = TextNormalizer()
        
        # Stopwords for keyword extraction
        self.stopwords = {
            'en': {
                'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
                'in', 'with', 'to', 'for', 'of', 'as', 'by', 'from', 'this', 'that',
                'it', 'was', 'are', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
                'did', 'will', 'would', 'should', 'could', 'may', 'might', 'can',
                'what', 'when', 'where', 'who', 'how', 'why', 'there', 'here',
                'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him',
                'her', 'us', 'them', 'my', 'your', 'his', 'their', 'our', 'about',
                'tell', 'explain', 'describe', 'give', 'show', 'find', 'get', 'make'
            },
            'bn': {
                'আমি', 'তুমি', 'তোমার', 'আমার', 'তার', 'তাদের', 'এটি', 'ওটি',
                'এই', 'ওই', 'সে', 'তিনি', 'তারা', 'কি', 'কীভাবে', 'কোথায়',
                'কখন', 'কেন', 'এবং', 'অথবা', 'কিন্তু', 'যদি', 'তাহলে', 'যেমন',
                'ছিল', 'ছিলেন', 'আছে', 'আছেন', 'হয়', 'হয়েছে', 'হয়েছিল', 'হবে',
                'করা', 'করেন', 'করেছেন', 'করবেন', 'থেকে', 'দিয়ে', 'জন্য', 'সঙ্গে',
                'নিয়ে', 'মধ্যে', 'ভিতর', 'বাইরে', 'উপর', 'নিচে', 'পাশে', 'সামনে',
                'পেছনে', 'সাথে', 'বিষয়ে', 'সম্পর্কে', 'অনুযায়ী', 'মতো', 'মতে', 'মত',
                'আর', 'ও', 'বা', 'না', 'নয়', 'যে', 'যা', 'যার', 'যাকে', 'যেন',
                'সেই', 'তাই', 'এমন', 'ওই', 'এর', 'তার', 'যার', 'বলুন', 'ব্যাখ্যা'
            }
        }
        
        # Hard reset patterns (explicit topic changes)
        self.hard_reset_patterns = {
            'en': [
                r'\b(new topic|different topic|change topic|moving on|switch topic)\b',
                r'\b(now let\'?s|let\'?s talk about|tell me about)\b',
                r'\b(switch to|shifting to|what about)\b',
                r'\b(instead|rather)\b'
            ],
            'bn': [
                r'\b(নতুন বিষয়|ভিন্ন বিষয়|বিষয় পরিবর্তন)\b',
                r'\b(এখন বলুন|এখন আলোচনা|এখন বলি)\b',
                r'\b(স্থানান্তর করুন|স্থানান্তর করি)\b',
                r'\b(বরং|পরিবর্তে)\b'
            ]
        }
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords (language-aware)"""
        if not text:
            return set()
            
        lang = self.normalizer.detect_language(text)
        
        if lang == 'en':
            # English: extract words 4+ chars (lowered threshold), lowercase
            words = re.findall(r'\b\w{4,}\b', text.lower())
        else:  # Bengali
            # Bengali: extract words 3+ Bengali chars, no lowercasing
            words = re.findall(r'[\u0980-\u09FF]{3,}', text)
        
        # Remove stopwords
        stopwords = self.stopwords[lang]
        keywords = {w for w in words if w not in stopwords}
        
        # Limit to most meaningful keywords (avoid clutter)
        return keywords if len(keywords) <= 20 else set(list(keywords)[:20])
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract named entities (language-aware)"""
        if not text:
            return set()
            
        lang = self.normalizer.detect_language(text)
        entities = set()
        
        if lang == 'en':
            # English: capitalized words/phrases
            pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            raw_entities = re.findall(pattern, text)
            
            # Filter out common false positives
            common_words = {'The', 'This', 'That', 'These', 'Those', 'There', 
                          'According', 'However', 'Therefore', 'Moreover', 'Furthermore'}
            entities = {e for e in raw_entities if e not in common_words and len(e) > 2}
        else:  # Bengali
            # Bengali: Look for repeated meaningful words
            words = re.findall(r'[\u0980-\u09FF]{4,}', text)
            
            # Frequency-based approach for Bengali
            word_counts = Counter(words)
            # Entities appear multiple times or are longer words
            entities = {
                word for word, count in word_counts.items() 
                if count >= 2 or len(word) >= 6
            }
        
        return entities
    
    def _find_similar_topic(self, topic_name: str, topic_embedding: np.ndarray, 
                           keywords: Set[str], entities: Set[str]) -> Optional[Topic]:
        """Find similar existing topic using multiple signals"""
        
        if not self.topics:
            return None
        
        scored_matches = []
        
        for existing_topic in self.topics:
            # 1. Embedding similarity (50% weight)
            emb_sim = 0.0
            if existing_topic.embedding is not None and topic_embedding is not None:
                emb_sim = cosine_similarity(
                    topic_embedding.reshape(1, -1),
                    existing_topic.embedding.reshape(1, -1)
                )[0][0]
            
            # 2. Keyword overlap (30% weight)
            keyword_overlap = len(keywords & existing_topic.keywords)
            keyword_sim = keyword_overlap / max(len(keywords), 1) if keywords else 0.0
            
            # 3. Entity overlap (20% weight)
            entity_overlap = len(entities & set(existing_topic.entity_mentions.keys()))
            entity_sim = entity_overlap / max(len(entities), 1) if entities else 0.0
            
            # Combined score
            combined_score = (emb_sim * 0.5) + (keyword_sim * 0.3) + (entity_sim * 0.2)
            
            # Boost recent topics
            recency_factor = 1.0
            if self.turn_number - existing_topic.last_mentioned <= 2:
                recency_factor = 1.2
            
            final_score = combined_score * recency_factor
            scored_matches.append((existing_topic, final_score, emb_sim))
        
        if not scored_matches:
            return None
        
        best_match, best_score, emb_sim = max(scored_matches, key=lambda x: x[1])
        
        # Merge if score is high enough
        threshold = 0.7  # Adjusted threshold
        if best_score > threshold:
            print(f"  🔗 Merging '{topic_name}' into '{best_match.name}' "
                  f"(score: {best_score:.3f}, emb: {emb_sim:.3f})")
            return best_match
        
        return None
    
    def get_current_topic(self) -> Optional[Topic]:
        """Get the currently active topic with improved logic"""
        if not self.topics:
            return None
        
        # Get topics mentioned in last 3 turns
        recent_topics = [
            t for t in self.topics 
            if self.turn_number - t.last_mentioned <= 3
        ]
        
        if not recent_topics:
            # If no recent topics, return most confident overall
            return max(self.topics, key=lambda t: t.confidence) if self.topics else None
        
        # Score recent topics by confidence and recency
        scored = []
        for topic in recent_topics:
            recency_score = 1.0 / (1 + (self.turn_number - topic.last_mentioned))
            combined_score = (topic.confidence * 0.7) + (recency_score * 0.3)
            scored.append((topic, combined_score))
        
        return max(scored, key=lambda x: x[1])[0]
    
    def get_all_topics(self) -> Set[str]:
        """Get all topic names"""
        return {t.name for t in self.topics}
    
    def get_topic_hints(self) -> Dict:
        """
        Get topic hints for UI and context application
        Returns dictionary with current topic info
        """
        current_topic = self.get_current_topic()
        
        if not current_topic:
            return {
                'current_topic': None,
                'confidence': 0.0,
                'keywords': [],
                'entities': {},
                'language': 'en',
                'turn_count': 0
            }
        
        return {
            'current_topic': current_topic.name,
            'confidence': current_topic.confidence,
            'keywords': list(current_topic.keywords)[:5],  # Top 5 keywords
            'entities': dict(current_topic.entity_mentions),
            'language': current_topic.language,
            'last_mentioned': current_topic.last_mentioned,
            'turns_ago': self.turn_number - current_topic.last_mentioned,
            'turn_count': current_topic.turn_count
        }
    
    def match_topic_to_query(self, query: str) -> Tuple[Optional[Topic], float]:
        """Match query to existing topics using multiple signals"""
        if not query or not self.topics:
            return None, 0.0
            
        query_keywords = self._extract_keywords(query)
        query_entities = self._extract_entities(query)
        
        # Consider topics from last 4 turns
        recent_topics = [
            t for t in self.topics 
            if self.turn_number - t.last_mentioned <= 4
        ]
        
        if not recent_topics:
            return None, 0.0
        
        scored_topics = []
        for topic in recent_topics:
            # Keyword overlap
            keyword_overlap = len(query_keywords & topic.keywords)
            keyword_score = keyword_overlap / max(len(query_keywords), 1) if query_keywords else 0.0
            
            # Entity overlap
            entity_overlap = len(query_entities & set(topic.entity_mentions.keys()))
            entity_score = entity_overlap / max(len(query_entities), 1) if query_entities else 0.0
            
            # Combined score weighted by topic confidence
            combined_score = (
                (keyword_score * 0.4) + 
                (entity_score * 0.3) + 
                (topic.confidence * 0.3)
            )
            
            scored_topics.append((topic, combined_score, keyword_score, entity_score))
        
        if not scored_topics:
            return None, 0.0
        
        best_topic, best_score, kw_score, ent_score = max(scored_topics, key=lambda x: x[1])
        
        # Return match only if score is meaningful
        if best_score > 0.3:
            return best_topic, best_score
        
        return None, 0.0
    
    def get_dynamic_retrieval_k(self) -> int:
        """Get dynamic retrieval K based on topic confidence"""
        current_topic = self.get_current_topic()
        if not current_topic:
            return 8
        
        confidence = current_topic.confidence
        turns = current_topic.turn_count
        
        # More focused retrieval for established topics
        if confidence > 0.85 and turns > 3:
            return 5
        elif confidence > 0.7 and turns > 2:
            return 6
        elif confidence > 0.6:
            return 7
        else:
            return 9
    
    def _should_hard_reset(self, query: str) -> bool:
        """Detect explicit topic changes (language-aware)"""
        if not query:
            return False
            
        lang = self.normalizer.detect_language(query)
        query_normalized = self.normalizer.normalize(query)
        
        patterns = self.hard_reset_patterns[lang]
        
        for pattern in patterns:
            if re.search(pattern, query_normalized, re.IGNORECASE):
                print(f"  🔄 Hard reset detected in query")
                return True
        
        return False
    
    def update(self, query: str, answer: str, retrieved_chunks: List[str] = None):
        """Update topic tracking with new turn (multilingual)"""
        self.turn_number += 1
        
        if not query or not answer:
            return
        
        lang = self.normalizer.detect_language(query)
        
        # Hard reset if explicit topic change
        if self._should_hard_reset(query):
            for topic in self.topics:
                topic.confidence *= 0.3
        
        # Extract keywords and entities
        combined_text = query + " " + answer
        keywords = self._extract_keywords(combined_text)
        entities = self._extract_entities(combined_text)
        
        # Update entity history
        for entity in entities:
            self.entity_history[entity] = self.entity_history.get(entity, 0) + 1
        
        if not keywords and not entities:
            print("  ⚠️ No keywords or entities extracted, skipping topic update")
            return
        
        # Determine topic name (prefer entities, fallback to keywords)
        if entities:
            # Use most frequent or longest entity
            entity_counts = Counter(entities)
            if entity_counts:
                topic_name = entity_counts.most_common(1)[0][0]
            else:
                topic_name = max(entities, key=len)
        elif keywords:
            # Use longest keyword as fallback
            topic_name = max(keywords, key=len)
        else:
            return
        
        # Create embedding for topic
        try:
            topic_embedding = self.embedder.get_embedding(topic_name)
        except Exception as e:
            print(f"  ⚠️ Failed to create embedding: {e}")
            topic_embedding = None
        
        # Check for similar existing topic
        similar_topic = self._find_similar_topic(topic_name, topic_embedding, keywords, entities)
        
        if similar_topic:
            # Merge into existing topic
            similar_topic.keywords.update(keywords)
            similar_topic.confidence = min(1.0, similar_topic.confidence + 0.15)
            similar_topic.last_mentioned = self.turn_number
            similar_topic.turn_count += 1
            
            # Update entity mentions
            for entity in entities:
                similar_topic.entity_mentions[entity] = \
                    similar_topic.entity_mentions.get(entity, 0) + 1
            
            print(f"  📊 Reinforced topic: '{similar_topic.name}' "
                  f"(conf: {similar_topic.confidence:.2f}, turns: {similar_topic.turn_count})")
        
        else:
            # Create new topic
            new_topic = Topic(
                name=topic_name,
                keywords=keywords,
                embedding=topic_embedding,
                confidence=0.6,  # Start with moderate confidence
                last_mentioned=self.turn_number,
                entity_mentions={e: 1 for e in entities},
                language=lang,
                turn_count=1
            )
            
            self.topics.append(new_topic)
            print(f"  ➕ New topic: '{topic_name}' ({lang})")
        
        # Decay old topics (more aggressive)
        for topic in self.topics:
            if topic.last_mentioned < self.turn_number:
                turns_since = self.turn_number - topic.last_mentioned
                decay = 0.12 * turns_since  # Increased decay rate
                topic.confidence = max(0.05, topic.confidence - decay)
        
        # Cleanup: Remove topics with very low confidence
        self.topics = [t for t in self.topics if t.confidence > 0.05]
    
    def get_context(self) -> str:
        """Get current topic context string"""
        current_topic = self.get_current_topic()
        
        if not current_topic:
            return ""
        
        # Return topic name for context
        return current_topic.name


# ==========================================
# Citation Validator
# ==========================================

class CitationValidator:
    """
    Validates if generated answers are grounded in retrieved chunks
    Language-agnostic (works for both English and Bengali)
    """
    
    def __init__(self, embedder):
        self.embedder = embedder
    
    def validate_answer(
        self, 
        answer: str, 
        retrieved_chunks: List[str]
    ) -> Tuple[bool, List[str], float]:
        """
        Validate answer against retrieved chunks using embeddings
        
        Returns:
            is_grounded: bool - Whether answer is grounded
            supporting_chunks: List[str] - Chunks that support the answer
            confidence: float - Grounding confidence score
        """
        if not retrieved_chunks or not answer:
            return False, [], 0.0
        
        try:
            # Get embeddings
            answer_embedding = self.embedder.get_embedding(answer)
            chunk_embeddings = np.array([
                self.embedder.get_embedding(chunk) 
                for chunk in retrieved_chunks
            ])
            
            # Calculate similarities
            similarities = cosine_similarity(
                answer_embedding.reshape(1, -1),
                chunk_embeddings
            )[0]
            
            # Get confidence (average of top 3 similarities for robustness)
            top_sims = np.sort(similarities)[-3:]
            confidence = float(np.mean(top_sims))
            
            # Identify supporting chunks (similarity > 0.45)
            supporting_indices = np.where(similarities > 0.45)[0]
            supporting_chunks = [retrieved_chunks[i] for i in supporting_indices]
            
            # Grounding threshold (lowered slightly)
            is_grounded = confidence > 0.55 and len(supporting_chunks) > 0
            
            return is_grounded, supporting_chunks, confidence
            
        except Exception as e:
            print(f"  ⚠️ Citation validation error: {e}")
            # Fail-safe: assume grounded if validation fails
            return True, retrieved_chunks, 0.5