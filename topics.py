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
                'her', 'us', 'them', 'my', 'your', 'his', 'their', 'our'
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
                'সেই', 'তাই', 'এমন', 'ওই', 'এর', 'তার', 'যার'
            }
        }
        
        # Hard reset patterns
        self.hard_reset_patterns = {
            'en': [
                r'\b(new topic|different topic|change topic|moving on)\b',
                r'\b(now let\'s|let\'s talk about|tell me about)\b',
                r'\b(switch to|shifting to)\b'
            ],
            'bn': [
                r'\b(নতুন বিষয়|ভিন্ন বিষয়|বিষয় পরিবর্তন)\b',
                r'\b(এখন বলুন|এখন আলোচনা|এখন বলি)\b',
                r'\b(স্থানান্তর করুন|স্থানান্তর করি)\b'
            ]
        }
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords (language-aware)"""
        lang = self.normalizer.detect_language(text)
        
        if lang == 'en':
            # English: extract words 5+ chars, lowercase
            words = re.findall(r'\b\w{5,}\b', text.lower())
        else:  # Bengali
            # Bengali: extract words 5+ Bengali chars, no lowercasing
            words = re.findall(r'[\u0980-\u09FF]{5,}', text)
        
        # Remove stopwords
        stopwords = self.stopwords[lang]
        keywords = {w for w in words if w not in stopwords}
        
        return keywords
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract named entities (language-aware)"""
        lang = self.normalizer.detect_language(text)
        entities = set()
        
        if lang == 'en':
            # English: capitalized words/phrases
            pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            entities = set(re.findall(pattern, text))
        else:  # Bengali
            # Bengali: Harder to detect entities without NER
            # Heuristic: Look for repeated noun-like patterns
            # For now, extract longer words that appear in both query and answer
            words = re.findall(r'[\u0980-\u09FF]{4,}', text)
            
            # Simple frequency-based approach
            word_counts = Counter(words)
            # Entities often appear multiple times
            entities = {word for word, count in word_counts.items() if count >= 2}
        
        return entities
    
    def _find_similar_topic(self, topic_name: str, topic_embedding: np.ndarray) -> Optional[Topic]:
        """Find similar existing topic using embeddings"""
        
        if not self.topics:
            return None
        
        similarities = []
        for existing_topic in self.topics:
            if existing_topic.embedding is not None:
                sim = cosine_similarity(
                    topic_embedding.reshape(1, -1),
                    existing_topic.embedding.reshape(1, -1)
                )[0][0]
                similarities.append((existing_topic, sim))
        
        if not similarities:
            return None
        
        best_match, best_sim = max(similarities, key=lambda x: x[1])
        
        # Merge if similarity is high
        if best_sim > 0.75:
            print(f"  🔗 Merging '{topic_name}' into '{best_match.name}' (sim: {best_sim:.3f})")
            return best_match
        
        return None
    
    def get_current_topic(self) -> Optional[Topic]:
        """Get the currently active topic"""
        if not self.topics:
            return None
        
        # Get most recently mentioned topic with high confidence
        recent_topics = [
            t for t in self.topics 
            if self.turn_number - t.last_mentioned <= 2
        ]
        
        if not recent_topics:
            return None
        
        # Return highest confidence recent topic
        return max(recent_topics, key=lambda t: t.confidence)
    
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
                'language': 'en'
            }
        
        return {
            'current_topic': current_topic.name,
            'confidence': current_topic.confidence,
            'keywords': list(current_topic.keywords)[:5],  # Top 5 keywords
            'entities': dict(current_topic.entity_mentions),
            'language': current_topic.language,
            'last_mentioned': current_topic.last_mentioned,
            'turns_ago': self.turn_number - current_topic.last_mentioned
        }
    
    def match_topic_to_query(self, query: str) -> Tuple[Optional[Topic], float]:
        """Match query to existing topics using entity overlap"""
        query_entities = self._extract_entities(query)
        
        recent_topics = [
            t for t in self.topics 
            if self.turn_number - t.last_mentioned <= 3
        ]
        
        if not recent_topics:
            return None, 0.0
        
        scored_topics = []
        for topic in recent_topics:
            overlap = len(set(query_entities) & set(topic.entity_mentions.keys()))
            total_entities = len(query_entities) if query_entities else 1
            entity_score = overlap / total_entities
            
            combined_score = (topic.confidence * 0.6) + (entity_score * 0.4)
            scored_topics.append((topic, combined_score, entity_score))
        
        best_topic, best_score, entity_overlap = max(scored_topics, key=lambda x: x[1])
        return best_topic, entity_overlap
    
    def get_dynamic_retrieval_k(self) -> int:
        """Get dynamic retrieval K based on topic confidence"""
        current_topic = self.get_current_topic()
        if not current_topic:
            return 8
        
        confidence = current_topic.confidence
        if confidence > 0.8:
            return 5
        elif confidence > 0.6:
            return 7
        else:
            return 10
    
    def _should_hard_reset(self, query: str) -> bool:
        """Detect explicit topic changes (language-aware)"""
        lang = self.normalizer.detect_language(query)
        query_normalized = self.normalizer.normalize(query)
        
        patterns = self.hard_reset_patterns[lang]
        
        for pattern in patterns:
            if re.search(pattern, query_normalized):
                print(f"  🔄 Hard reset detected")
                return True
        
        return False
    
    def update(self, query: str, answer: str, retrieved_chunks: List[str] = None):
        """Update topic tracking with new turn (multilingual)"""
        self.turn_number += 1
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
        
        if not keywords:
            return
        
        # Determine topic name (most common entity or keyword)
        if entities:
            # Use most frequent entity
            entity_counts = Counter(entities)
            topic_name = entity_counts.most_common(1)[0][0]
        else:
            # Use most common keyword
            topic_name = list(keywords)[0]
        
        # Create embedding for topic
        topic_embedding = self.embedder.get_embedding(topic_name)
        
        # Check for similar existing topic
        similar_topic = self._find_similar_topic(topic_name, topic_embedding)
        
        if similar_topic:
            # Merge into existing topic
            similar_topic.keywords.update(keywords)
            similar_topic.confidence = min(1.0, similar_topic.confidence + 0.2)
            similar_topic.last_mentioned = self.turn_number
            
            # Update entity mentions
            for entity in entities:
                similar_topic.entity_mentions[entity] = \
                    similar_topic.entity_mentions.get(entity, 0) + 1
            
            print(f"  📊 Reinforced topic: '{similar_topic.name}' (conf: {similar_topic.confidence:.2f})")
        
        else:
            # Create new topic
            new_topic = Topic(
                name=topic_name,
                keywords=keywords,
                embedding=topic_embedding,
                confidence=0.7,
                last_mentioned=self.turn_number,
                entity_mentions={e: 1 for e in entities},
                language=lang
            )
            
            self.topics.append(new_topic)
            print(f"  ➕ New topic: '{topic_name}' ({lang})")
        
        # Decay old topics
        for topic in self.topics:
            if topic.last_mentioned < self.turn_number:
                decay = 0.1 * (self.turn_number - topic.last_mentioned)
                topic.confidence = max(0.1, topic.confidence - decay)
    
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
        
        # Get confidence (max similarity)
        confidence = float(np.max(similarities))
        
        # Identify supporting chunks (similarity > 0.5)
        supporting_indices = np.where(similarities > 0.5)[0]
        supporting_chunks = [retrieved_chunks[i] for i in supporting_indices]
        
        # Grounding threshold
        is_grounded = confidence > 0.6
        
        return is_grounded, supporting_chunks, confidence