import numpy as np
import re  # Moved to top level for better performance
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Topic:
    """Enhanced Topic with embeddings"""
    name: str
    keywords: List[str]
    confidence: float
    first_mentioned: int
    last_mentioned: int
    embedding: Optional[np.ndarray] = None
    entity_mentions: Counter = None
    
    def __post_init__(self):
        if self.entity_mentions is None:
            self.entity_mentions = Counter()

class EnhancedTopicTracker:
    """
    Tier 2+ Topic Tracking with Embedding merging and Intent detection
    """
    
    def __init__(self, embedder):
        self.topics: List[Topic] = []
        self.turn_number = 0
        self.entity_history = defaultdict(int)
        self.embedder = embedder
        
        # Intent tracking
        self.last_intent = None
        self.intent_keywords = {
            'comparison': ['compare', 'difference', 'versus', 'vs', 'contrast'],
            'definition': ['what is', 'define', 'meaning', 'explain'],
            'causal': ['why', 'cause', 'reason', 'because', 'lead to'],
            'temporal': ['when', 'date', 'time', 'year', 'period'],
            'listing': ['list', 'types', 'kinds', 'categories', 'examples']
        }
    
    # =========================================================
    # 🚨 MISSING METHODS ADDED HERE
    # =========================================================
    def get_all_topics(self) -> List[str]:
        """Returns a list of all unique topic names tracked so far."""
        return [t.name for t in self.topics]

    def get_topic_hints(self) -> Dict:
        """Returns hints for the query refiner."""
        current = self.get_current_topic()
        return {
            'current_topic': current.name if current else None,
            'keywords': current.keywords if current else [],
            'entities': list(current.entity_mentions.keys()) if current else []
        }
    # =========================================================

    def _find_similar_topic(self, topic_name: str, topic_embedding: np.ndarray) -> Optional[Topic]:
        """Find existing topic using semantic similarity"""
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
    
    def _detect_intent(self, query: str) -> Optional[str]:
        query_lower = query.lower()
        for intent, keywords in self.intent_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return intent
        return None
    
    def _should_hard_reset(self, query: str) -> bool:
        current_intent = self._detect_intent(query)
        critical_intents = ['comparison', 'causal', 'listing']
        
        if (self.last_intent in critical_intents and 
            current_intent in critical_intents and 
            self.last_intent != current_intent):
            print(f"  🔄 Hard reset: Intent changed from {self.last_intent} → {current_intent}")
            self.last_intent = current_intent
            return True
        
        self.last_intent = current_intent
        return False
    
    def get_current_topic_with_entities(self, query: str) -> Tuple[Optional[Topic], float]:
        if not self.topics:
            return None, 0.0
        
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
        current_topic = self.get_current_topic()
        if not current_topic:
            return 8
        
        confidence = current_topic.confidence
        if confidence > 0.8: return 5
        elif confidence > 0.6: return 7
        else: return 10
    
    def update(self, query: str, answer: str, retrieved_chunks: List[str] = None):
        self.turn_number += 1
        
        if self._should_hard_reset(query):
            for topic in self.topics:
                topic.confidence *= 0.3
        
        entities = self._extract_entities(query + " " + answer)
        keywords = self._extract_keywords(query + " " + answer)
        
        for entity in entities:
            self.entity_history[entity] += 1
        
        current_topic_name = self._identify_topic(entities, keywords)
        
        if current_topic_name:
            topic_embedding = self.embedder.get_embedding(current_topic_name)
            existing = self._find_similar_topic(current_topic_name, topic_embedding)
            
            if existing:
                existing.last_mentioned = self.turn_number
                existing.keywords = list(set(existing.keywords + keywords[:5]))
                existing.confidence = min(1.0, existing.confidence + 0.1)
                for entity in entities:
                    existing.entity_mentions[entity] += 1
            else:
                new_topic = Topic(
                    name=current_topic_name,
                    keywords=keywords[:5],
                    confidence=0.5,
                    first_mentioned=self.turn_number,
                    last_mentioned=self.turn_number,
                    embedding=topic_embedding,
                    entity_mentions=Counter(entities)
                )
                self.topics.append(new_topic)
        
        # Decay
        for topic in self.topics:
            if topic.last_mentioned < self.turn_number - 3:
                topic.confidence *= 0.8
    
    def get_current_topic(self) -> Optional[Topic]:
        if not self.topics: return None
        recent_topics = [t for t in self.topics if self.turn_number - t.last_mentioned <= 3]
        if not recent_topics: return None
        return max(recent_topics, key=lambda t: t.confidence)
    
    def _extract_entities(self, text: str) -> List[str]:
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities = re.findall(pattern, text)
        stopwords = {'According', 'The', 'This', 'That', 'These', 'Those',
                    'There', 'Here', 'What', 'When', 'Where', 'Who', 'How'}
        entities = [e for e in entities if e not in stopwords]
        return list(set(entities))
    
    def _extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'\b\w{5,}\b', text.lower())
        common = {'about', 'would', 'could', 'should', 'their', 'there',
                 'which', 'these', 'those', 'where', 'other', 'being'}
        words = [w for w in words if w not in common]
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(10)]
    
    def _identify_topic(self, entities: List[str], keywords: List[str]) -> Optional[str]:
        if entities:
            entity_scores = {e: self.entity_history.get(e, 0) for e in entities}
            if entity_scores:
                return max(entity_scores, key=entity_scores.get)
        if keywords:
            keyword_scores = Counter(keywords)
            if keyword_scores:
                return keyword_scores.most_common(1)[0][0]
        return None

class CitationValidator:
    """
    Validates that answers are grounded in retrieved chunks using embeddings.
    """
    
    def __init__(self, embedder):
        self.embedder = embedder
    
    def validate_answer(self, answer: str, chunks: List[str]) -> Tuple[bool, List[str], float]:
        claims = [s.strip() for s in answer.split('.') if len(s.strip()) > 10]
        
        if not claims:
            return True, [], 1.0
        
        supported_claims = 0
        supporting_chunks = []
        
        if not chunks:
            return False, [], 0.0
            
        # Batch encode chunks
        chunk_embs = np.array([
            self.embedder.get_embedding(chunk) 
            for chunk in chunks
        ])
        
        for claim in claims:
            claim_emb = self.embedder.get_embedding(claim)
            
            similarities = cosine_similarity(
                claim_emb.reshape(1, -1),
                chunk_embs
            )[0]
            
            best_match_idx = np.argmax(similarities)
            best_match_score = similarities[best_match_idx]
            
            # Adjusted threshold to 0.65 for better precision
            if best_match_score > 0.65: 
                supported_claims += 1
                supporting_chunks.append(best_match_idx)
        
        confidence = supported_claims / len(claims) if claims else 1.0
        
        is_grounded = confidence > 0.7
        
        return is_grounded, list(set(supporting_chunks)), confidence