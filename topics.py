import re
from typing import List, Optional, Dict
from dataclasses import dataclass
from collections import Counter, defaultdict

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