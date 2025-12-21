import os
import re
import pickle
import numpy as np
from typing import List, Dict, Set
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from config import Config, SystemUtils, SimpleCache, HistoryObject
from models import GroqLLM, Embedder, Reranker
from retrieval import HybridRetriever, TopicAwareRetriever, MultiQueryRetriever
from processing import PromptBuilder, AdvancedQueryRefiner, SelfReflector, MultiQueryGenerator
from topics import ConversationTopicTracker

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
    
    def clear_cache(self):
        """Clear the answer cache"""
        self.cache.cache = {}
        self.cache.save()
        print("🗑️ Cache cleared")

    def get_topic_status(self) -> Dict:
        return self.topic_tracker.get_topic_hints()