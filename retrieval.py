import re
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from config import Config
from models import Embedder, Reranker
from processing import MultiQueryGenerator

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