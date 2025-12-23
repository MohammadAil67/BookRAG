import re
import numpy as np
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from config import Config
from models import Embedder, Reranker

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
    
class MultiQueryRetriever:
    """
    Enhanced retriever with PARALLEL multi-query processing.
    Uses ThreadPoolExecutor for concurrent retrieval.
    """
    
    def __init__(self, base_retriever, query_generator, reranker, config, chunks):
        self.base_retriever = base_retriever
        self.query_generator = query_generator
        self.reranker = reranker
        self.config = config
        self.chunks = chunks
        
    def retrieve(self, query: str, history: List[Dict]) -> List[int]:
        """
        Retrieve using multiple query variants IN PARALLEL:
        1. Generate query variants
        2. Retrieve with each variant concurrently
        3. Merge and deduplicate results
        4. Rerank all candidates with original query
        """
        
        # Generate variants
        queries = self.query_generator.generate_variants(query, num_variants=2)
        
        if len(queries) > 1:
            print(f"  🔄 Using {len(queries)} query variants (parallel processing)")
            for i, q in enumerate(queries):
                if i > 0: print(f"     {i}. {q[:60]}...")
        
        # Retrieve with each variant IN PARALLEL
        all_indices = set()
        
        if len(queries) > 1:
            # Use multithreading for multiple queries
            with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
                # Submit all retrieval tasks
                future_to_query = {
                    executor.submit(self.base_retriever.retrieve, variant_query): variant_query
                    for variant_query in queries
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_query):
                    try:
                        indices = future.result()
                        all_indices.update(indices)
                    except Exception as e:
                        print(f"  ⚠️ Retrieval failed for a variant: {e}")
        else:
            # Single query - no threading needed
            indices = self.base_retriever.retrieve(queries[0])
            all_indices.update(indices)
        
        # Convert to list
        candidate_indices = list(all_indices)
        
        print(f"  📊 Multi-query retrieved {len(candidate_indices)} unique chunks")
        
        # Rerank with ORIGINAL query
        if candidate_indices:
            candidate_chunks = [
                self.chunks[i] 
                for i in candidate_indices
            ]
            
            # Rerank with higher top_k to avoid cutting off good results
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