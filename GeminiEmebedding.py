import hashlib
import json
import os
import time
from typing import List
import google.generativeai as genai
import numpy as np

BATCH_SIZE = 100 
GEMINI_EMBEDDINGS_CACHE = "gemini_embeddings_cache.json"


# ----------------------
# 3️⃣ Optimized Gemini Embeddings with Caching & Batching
# ----------------------
class OptimizedGeminiEmbeddings:
    def __init__(self, cache_file=GEMINI_EMBEDDINGS_CACHE):  # ✅ Already accepts custom file
        self.model_name = "models/text-embedding-004"
        self.call_count = 0
        self.min_delay = 3.0  # Increased to 3 seconds between calls
        self.last_call_time = None
        self.cache_file = cache_file
        self.cache = self.load_cache()
        print(f"📦 Loaded {len(self.cache)} cached Gemini embeddings")
    
    def load_cache(self) -> dict:
        """Load cached embeddings"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_cache(self):
        """Save embeddings cache"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False)
    
    def get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def rate_limit(self):
        """Apply rate limiting"""
        if self.last_call_time:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_delay:
                sleep_time = self.min_delay - elapsed
                time.sleep(sleep_time)
        self.last_call_time = time.time()
    
    def embed_texts_batch(self, texts: List[str], task_type="retrieval_document") -> np.ndarray:
        """
        Embed multiple texts in batches with caching
        MAJOR OPTIMIZATION: Batching reduces API calls by ~100x
        """
        embeddings = []
        texts_to_embed = []
        cached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            text_hash = self.get_text_hash(text)
            if text_hash in self.cache:
                embeddings.append(self.cache[text_hash])
                cached_indices.append(i)
            else:
                texts_to_embed.append((i, text, text_hash))
        
        print(f"  💾 Found {len(cached_indices)} cached embeddings")
        print(f"  🔄 Need to embed {len(texts_to_embed)} new texts")
        
        if not texts_to_embed:
            return np.array(embeddings)
        
        # Embed in batches
        new_embeddings = [None] * len(texts)
        for i in cached_indices:
            new_embeddings[i] = embeddings[cached_indices.index(i)]
        
        # IMPORTANT: Add delay before first embedding call
        if self.call_count == 0:
            print(f"  ⏸️  Initial 5s delay before first API call...")
            time.sleep(5)
        
        for batch_start in range(0, len(texts_to_embed), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(texts_to_embed))
            batch = texts_to_embed[batch_start:batch_end]
            
            batch_texts = [item[1] for item in batch]
            
            self.rate_limit()
            
            try:
                print(f"  🔄 Gemini batch embedding {batch_start+1}-{batch_end}/{len(texts_to_embed)}...")
                
                # BATCH API CALL - This is the key optimization!
                result = genai.embed_content(
                    model=self.model_name,
                    content=batch_texts,
                    task_type=task_type
                )
                
                self.call_count += 1  # Only 1 API call for entire batch!
                print(f"  ✅ Batch embedded successfully (API call #{self.call_count})")
                
                # Store results and cache them
                for j, (orig_idx, text, text_hash) in enumerate(batch):
                    embedding = result['embedding'][j] if isinstance(result['embedding'][0], list) else result['embedding']
                    new_embeddings[orig_idx] = embedding
                    self.cache[text_hash] = embedding
                
            except Exception as e:
                error_msg = str(e)
                print(f"  ⚠️ Error in batch {batch_start+1}-{batch_end}: {error_msg}")
                
                # Check if rate limit error
                if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    print(f"  ⚠️  RATE LIMIT detected during embedding")
                    print(f"  ⏸️  Waiting 60 seconds before retry...")
                    time.sleep(60)
                
                # Fallback to individual embedding with longer delays
                for orig_idx, text, text_hash in batch:
                    try:
                        time.sleep(5)  # 5 second delay between individual embeds
                        result = genai.embed_content(
                            model=self.model_name,
                            content=text,
                            task_type=task_type
                        )
                        self.call_count += 1
                        embedding = result['embedding']
                        new_embeddings[orig_idx] = embedding
                        self.cache[text_hash] = embedding
                    except Exception as e2:
                        print(f"  ⚠️ Error embedding individual text: {e2}")
                        new_embeddings[orig_idx] = [0.0] * 768
        
        # Save updated cache
        self.save_cache()
        
        return np.array([emb for emb in new_embeddings if emb is not None])
    
    def embed_query(self, text: str) -> np.ndarray:
        """Embed query with caching"""
        text_hash = self.get_text_hash(text)
        
        if text_hash in self.cache:
            print(f"  💾 Using cached query embedding")
            return np.array(self.cache[text_hash])
        
        self.rate_limit()
        
        try:
            print(f"  🔄 Gemini embedding query...")
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            self.call_count += 1
            embedding = result['embedding']
            self.cache[text_hash] = embedding
            self.save_cache()
            return np.array(embedding)
            
        except Exception as e:
            print(f"  ⚠️ Error embedding query: {e}")
            return np.zeros(768)