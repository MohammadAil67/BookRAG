import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import json
import numpy as np
from datetime import datetime
from pdf2image import convert_from_path
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import google.api_core.exceptions
from typing import List, Tuple, Optional
import google.generativeai as genai
import subprocess  # Add this with your other imports
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from GeminiEmebedding import OptimizedGeminiEmbeddings
from PDFprocessing import PDFProcess

from LocalEmbedds import LocalEmbeddings

# ----------------------
# 0️⃣ Configuration
# ----------------------


CACHE_FILE = "answers_cache.json"
PDF_PATH = "ENHANCING ENVIRONMENTAL SUSTAINABILITY IN ASIAN TEXTILE .pdf"
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\ProgramData\chocolatey\lib\poppler-25.12.0\Library\bin'
API_KEY = "AIzaSyD3b3sUEL89Z2SyMQxs0ONaP16WAk1BeMI"

# Optimization settings
BATCH_SIZE = 100 # Gemini supports batching up to 100 texts
SIMILARITY_THRESHOLD = 0.3  # Skip chunks below this threshold
DEDUPLICATION_THRESHOLD = 0.95  # Remove chunks more similar than this

# ----------------------
# 1️⃣ Configure Gemini
# ----------------------
os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])





# ----------------------
# 4️⃣ Semantic Deduplication
# ----------------------
def deduplicate_chunks(chunks: List[str], embeddings: np.ndarray, threshold: float = DEDUPLICATION_THRESHOLD) -> Tuple[List[str], np.ndarray, List[int]]:
    """
    Remove near-duplicate chunks to reduce API calls
    Returns deduplicated chunks, embeddings, and kept indices
    """
    if len(chunks) <= 1:
        return chunks, embeddings, list(range(len(chunks)))
    
    print(f"\n🔍 Deduplicating {len(chunks)} chunks (threshold: {threshold})...")
    
    kept_indices = [0]  # Always keep the first chunk
    kept_chunks = [chunks[0]]
    kept_embeddings = [embeddings[0]]
    
    for i in range(1, len(chunks)):
        # Calculate similarity with all kept chunks
        similarities = cosine_similarity(
            embeddings[i].reshape(1, -1),
            np.array(kept_embeddings)
        )[0]
        
        # If not too similar to any kept chunk, keep it
        if max(similarities) < threshold:
            kept_indices.append(i)
            kept_chunks.append(chunks[i])
            kept_embeddings.append(embeddings[i])
    
    removed = len(chunks) - len(kept_chunks)
    print(f"✅ Removed {removed} duplicate chunks, kept {len(kept_chunks)}")
    
    return kept_chunks, np.array(kept_embeddings), kept_indices

# ----------------------
# 5️⃣ Rate-Limited Gemini LLM
# ----------------------
class GeminiLLM:
    def __init__(self):
        # ✅ UPDATED: Using a model explicitly listed in your account
        self.model_name = "models/gemini-2.5-flash" 
        self.max_retries = 3
        self.last_call_time = None
        self.call_count = 0  # ✅ FIXED: Restored this to prevent AttributeError
        
    def generate(self, prompt: str) -> str:
        # 1. ESTIMATE TOKENS
        est_tokens = len(prompt) / 4
        print(f"📊 Sending approx {int(est_tokens)} tokens...")

        # 2. Safety cooldown
        if self.last_call_time:
            time_since_last = time.time() - self.last_call_time
            if time_since_last < 5:
                time.sleep(5 - time_since_last)

        for attempt in range(self.max_retries):
            try:
                model = genai.GenerativeModel(self.model_name)
                
                # Increment counter before call
                self.call_count += 1
                self.last_call_time = time.time()
                
                print(f"🤖 API Attempt {attempt + 1} (Model: {self.model_name})...")
                
                response = model.generate_content(prompt)
                return response.text
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error Detail: {error_msg}")
                
                # Rate Limit Handling
                if "429" in error_msg or "quota" in error_msg.lower():
                    print("⚠️ QUOTA HIT. Waiting 60 seconds...")
                    time.sleep(60)
                elif "400" in error_msg:
                    print("❌ TOKEN LIMIT or SAFETY FILTER hit.")
                    return "Error: Context too large or safety block."
                elif "404" in error_msg:
                    # Fallback to 'latest' if 2.5 fails for any reason
                    print("⚠️ Model specific version not found. Trying generic 'flash-latest'...")
                    self.model_name = "models/gemini-flash-latest"
        
        return "Failed to generate answer."
# ----------------------
# 6️⃣ Cache System
# ----------------------
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)



# ----------------------
# 9️⃣ Optimized Two-Stage Retrieval
# ----------------------
def optimized_two_stage_retrieval(
    query: str,
    chunks: List[str],
    minilm_embeddings: np.ndarray,
    minilm_embedder: LocalEmbeddings,
    gemini_embedder: OptimizedGeminiEmbeddings,
    top_k_stage1: int = 50,
    top_k_stage2: int = 5
) -> Tuple[List[str], List[float]]:
    """
    Optimized two-stage retrieval with:
    - Similarity threshold filtering
    - Semantic deduplication
    - Batch embedding
    """
    
    print(f"\n🔍 Stage 1: MiniLM Retrieval (top {top_k_stage1})")
    print("="*60)
    
    # Stage 1: MiniLM embeddings
    query_embedding_minilm = minilm_embedder.embed_query(query)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_embedding_minilm.reshape(1, -1),
        minilm_embeddings
    )[0]
    
    # Apply similarity threshold
    threshold_mask = similarities >= SIMILARITY_THRESHOLD
    filtered_indices = np.where(threshold_mask)[0]
    
    if len(filtered_indices) == 0:
        print(f"⚠️  No chunks above threshold {SIMILARITY_THRESHOLD}")
        filtered_indices = np.arange(len(similarities))
    
    filtered_similarities = similarities[filtered_indices]
    
    # Get top K from filtered
    top_k = min(top_k_stage1, len(filtered_indices))
    top_k_relative_indices = np.argsort(filtered_similarities)[-top_k:][::-1]
    top_k_indices = filtered_indices[top_k_relative_indices]
    
    top_k_chunks = [chunks[i] for i in top_k_indices]
    top_k_scores = [similarities[i] for i in top_k_indices]
    top_k_embeddings = minilm_embeddings[top_k_indices]
    
    print(f"✅ Filtered to {len(filtered_indices)} chunks above threshold")
    print(f"✅ Retrieved {len(top_k_chunks)} top chunks")
    print(f"   Score range: {min(top_k_scores):.4f} to {max(top_k_scores):.4f}")
    
    # Deduplication
    dedup_chunks, dedup_embeddings, kept_indices = deduplicate_chunks(
        top_k_chunks, 
        top_k_embeddings,
        threshold=DEDUPLICATION_THRESHOLD
    )
    
    # Stage 2: Gemini re-ranking
    print(f"\n🔍 Stage 2: Gemini Re-ranking (top {top_k_stage2})")
    print("="*60)
    
    # Embed query with Gemini (cached)
    query_embedding_gemini = gemini_embedder.embed_query(query)
    
    # Batch embed chunks with Gemini (cached + batched)
    print(f"🔄 Batch embedding {len(dedup_chunks)} chunks with Gemini...")
    chunk_embeddings_gemini = gemini_embedder.embed_texts_batch(dedup_chunks)
    
    # Calculate cosine similarity with Gemini embeddings
    similarities_gemini = cosine_similarity(
        query_embedding_gemini.reshape(1, -1),
        chunk_embeddings_gemini
    )[0]
    
    # Get final top K
    final_top_k = min(top_k_stage2, len(dedup_chunks))
    final_top_k_indices = np.argsort(similarities_gemini)[-final_top_k:][::-1]
    final_chunks = [dedup_chunks[i] for i in final_top_k_indices]
    final_scores = [similarities_gemini[i] for i in final_top_k_indices]
    
    print(f"✅ Final {len(final_chunks)} chunks selected")
    print(f"   Score range: {min(final_scores):.4f} to {max(final_scores):.4f}")
    
    return final_chunks, final_scores

# ----------------------
# 🔟 Ask Question with Optimized RAG
# ----------------------
def ask_question(
    query: str,
    chunks: List[str],
    minilm_embeddings: np.ndarray,
    minilm_embedder: LocalEmbeddings,
    gemini_embedder: OptimizedGeminiEmbeddings,
    llm: GeminiLLM,
    cache: dict
) -> str:
    """Ask question using optimized two-stage retrieval"""
    
    print(f"\n{'='*60}")
    print(f"📝 Question: {query}")
    print(f"{'='*60}")
    
    # Check cache
    if query in cache:
        print("💾 Found cached answer!")
        print(f"\n💡 Answer:\n{cache[query]}")
        return cache[query]
    
    # Optimized two-stage retrieval
    retrieved_chunks, scores = optimized_two_stage_retrieval(
        query=query,
        chunks=chunks,
        minilm_embeddings=minilm_embeddings,
        minilm_embedder=minilm_embedder,
        gemini_embedder=gemini_embedder,
        top_k_stage1=50,
        top_k_stage2=5
    )
    
    # Build context
    context = "\n\n".join([
        f"[Chunk {i+1}, Relevance: {scores[i]:.4f}]\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    ])
    
    # Create prompt
    prompt = f"""You are a helpful AI assistant analyzing an English textbook. 
Answer the question based on the provided context. 
Provide a detailed, comprehensive answer with explanations when appropriate.

Context:
{context}

Question: {query}

Answer:"""
    
    # Generate answer
    print(f"\n🤖 Generating answer with Gemini LLM...")
    try:
        answer = llm.generate(prompt)
        
        # Cache the answer
        cache[query] = answer
        save_cache(cache)
        
        print(f"\n💡 Answer:\n{answer}")
        return answer
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

# ----------------------
# 1️⃣1️⃣ Main Execution
# ----------------------
def main():
    print("="*60)
    print("🚀 OPTIMIZED Two-Stage RAG System")
    print("="*60)
    print("✨ Optimizations:")
    print("   • Batch Gemini embeddings (100 texts/call)")
    print("   • Cache Gemini embeddings permanently")
    print("   • Similarity threshold filtering")
    print("   • Semantic deduplication")
    print("   • Fast page detection with pdfinfo")
    print("="*60)
    
    # Load cache
    cache = load_cache()
    print(f"📦 Loaded {len(cache)} cached answers")
    
    # Load or create chunks with embeddings
    chunks, minilm_embeddings = LocalEmbeddings.load_chunks_and_embeddings()
    
    if chunks is None:
        # ✅ CLEAN: process_pdf handles page count automatically using pdfinfo!
        all_text = PDFProcess.process_pdf(PDF_PATH, POPPLER_PATH)
        chunks = PDFProcess.create_chunks(all_text)
        chunks, minilm_embeddings = LocalEmbeddings.embed_and_save_chunks(chunks)
    
    # Initialize models
    minilm_embedder = LocalEmbeddings()
    gemini_embedder = OptimizedGeminiEmbeddings()
    llm = GeminiLLM()
    
    print("\n" + "="*60)
    print("✅ Optimized RAG System Ready!")
    print("="*60)
    
    # Ask questions
    questions = [
        "What can a responsive system do to enhance environmental sustainability in the Asian textile and apparel industry?",
    ]
    
    for i, question in enumerate(questions):
        answer = ask_question(
            query=question,
            chunks=chunks,
            minilm_embeddings=minilm_embeddings,
            minilm_embedder=minilm_embedder,
            gemini_embedder=gemini_embedder,
            llm=llm,
            cache=cache
        )
        
        # Wait between questions
        if answer and i < len(questions) - 1:
            wait_time = 10
            print(f"\n⏳ Waiting {wait_time}s before next question...")
            time.sleep(wait_time)
    
    # Final stats
    print("\n" + "="*60)
    print(f"📊 Session Stats:")
    print(f"   Total LLM calls: {llm.call_count}")
    print(f"   Total Gemini embedding API calls: {gemini_embedder.call_count}")
    print(f"   Cached answers: {len(cache)}")
    print(f"   Cached embeddings: {len(gemini_embedder.cache)}")
    print("="*60)
    print(f"\n💡 Optimization Impact:")
    print(f"   • Without batching: ~50 API calls per query")
    print(f"   • With batching: ~1-2 API calls per query (first run)")
    print(f"   • With caching: 0 embedding calls (subsequent runs)")
    print("="*60)


if __name__ == "__main__":
    main()