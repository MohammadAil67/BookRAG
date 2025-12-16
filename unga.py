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

# ----------------------
# 0️⃣ Configuration
# ----------------------
CHUNKS_FILE = "chunks_with_embeddings.json"
GEMINI_EMBEDDINGS_CACHE = "gemini_embeddings_cache.json"
CACHE_FILE = "answers_cache.json"
PDF_PATH = "Dakhil - 2018 - Class-(9-10) English For Today PDF Web.pdf"
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\poppler\Library\bin'
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
# 2️⃣ Local MiniLM Embeddings
# ----------------------
class LocalEmbeddings:
    def __init__(self):
        print("📦 Loading MiniLM embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ MiniLM model loaded!")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts"""
        print(f"🔄 Embedding {len(texts)} texts with MiniLM...")
        return self.model.encode(texts, show_progress_bar=True)
    
    def embed_query(self, text: str) -> np.ndarray:
        """Embed single query"""
        return self.model.encode([text])[0]

# ----------------------
# 3️⃣ Optimized Gemini Embeddings with Caching & Batching
# ----------------------
class OptimizedGeminiEmbeddings:
    def __init__(self, cache_file=GEMINI_EMBEDDINGS_CACHE):
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
# 7️⃣ OCR PDF → Chunks
# ----------------------
def process_pdf(pdf_path, poppler_path, start_page=1, end_page=None):
    """
    Process PDF with OCR - automatically detects page count if not specified
    """
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    os.makedirs("page_images", exist_ok=True)
    
    # Auto-detect page count using pdfinfo if not specified
    if end_page is None:
        end_page = get_pdf_page_count_fast(pdf_path, poppler_path)
        if end_page is None:
            raise ValueError("Could not determine PDF page count using pdfinfo")
    
    all_text = []
    
    print(f"\n📄 Processing PDF with OCR: {pdf_path}")
    print(f"   Pages: {start_page} to {end_page}")
    
    for batch_start in range(start_page, end_page + 1, 5):
        batch_end = min(batch_start + 4, end_page)
        print(f"\n📖 Processing pages {batch_start} to {batch_end}...")
        
        images = convert_from_path(
            pdf_path,
            first_page=batch_start,
            last_page=batch_end,
            poppler_path=poppler_path,
            dpi=200
        )
        
        for i, image in enumerate(images):
            page_num = batch_start + i
            text = pytesseract.image_to_string(image, lang='eng')
            all_text.append(f"--- Page {page_num} ---\n{text}")
            print(f"  ✓ Page {page_num}: {len(text)} characters")
    
    return all_text

def create_chunks(all_text):
    """Split text into chunks"""
    print(f"\n🔧 Creating text chunks...")
    full_text = "\n\n".join(all_text)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(full_text)
    print(f"✅ Created {len(chunks)} text chunks")
    
    return chunks

# ----------------------
# 8️⃣ Embed and Save Chunks
# ----------------------
def embed_and_save_chunks(chunks):
    """Embed all chunks with MiniLM and save to JSON"""
    print(f"\n🔧 Embedding {len(chunks)} chunks with MiniLM...")
    
    embedder = LocalEmbeddings()
    embeddings = embedder.embed_texts(chunks)
    
    # Save chunks with embeddings
    data = {
        "chunks": chunks,
        "embeddings": embeddings.tolist()
    }
    
    with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    
    print(f"✅ Saved {len(chunks)} chunks with embeddings to {CHUNKS_FILE}")
    return chunks, embeddings

def load_chunks_and_embeddings():
    """Load pre-computed chunks and embeddings"""
    if not os.path.exists(CHUNKS_FILE):
        return None, None
    
    print(f"\n📦 Loading chunks and embeddings from {CHUNKS_FILE}...")
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data["chunks"]
    embeddings = np.array(data["embeddings"])
    
    print(f"✅ Loaded {len(chunks)} chunks with embeddings")
    return chunks, embeddings

def get_pdf_page_count_fast(pdf_path, poppler_path):
    """
    Fast page count using pdfinfo from Poppler (same tool used for OCR)
    No additional dependencies needed!
    
    Args:
        pdf_path: Path to the PDF file
        poppler_path: Path to poppler bin directory (e.g., r'C:\poppler\Library\bin')
    
    Returns:
        int: Number of pages, or None if failed
    """
    try:
        # Construct path to pdfinfo executable
        pdfinfo_exe = os.path.join(poppler_path, 'pdfinfo.exe')
        
        # Check if pdfinfo exists
        if not os.path.exists(pdfinfo_exe):
            print(f"❌ pdfinfo not found at: {pdfinfo_exe}")
            return None
        
        # Run pdfinfo command
        result = subprocess.run(
            [pdfinfo_exe, pdf_path],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        
        # Parse output for page count
        for line in result.stdout.split('\n'):
            if line.startswith('Pages:'):
                page_count = int(line.split(':')[1].strip())
                print(f"📊 Detected {page_count} total pages in PDF (using pdfinfo)")
                return page_count
        
        print("⚠️ Could not find 'Pages:' in pdfinfo output")
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"❌ pdfinfo command failed: {e}")
        print(f"   stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Error reading PDF metadata: {e}")
        return None

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
    chunks, minilm_embeddings = load_chunks_and_embeddings()
    
    if chunks is None:
        # ✅ CLEAN: process_pdf handles page count automatically using pdfinfo!
        all_text = process_pdf(PDF_PATH, POPPLER_PATH)
        chunks = create_chunks(all_text)
        chunks, minilm_embeddings = embed_and_save_chunks(chunks)
    
    # Initialize models
    minilm_embedder = LocalEmbeddings()
    gemini_embedder = OptimizedGeminiEmbeddings()
    llm = GeminiLLM()
    
    print("\n" + "="*60)
    print("✅ Optimized RAG System Ready!")
    print("="*60)
    
    # Ask questions
    questions = [
        "Give me a brief overview of ferry boat",
        "Solve the questions mentioned in meherjan and the greedy jamuna",
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