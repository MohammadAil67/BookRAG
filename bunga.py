import time
import hashlib
import json
import numpy as np
from typing import List, Tuple, Optional, Dict
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PDFprocessing import PDFProcess
import os
import shutil
from pathlib import Path
import pickle


# ----------------------
# System Utilities
# ----------------------
class SystemUtils:
    """Utility methods for system path detection"""
    
    @staticmethod
    def find_tesseract() -> Optional[str]:
        """Auto-detect Tesseract installation"""
        if tesseract_path := os.getenv("TESSERACT_PATH"):
            return tesseract_path
        
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        
        for path in common_paths:
            if Path(path).exists():
                return path
        
        if tesseract_cmd := shutil.which("tesseract"):
            return tesseract_cmd
        
        return None

    @staticmethod
    def find_poppler() -> Optional[str]:
        """Auto-detect Poppler installation"""
        if poppler_path := os.getenv("POPPLER_PATH"):
            return poppler_path
        
        common_paths = [
            r'C:\ProgramData\chocolatey\lib\poppler\Library\bin',
            r'C:\ProgramData\chocolatey\lib\poppler-25.12.0\Library\bin',
        ]
        
        for path in common_paths:
            if Path(path).exists():
                return path
        
        try:
            choco_lib = Path(r'C:\ProgramData\chocolatey\lib')
            if choco_lib.exists():
                for folder in choco_lib.iterdir():
                    if folder.is_dir() and 'poppler' in folder.name.lower():
                        bin_path = folder / 'Library' / 'bin'
                        if bin_path.exists():
                            return str(bin_path)
        except (PermissionError, OSError):
            pass
        
        return None


# ----------------------
# Configuration Manager
# ----------------------
class Config:
    """Central configuration management"""
    
    def __init__(self, pdf_path: str = None, groq_api_key: str = None):
        # Paths
        self.PDF_PATH = pdf_path or os.getenv("PDF_PATH", "Dakhil - 2018 - Class-(9-10) English For Today PDF Web.pdf")
        self.TESSERACT_PATH = SystemUtils.find_tesseract()
        self.POPPLER_PATH = SystemUtils.find_poppler()
        self.GROQ_API_KEY = groq_api_key or os.getenv("GROQ_API_KEY", "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")
        
        # Validate
        if not self.TESSERACT_PATH:
            raise RuntimeError(
                "Tesseract not found. Install it or set TESSERACT_PATH environment variable.\n"
                "Download from: https://github.com/UB-Mannheim/tesseract/wiki"
            )
        
        if not self.POPPLER_PATH:
            raise RuntimeError(
                "Poppler not found. Install it or set POPPLER_PATH environment variable.\n"
                "Install via: choco install poppler"
            )
        
        # Settings
        self.TOP_K_CHUNKS = 3
        self.SIMILARITY_THRESHOLD = 0.3
        self.MAX_CONVERSATION_HISTORY = 3
        self.CONTEXT_CACHE_FILE = "context_cache.json"
        self.CONTEXT_CACHE_MAX_SIZE = 100
        
        # Generate PDF-specific filenames
        pdf_name = os.path.splitext(os.path.basename(self.PDF_PATH))[0]
        self.CHUNKS_FILE = f"{pdf_name}_chunks.pkl"
        self.EMBEDDINGS_FILE = f"{pdf_name}_bge_embeddings.pkl"


# ----------------------
# BGE Embeddings Manager
# ----------------------
class BGEEmbeddings:
    """Handles BGE-M3 embeddings with caching"""
    
    def __init__(self, chunks_file: str, embeddings_file: str):
        print("🔄 Loading BGE-M3 model...")
        self.model = SentenceTransformer('BAAI/BGE-M3')
        print("✅ BGE model loaded!")
        self.query_prefix = "Represent this sentence for searching relevant passages: "
        self.chunks_file = chunks_file
        self.embeddings_file = embeddings_file
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query"""
        prefixed_query = self.query_prefix + query
        embedding = self.model.encode(prefixed_query, normalize_embeddings=True)
        return embedding
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts"""
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return embeddings
    
    def load_chunks_and_embeddings(self) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
        """Load cached chunks and embeddings"""
        if os.path.exists(self.chunks_file) and os.path.exists(self.embeddings_file):
            print("📦 Loading cached chunks and embeddings...")
            with open(self.chunks_file, 'rb') as f:
                chunks = pickle.load(f)
            with open(self.embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"✅ Loaded {len(chunks)} chunks with BGE embeddings")
            return chunks, embeddings
        return None, None
    
    def save_chunks_and_embeddings(self, chunks: List[str], embeddings: np.ndarray):
        """Save chunks and embeddings to disk"""
        print("💾 Saving chunks and embeddings...")
        with open(self.chunks_file, 'wb') as f:
            pickle.dump(chunks, f)
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print("✅ Saved!")


# ----------------------
# Conversation History Manager
# ----------------------
class ConversationHistory:
    """Manages conversation history - stores entities and context separately"""
    
    def __init__(self, max_history: int = 3):
        self.history: List[Dict] = []
        self.last_entities: Dict[str, str] = {}
        self.max_history = max_history
    
    def add_exchange(self, question: str, answer: str):
        """Add Q&A to history"""
        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.time()
        })
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def update_entities(self, entities: Dict[str, str]):
        """Update tracked entities from the conversation"""
        self.last_entities.update(entities)
    
    def get_last_entity(self, entity_type: str = "person") -> Optional[str]:
        """Get the most recently mentioned entity of a type"""
        return self.last_entities.get(entity_type)
    
    def get_recent_context(self, max_exchanges: int = 2) -> str:
        """Get recent Q&A for entity tracking only"""
        if not self.history:
            return ""
        
        recent = self.history[-max_exchanges:]
        parts = []
        for ex in recent:
            parts.append(f"Q: {ex['question']}\nA: {ex['answer'][:200]}")
        
        return "\n\n".join(parts)
    
    def clear(self):
        """Clear conversation history"""
        self.history = []
        self.last_entities = {}
        print("🗑️  Conversation history cleared!")


# ----------------------
# Query Resolver
# ----------------------
class QueryResolver:
    """Resolves references in queries using conversation history"""
    
    def __init__(self, llm_client: Groq):
        self.client = llm_client
        self.model_name = "llama-3.3-70b-versatile"
    
    def needs_resolution(self, query: str) -> bool:
        """Check if query contains references that need resolution"""
        reference_words = ['he', 'she', 'it', 'they', 'this', 'that', 'these', 
                          'those', 'him', 'her', 'them', 'his', 'hers', 'their']
        query_lower = query.lower()
        return any(word in query_lower.split() for word in reference_words)
    
    def resolve_query(self, query: str, conversation_history: str) -> Tuple[str, Dict[str, str]]:
        """
        Resolve references in the query using conversation history.
        Returns: (resolved_query, extracted_entities)
        """
        if not self.needs_resolution(query):
            return query, {}
        
        try:
            prompt = f"""Given the conversation history and a new query with references (he/she/it/they/this/that), 
resolve the references to their actual entities.

Conversation History:
{conversation_history}

New Query: {query}

Provide:
1. RESOLVED_QUERY: The query with references replaced by actual entities
2. ENTITIES: Key entities mentioned (format: type=value, e.g., person=John)

Format your response as:
RESOLVED_QUERY: <resolved query here>
ENTITIES: <entity1=value1, entity2=value2>"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            
            resolved_query = query
            entities = {}
            
            for line in result.split('\n'):
                if line.startswith('RESOLVED_QUERY:'):
                    resolved_query = line.replace('RESOLVED_QUERY:', '').strip()
                elif line.startswith('ENTITIES:'):
                    entity_str = line.replace('ENTITIES:', '').strip()
                    if entity_str and entity_str != 'None':
                        for pair in entity_str.split(','):
                            if '=' in pair:
                                key, value = pair.split('=', 1)
                                entities[key.strip()] = value.strip()
            
            print(f"   Original: {query}")
            print(f"   Resolved: {resolved_query}")
            if entities:
                print(f"   Entities: {entities}")
            
            return resolved_query, entities
            
        except Exception as e:
            print(f"⚠️  Resolution failed: {e}, using original query")
            return query, {}


# ----------------------
# Context Cache
# ----------------------
class ContextCache:
    """Cache answers by resolved query + retrieved chunks"""
    
    def __init__(self, cache_file: str, max_size: int = 100):
        self.cache_file = cache_file
        self.max_size = max_size
        self.cache: Dict[str, str] = {}
        self.load_cache()
    
    def _make_key(self, resolved_query: str, chunk_ids: List[int]) -> str:
        """Generate cache key from resolved query and chunk IDs"""
        chunk_str = ','.join(map(str, sorted(chunk_ids)))
        combined = f"{resolved_query}|{chunk_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def load_cache(self):
        """Load cached answers from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"💾 Loaded {len(self.cache)} cached answers")
            except:
                self.cache = {}
    
    def save_cache(self):
        """Save cache to disk with size limit"""
        if len(self.cache) > self.max_size:
            items = list(self.cache.items())
            self.cache = dict(items[-self.max_size:])
        
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
    
    def get(self, resolved_query: str, chunk_ids: List[int]) -> Optional[str]:
        """Get cached answer"""
        cache_key = self._make_key(resolved_query, chunk_ids)
        return self.cache.get(cache_key)
    
    def set(self, resolved_query: str, chunk_ids: List[int], answer: str):
        """Set cached answer"""
        cache_key = self._make_key(resolved_query, chunk_ids)
        self.cache[cache_key] = answer
        self.save_cache()


# ----------------------
# Groq LLM
# ----------------------
class GroqLLM:
    """Groq LLM wrapper with rate limiting"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.3-70b-versatile"
    
    def generate(self, prompt: str) -> str:
        """Generate answer from prompt"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return "Sorry, I couldn't generate an answer."


# ----------------------
# Retrieval Engine
# ----------------------
class RetrievalEngine:
    """Handles semantic search and retrieval"""
    
    def __init__(self, embedder: BGEEmbeddings, top_k: int = 3, threshold: float = 0.3):
        self.embedder = embedder
        self.top_k = top_k
        self.threshold = threshold
    
    def retrieve(self, query: str, chunks: List[str], chunk_embeddings: np.ndarray) -> Tuple[List[str], List[int]]:
        """Retrieve top-k relevant chunks"""
        query_embedding = self.embedder.embed_query(query)
        
        # Calculate similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            chunk_embeddings
        )[0]
        
        # Filter by threshold
        threshold_mask = similarities >= self.threshold
        filtered_indices = np.where(threshold_mask)[0]
        
        if len(filtered_indices) == 0:
            print(f"⚠️  No chunks above threshold {self.threshold}")
            filtered_indices = np.arange(len(similarities))
        
        filtered_similarities = similarities[filtered_indices]
        
        # Get top K
        top_k = min(self.top_k, len(filtered_indices))
        top_k_relative_indices = np.argsort(filtered_similarities)[-top_k:][::-1]
        top_k_indices = filtered_indices[top_k_relative_indices]
        
        retrieved_chunks = [chunks[i] for i in top_k_indices]
        chunk_ids = top_k_indices.tolist()
        
        print(f"✅ Retrieved {len(retrieved_chunks)} chunks")
        
        return retrieved_chunks, chunk_ids


# ----------------------
# RAG System
# ----------------------
class RAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize components
        self.embedder = BGEEmbeddings(config.CHUNKS_FILE, config.EMBEDDINGS_FILE)
        self.llm = GroqLLM(config.GROQ_API_KEY)
        self.resolver = QueryResolver(Groq(api_key=config.GROQ_API_KEY))
        self.cache = ContextCache(config.CONTEXT_CACHE_FILE, config.CONTEXT_CACHE_MAX_SIZE)
        self.history = ConversationHistory(config.MAX_CONVERSATION_HISTORY)
        self.retrieval = RetrievalEngine(self.embedder, config.TOP_K_CHUNKS, config.SIMILARITY_THRESHOLD)
        
        # Load or create chunks
        self.chunks, self.embeddings = self._load_or_create_chunks()
    
    def _load_or_create_chunks(self) -> Tuple[List[str], np.ndarray]:
        """Load existing chunks or create from PDF"""
        chunks, embeddings = self.embedder.load_chunks_and_embeddings()
        
        if chunks is None:
            print("\n📄 Processing PDF...")
            all_text = PDFProcess.process_pdf(self.config.PDF_PATH, self.config.POPPLER_PATH)
            chunks = PDFProcess.create_chunks(all_text)
            
            print(f"\n🔄 Embedding {len(chunks)} chunks...")
            embeddings = self.embedder.embed_texts(chunks)
            
            self.embedder.save_chunks_and_embeddings(chunks, embeddings)
        
        return chunks, embeddings
    
    def ask(self, query: str) -> str:
        """Ask a question and get an answer"""
        print(f"\n{'='*60}")
        print(f"📝 Question: {query}")
        print(f"{'='*60}")
        
        # Resolve references
        conversation_context = self.history.get_recent_context()
        resolved_query, entities = self.resolver.resolve_query(query, conversation_context)
        self.history.update_entities(entities)
        
        # Retrieve chunks
        retrieved_chunks, chunk_ids = self.retrieval.retrieve(
            resolved_query,
            self.chunks,
            self.embeddings
        )
        
        # Check cache
        cached_answer = self.cache.get(resolved_query, chunk_ids)
        if cached_answer:
            print("💾 Found cached answer!")
            print(f"\n💡 Answer:\n{cached_answer}")
            self.history.add_exchange(query, cached_answer)
            return cached_answer
        
        # Build context
        context = "\n\n".join([
            f"[Chunk {i+1}]\n{chunk}"
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        # Generate answer
        prompt = f"""You are a helpful AI assistant analyzing an English textbook.
Answer the question based on the provided context.
Provide a clear, comprehensive answer.

Context:
{context}

Question: {resolved_query}

Answer:"""
        
        print(f"\n🤖 Generating answer...")
        answer = self.llm.generate(prompt)
        
        # Cache and save to history
        self.cache.set(resolved_query, chunk_ids, answer)
        self.history.add_exchange(query, answer)
        
        print(f"\n💡 Answer:\n{answer}")
        return answer
    
    def clear_history(self):
        """Clear conversation history"""
        self.history.clear()


# ----------------------
# Main Entry Point (for testing)
# ----------------------
def main():
    """Example usage"""
    print("="*60)
    print("🚀 RAG System with Reference Resolution")
    print("="*60)
    
    config = Config()
    rag = RAGSystem(config)
    
    # Interactive loop
    while True:
        print("\n" + "="*60)
        question = input("❓ Your question (or 'quit' to exit): ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        rag.ask(question)


if __name__ == "__main__":
    main()