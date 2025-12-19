# ----------------------
# 2️⃣ Local MiniLM Embeddings
# ----------------------
import json
import os
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


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
    
    @staticmethod
    def embed_and_save_chunks(chunks, chunks_file):
        """Embed all chunks with MiniLM and save to JSON"""
        print(f"\n🔧 Embedding {len(chunks)} chunks with MiniLM...")
        
        embedder = LocalEmbeddings()
        embeddings = embedder.embed_texts(chunks)
        
        # Save chunks with embeddings
        data = {
            "chunks": chunks,
            "embeddings": embeddings.tolist()
        }
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        
        print(f"✅ Saved {len(chunks)} chunks with embeddings to {chunks_file}")
        return chunks, embeddings
    
    @staticmethod
    def load_chunks_and_embeddings(chunks_file):
        """Load pre-computed chunks and embeddings"""
        if not os.path.exists(chunks_file):
            return None, None
        
        print(f"\n📦 Loading chunks and embeddings from {chunks_file}...")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data["chunks"]
        embeddings = np.array(data["embeddings"])
        
        print(f"✅ Loaded {len(chunks)} chunks with embeddings")
        return chunks, embeddings
