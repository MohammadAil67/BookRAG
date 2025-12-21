import numpy as np
from typing import List, Tuple
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import Config

class GroqLLM:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.3-70b-versatile"
    
    def generate(self, prompt: str, temperature: float = 0.5) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM Error: {e}"

class Embedder:
    def __init__(self, config: Config):
        self.model = SentenceTransformer('BAAI/bge-m3')
        self.config = config

    def get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

class Reranker:
    def __init__(self, config: Config):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank(self, query: str, chunks: List[str], top_k: int) -> List[Tuple[int, float]]:
        if not chunks: return []
        
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.model.predict(pairs)
        
        results = []
        for i, score in enumerate(scores):
            results.append((i, float(score)))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]