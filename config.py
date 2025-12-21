import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field

# --- CRITICAL FIX: Set Cache Path Globally ---
os.environ['HF_HOME'] = os.path.join(os.getcwd(), "model_cache")
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(os.getcwd(), "model_cache")

class SystemUtils:
    @staticmethod
    def find_tesseract() -> Optional[str]:
        if tesseract_path := os.getenv("TESSERACT_PATH"): return tesseract_path
        common_paths = [r'C:\Program Files\Tesseract-OCR\tesseract.exe', r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe']
        for path in common_paths:
            if Path(path).exists(): return path
        if tesseract_cmd := shutil.which("tesseract"): return tesseract_cmd
        return None

    @staticmethod
    def find_poppler() -> Optional[str]:
        if poppler_path := os.getenv("POPPLER_PATH"): return poppler_path
        common_paths = [r'C:\poppler\Library\bin', r'C:\ProgramData\chocolatey\lib\poppler\Library\bin']
        for path in common_paths:
            if Path(path).exists(): return path
        return None

class Config:
    def __init__(self, pdf_path: str = None, groq_api_key: str = None):
        self.PDF_PATH = pdf_path or os.getenv("PDF_PATH", "Dakhil - 2018 - Class-(9-10) English For Today PDF Web.pdf")
        self.GROQ_API_KEY = groq_api_key or os.getenv("GROQ_API_KEY", "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")
        
        # --- PATHS ---
        self.CACHE_DIR = "./cache"
        self.MODEL_CACHE_DIR = os.path.join(os.getcwd(), "model_cache")
        pdf_name = os.path.splitext(os.path.basename(self.PDF_PATH))[0]
        self.CHUNKS_FILE = f"{pdf_name}_chunks.pkl"
        self.EMBEDDINGS_FILE = f"{pdf_name}_bge_embeddings.pkl"
        
        # --- TUNING PARAMETERS ---
        self.INITIAL_RETRIEVAL_K = 15
        self.FINAL_TOP_K = 5
        self.RERANK_THRESHOLD = -2.0
        
        # --- LEGACY SUPPORT (For UI Compatibility) ---
        self.TOP_K_CHUNKS = self.FINAL_TOP_K
        self.SIMILARITY_THRESHOLD = 0.0 
        self.MAX_CONVERSATION_HISTORY = 10
        self.CONTEXT_CACHE_FILE = os.path.join(self.CACHE_DIR, "answer_cache.json")

class SimpleCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "answer_cache.json")
        self.cache = {} 
        self.answer_cache = self.cache 
        self.load()
    
    def load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache.update(json.load(f))
            except: pass
            
    def save(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except: pass
        
    def get(self, key):
        return self.cache.get(key)
        
    def set(self, key, value):
        self.cache[key] = value
        self.save()

@dataclass
class HistoryObject:
    history: List[Dict] = field(default_factory=list)
    last_entities: Dict = field(default_factory=dict)