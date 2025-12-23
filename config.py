import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field



# --- CRITICAL FIX: Set Cache Path Globally ---
os.environ['HF_HOME'] = os.path.join(os.getcwd(), "model_cache")
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(os.getcwd(), "model_cache")

#This class just holds the path to find system paths for Tesseract and Poppler

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
        """Auto-detect Poppler installation"""
        # Check environment variable first
        if poppler_path := os.getenv("POPPLER_PATH"):
            if Path(poppler_path).exists():
                return poppler_path
        
        # Common static paths
        common_paths = [
            r'C:\poppler\Library\bin',
            r'C:\ProgramData\chocolatey\lib\poppler\Library\bin',
            r'C:\ProgramData\chocolatey\lib\poppler\tools\Library\bin',
            r'C:\Program Files\poppler\Library\bin',
        ]
        
        for path in common_paths:
            if Path(path).exists():
                pdfinfo = Path(path) / 'pdfinfo.exe'
                if pdfinfo.exists():
                    return path
        
        # Dynamic search in Chocolatey lib folder
        try:
            choco_lib = Path(r'C:\ProgramData\chocolatey\lib')
            if choco_lib.exists():
                for folder in choco_lib.iterdir():
                    if folder.is_dir() and 'poppler' in folder.name.lower():
                        # Check multiple possible bin locations
                        possible_bins = [
                            folder / 'Library' / 'bin',
                            folder / 'tools' / 'Library' / 'bin',
                            folder / 'tools' / 'bin',
                            folder / 'bin',
                        ]
                        for bin_path in possible_bins:
                            if bin_path.exists():
                                pdfinfo = bin_path / 'pdfinfo.exe'
                                if pdfinfo.exists():
                                    return str(bin_path)
        except (PermissionError, OSError):
            pass
        
        return None

#This class holds configuration parameters
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
        self.FINAL_TOP_K = 11
        self.RERANK_THRESHOLD = -2.0
        
        # --- LEGACY SUPPORT (For UI Compatibility) --- (Docile code)
        self.TOP_K_CHUNKS = self.FINAL_TOP_K
        self.SIMILARITY_THRESHOLD = 0.0 
        self.MAX_CONVERSATION_HISTORY = 10
        self.CONTEXT_CACHE_FILE = os.path.join(self.CACHE_DIR, "answer_cache.json")

#cache class with cache ordinal numbers


@dataclass
class HistoryObject:
    history: List[Dict] = field(default_factory=list)
    last_entities: Dict = field(default_factory=dict)