import numpy as np
import requests
from typing import List, Tuple, Optional
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import Config
import os
import glob

# Try to import llama-cpp-python for local model support
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("⚠️ llama-cpp-python not installed. Local model fallback unavailable.")
    print("   Install with: pip install llama-cpp-python")


def get_offline_model_path(base_dir: str, model_name: str) -> str:
    """
    Helper to find the actual snapshot folder inside the HuggingFace cache.
    Resolves 'BAAI/bge-m3' -> '.../snapshots/HASH/...'
    """
    # 1. Check for standard HF Hub structure: models--org--repo
    safe_name = "models--" + model_name.replace("/", "--")
    hf_path = os.path.join(base_dir, safe_name)
    
    if os.path.exists(hf_path):
        # Look for snapshots
        snapshots_dir = os.path.join(hf_path, "snapshots")
        if os.path.exists(snapshots_dir):
            # Get the first folder in snapshots (usually the hash)
            subfolders = [f.path for f in os.scandir(snapshots_dir) if f.is_dir()]
            if subfolders:
                print(f"   📂 Resolved '{model_name}' to: {os.path.basename(subfolders[0])}")
                return subfolders[0]  # Return full path to the snapshot
    
    # 2. Check for direct folder (if manually extracted)
    direct_path = os.path.join(base_dir, model_name.replace("/", "_"))
    if os.path.exists(direct_path):
        return direct_path

    # 3. Fallback: Return original name (let library try to find it)
    return model_name


class GroqLLM:
    def __init__(self, api_key: str, local_model_path: Optional[str] = None):
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.3-70b-versatile"
        self.local_model_path = local_model_path
        self.local_model = None
        self.using_local = False
        
        # Check if local model exists and is valid
        if local_model_path and os.path.exists(local_model_path):
            if LLAMA_CPP_AVAILABLE:
                print(f"📦 Local model found at: {local_model_path}")
            else:
                print(f"⚠️ Local model found but llama-cpp-python not installed")
        elif local_model_path:
            print(f"⚠️ Local model not found at: {local_model_path}")
    
    def _check_internet(self) -> bool:
        """Check if internet connection is available"""
        try:
            requests.get("https://api.groq.com", timeout=3)
            return True
        except (requests.ConnectionError, requests.Timeout):
            return False
    
    def _load_local_model(self):
        """Load the local GGUF model (lazy loading)"""
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python not installed. Cannot use local model.")
        
        if not self.local_model_path or not os.path.exists(self.local_model_path):
            raise FileNotFoundError(f"Local model not found at: {self.local_model_path}")
        
        if self.local_model is None:
            print(f"🔄 Loading local model: {os.path.basename(self.local_model_path)}")
            print("   This may take a moment...")
            
            self.local_model = Llama(
                model_path=self.local_model_path,
                n_ctx=4096,  # Context window
                n_threads=8,  # CPU threads to use
                n_gpu_layers=-1,  # Set to >0 if you have GPU
                verbose=False
            )
            print("✅ Local model loaded successfully")
    
    def _generate_local(self, prompt: str, temperature: float = 0.5) -> str:
        """Generate response using local model"""
        try:
            self._load_local_model()
            
            response = self.local_model(
                prompt,
                max_tokens=1500,
                temperature=temperature,
                stop=["User:", "Question:"],  # Stop tokens
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            return f"Local Model Error: {e}"
    
    def generate(self, prompt: str, temperature: float = 0.5) -> str:
        """
        Generate response with automatic fallback:
        1. Try Groq API (online)
        2. Fall back to local model if offline
        """

        response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1500
        )
        return response.choices[0].message.content.strip()
        # Try online API first
        """""
        if not self.using_local and self._check_internet():
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1500
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"⚠️ Groq API failed: {e}")
                print("🔄 Attempting local fallback...")
        
        # Fallback to local model
        if self.local_model_path:
            if not self.using_local:
                print("🏠 Switching to LOCAL MODEL (offline mode)")
                self.using_local = True
            
            return self._generate_local(prompt, temperature)
        else:
            return "Error: No internet connection and no local model configured."
        """
    
    def force_online(self):
        """Force switch back to online mode"""
        self.using_local = False
        print("🌐 Forced switch to ONLINE mode")
    
    def force_local(self):
        """Force switch to local mode"""
        if not self.local_model_path:
            print("⚠️ No local model configured")
            return False
        self.using_local = True
        print("🏠 Forced switch to LOCAL mode")
        return True


class Embedder:
    def __init__(self, config: Config):
        """Initialize embedder with robust offline support"""
        self.config = config
        cache_dir = config.MODEL_CACHE_DIR
        
        # Determine strict offline mode
        is_offline = not self._check_internet()
        
        # 1. Resolve the ACTUAL path on disk
        # This is critical: We find the folder inside 'models--BAAI--bge-m3/snapshots/...'
        model_name_or_path = get_offline_model_path(cache_dir, 'BAAI/bge-m3')
        
        try:
            print(f"🔌 Loading Embedder from: {model_name_or_path}")
            
            # If we found a local path, load directly (no internet needed)
            if os.path.exists(model_name_or_path) and model_name_or_path != 'BAAI/bge-m3':
                self.model = SentenceTransformer(model_name_or_path, device='cpu')
            else:
                # Fallback to standard loading (might trigger download)
                self.model = SentenceTransformer(
                    'BAAI/bge-m3',
                    cache_folder=cache_dir,
                    local_files_only=is_offline
                )
            print(f"✅ Embedder loaded successfully")
            
        except Exception as e:
            print(f"❌ Embedder Load Error: {e}")
            if is_offline:
                print("⚠️ You seem to be offline and the model is not fully cached.")
                print("   Run 'python download_offline_models.py' once when online.")
            raise e

    def _check_internet(self) -> bool:
        try:
            requests.get("https://huggingface.co", timeout=2)
            return True
        except:
            return False

    def get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)


class Reranker:
    def __init__(self, config: Config):
        """Initialize reranker with robust offline support"""
        cache_dir = config.MODEL_CACHE_DIR
        model_name = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
        
        # 1. Resolve path using the helper function
        model_name_or_path = get_offline_model_path(cache_dir, model_name)
        
        try:
            print(f"🔌 Loading Reranker from: {model_name_or_path}")
            
            # FIXED: Removed 'automodel_args' and 'local_files_only'.
            # Since 'model_name_or_path' is now a real local path (D:\...),
            # the library automatically treats it as offline/local.
            self.model = CrossEncoder(
                model_name_or_path,
                max_length=512
            )
            print(f"✅ Reranker loaded successfully")
            
        except Exception as e:
            print(f"❌ Reranker Load Error: {e}")
            print("⚠️ Reranker offline mode failed. Run download_offline_models.py first!")
            raise e

    def rerank(self, query: str, chunks: List[str], top_k: int) -> List[Tuple[int, float]]:
        if not chunks: return []
        
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.model.predict(pairs)
        
        results = []
        for i, score in enumerate(scores):
            results.append((i, float(score)))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]