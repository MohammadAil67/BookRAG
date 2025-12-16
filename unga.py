import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import shutil
import json
from datetime import datetime
import subprocess
from pdf2image import convert_from_path
import pytesseract
from langchain_chroma import Chroma 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from typing import Any, List, Optional
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ----------------------
# 0️⃣ Configuration
# ----------------------
CHROMA_DIR = "./chroma_db"
CACHE_FILE = "answers_cache.json"
PDF_PATH = "Dakhil - 2018 - Class-(9-10) English For Today PDF Web.pdf"
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\poppler\Library\bin'
API_KEY = "AIzaSyDhG3i6a1GAP04LKkgA5Plie4nZLbFuoZI"

# Function to safely clear Chroma database
def clear_chroma_db(chroma_dir, max_attempts=3):
    """Safely clear Chroma database with retry logic"""
    if not os.path.exists(chroma_dir):
        return
    
    for attempt in range(max_attempts):
        try:
            shutil.rmtree(chroma_dir)
            print("🗑️ Cleared old Chroma database")
            return
        except PermissionError as e:
            if attempt < max_attempts - 1:
                print(f"⚠️ Database locked, waiting 2 seconds... (Attempt {attempt + 1}/{max_attempts})")
                time.sleep(2)
            else:
                print(f"⚠️ Could not clear database (files in use). Will reuse existing database.")
                print(f"   To force clear: close all Python processes and delete '{chroma_dir}' manually")
                return

# ----------------------
# 1️⃣ Configure Gemini
# ----------------------
os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ----------------------
# 2️⃣ Google Gemini Embeddings
# ----------------------
class GeminiEmbeddings(Embeddings):
    def __init__(self):
        print("📦 Initializing Gemini embedding model...")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-001",
            google_api_key=API_KEY,
            task_type="retrieval_document"
        )
        print("✅ Gemini embedding model ready!")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print(f"🔄 Embedding {len(texts)} documents with Gemini...")
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

# ----------------------
# 3️⃣ Rate-Limited Gemini 2.5 Flash LLM
# ----------------------
class GeminiLLM(LLM):
    model_name: str = "models/gemini-2.5-flash"  # Latest stable model
    max_retries: int = 5
    min_delay_between_calls: float = 5.0  # 5 seconds = 12 requests/min (under 15 RPM limit)
    last_call_time: Optional[float] = None  # Declare as Pydantic field
    call_count: int = 0  # Declare as Pydantic field
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(self,prompt: str,stop: Optional[List[str]] = None,**kwargs: Any,) -> str:
        # Rate limiting: ensure minimum delay between calls
        if self.last_call_time:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_delay_between_calls:
                sleep_time = self.min_delay_between_calls - elapsed
                print(f"⏸️  Rate limiting: waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        for attempt in range(self.max_retries):
            try:
                # Use the correct model initialization
                model = genai.GenerativeModel(self.model_name)
                self.last_call_time = time.time()
                self.call_count += 1
                
                print(f"🤖 API Call #{self.call_count} (Attempt {attempt + 1}/{self.max_retries})")
                
                # Generate content with the model
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                    )
                )
                return response.text
                
            except Exception as e:
                error_msg = str(e)
                
                # Check for rate limit errors
                if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower() or "resource_exhausted" in error_msg.lower():
                    if attempt < self.max_retries - 1:
                        # Exponential backoff: 60s, 120s, 240s, 480s, 960s
                        wait_time = 60 * (2 ** attempt)
                        print(f"⚠️  RATE LIMIT HIT!")
                        print(f"   Waiting {wait_time}s before retry {attempt + 2}/{self.max_retries}...")
                        print(f"   Error: {error_msg}")
                        time.sleep(wait_time)
                    else:
                        print(f"\n❌ RATE LIMIT EXCEEDED - All retries exhausted")
                        print(f"📊 Total API calls made: {self.call_count}")
                        print(f"💡 Solutions:")
                        print(f"   1. Wait 1 hour for rate limit to reset")
                        print(f"   2. Check your quota at: https://aistudio.google.com/")
                        print(f"   3. Consider upgrading to paid tier for higher limits")
                        raise Exception(f"Rate limit exceeded after {self.max_retries} attempts. Please wait 1 hour.")
                else:
                    # Non-rate-limit error
                    print(f"❌ API Error: {error_msg}")
                    raise
        
        return ""

# ----------------------
# 4️⃣ Cache System
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
#  Dynamic PDF Page Count
# ----------------------

def get_pdf_page_count(pdf_path, poppler_path):
    """Get the total number of pages in a PDF"""
    try:
        pdfinfo_path = os.path.join(poppler_path, 'pdfinfo.exe')
        result = subprocess.run(
            [pdfinfo_path, pdf_path],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.split('\n'):
            if line.startswith('Pages:'):
                return int(line.split(':')[1].strip())
    except Exception as e:
        print(f"⚠️ Could not get page count: {e}")
        return None



# ----------------------
# 5️⃣ OCR PDF → chunked text
# ----------------------
def process_pdf(pdf_path, start_page=1, end_page=None): 
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    os.makedirs("page_images", exist_ok=True)

    if end_page is None:
        end_page = get_pdf_page_count(pdf_path, POPPLER_PATH)
        if end_page is None:
            raise ValueError("Could not determine PDF page count. Please specify end_page manually.")
        print(f"📊 Detected {end_page} total pages in PDF")    
    
    all_text = []
    all_images = []
    
    print(f"\n📄 Processing PDF: {pdf_path}")
    print(f"   Pages: {start_page} to {end_page}")


    
    for batch_start in range(start_page, end_page + 1, 5):
        batch_end = min(batch_start + 4, end_page)
        print(f"\n📖 Processing pages {batch_start} to {batch_end}...")
        
        images = convert_from_path(
            pdf_path,
            first_page=batch_start,
            last_page=batch_end,
            poppler_path=POPPLER_PATH,
            dpi=200
        )
        
        for i, image in enumerate(images):
            page_num = batch_start + i
            image_filename = f"page_images/page_{page_num}.png"
            image.save(image_filename)
            all_images.append(image_filename)
            
            text = pytesseract.image_to_string(image, lang='eng')
            all_text.append(f"--- Page {page_num} ---\n{text}")
            print(f"  ✓ Page {page_num}: {len(text)} characters")
    
    return all_text, all_images

# ----------------------
# 6️⃣ Build Vector Store
# ----------------------
def build_vectorstore(all_text):
    print(f"\n🔧 Creating text chunks...")
    full_text = "\n\n".join(all_text)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(full_text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    print(f"✅ Created {len(chunks)} text chunks")
    
    print(f"\n🔧 Building vector store with Gemini embeddings...")
    embeddings = GeminiEmbeddings()
    
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    print(f"✅ Vector store created with {len(chunks)} chunks")
    return vectorstore

# ----------------------
# 7️⃣ Setup RAG Chain
# ----------------------
def setup_rag_chain(vectorstore):
    # Reduce number of retrieved documents to minimize API calls
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}  # Only top 5 most relevant chunks
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the context below. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""
    )
    
    print("\n🤖 Initializing Gemini 2.5 Flash LLM with rate limiting...")
    llm = GeminiLLM()
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, llm

# ----------------------
# 8️⃣ Ask Questions with Caching
# ----------------------
def ask_question(rag_chain, question, cache):
    print(f"\n{'='*60}")
    print(f"📝 Question: {question}")
    print(f"{'='*60}")
    
    # Check cache first
    if question in cache:
        print("💾 Found cached answer!")
        print(f"\n💡 Answer:\n{cache[question]}")
        return cache[question]
    
    # Get new answer
    print("⏳ Getting answer from Gemini 2.0 Flash...")
    try:
        answer = rag_chain.invoke(question)
        
        # Save to cache
        cache[question] = answer
        save_cache(cache)
        
        print(f"\n💡 Answer:\n{answer}")
        return answer
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

# ----------------------
# 9️⃣ Main Execution
# ----------------------
def main():
    print("="*60)
    print("🚀 RAG System with Rate Limiting - Gemini 2.5 Flash")
    print("="*60)
    
    # Clear old database (optional - comment out to keep existing database)
    # clear_chroma_db(CHROMA_DIR)
    
    # Load cache
    cache = load_cache()
    print(f"📦 Loaded {len(cache)} cached answers")
    
    # Process PDF (or load existing vectorstore)
    if os.path.exists(CHROMA_DIR):
        print(f"\n✅ Found existing vector store at {CHROMA_DIR}")
        embeddings = GeminiEmbeddings()
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="pdf_collection"
        )
    else:
        all_text, all_images = process_pdf(PDF_PATH)
        print(f"\n✅ Processed {len(all_images)} pages")
        vectorstore = build_vectorstore(all_text)
    
    # Setup RAG chain
    rag_chain, llm = setup_rag_chain(vectorstore)
    
    print("\n" + "="*60)
    print("✅ RAG System Ready!")
    print("="*60)
    
    # Ask questions
    questions = [
        "What is this book about?",
        # Add more questions here if needed
    ]
    
    for i, question in enumerate(questions):
        answer = ask_question(rag_chain, question, cache)
        
        # Wait between questions if there are more
        if answer and i < len(questions) - 1:
            wait_time = 10
            print(f"\n⏳ Waiting {wait_time}s before next question...")
            time.sleep(wait_time)
    
    # Final stats
    print("\n" + "="*60)
    print(f"📊 Session Stats:")
    print(f"   Total API calls made: {llm.call_count}")
    print(f"   Cached answers: {len(cache)}")
    print("✅ Done!")
    print("="*60)

if __name__ == "__main__":
    main()