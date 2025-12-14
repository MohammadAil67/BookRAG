import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import shutil
from pdf2image import convert_from_path
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from sentence_transformers import SentenceTransformer
from typing import Any, List, Optional
import google.generativeai as genai

# ----------------------
# 0️⃣ Clear old Chroma database
# ----------------------
chroma_dir = "./chroma_db"
if os.path.exists(chroma_dir):
    shutil.rmtree(chroma_dir)
    print("🗑️ Cleared old Chroma database")

# ----------------------
# 1️⃣ Configure Gemini
# ----------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyD3b3sUEL89Z2SyMQxs0ONaP16WAk1BeMI"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ----------------------
# 2️⃣ Local Embeddings (No API limits!)
# ----------------------
class LocalEmbeddings(Embeddings):
    def __init__(self):
        print("Loading local embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded!")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print(f"Embedding {len(texts)} documents locally...")
        return self.model.encode(texts, show_progress_bar=True).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

# ----------------------
# 3️⃣ Gemini 2.5 Flash LLM with retry logic
# ----------------------
class GeminiLLM(LLM):
    model_name: str = "gemini-2.0-flash-exp"
    max_retries: int = 3
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        for attempt in range(self.max_retries):
            try:
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    if attempt < self.max_retries - 1:
                        wait_time = 60
                        print(f"⚠️ Rate limit hit. Waiting {wait_time} seconds... (Attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(wait_time)
                    else:
                        raise Exception(f"Rate limit exceeded. Please wait and try again later.")
                else:
                    raise
        return ""

# ----------------------
# 4️⃣ OCR PDF → chunked text
# ----------------------
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
poppler_path = r'C:\poppler\Library\bin'
pdf_path = "Dakhil - 2018 - Class-(9-10) English For Today PDF Web.pdf"

os.makedirs("page_images", exist_ok=True)

all_text = []
all_images = []

print("\n📄 Starting PDF processing...")
for start_page in range(1, 210, 5):
    end_page = min(start_page + 4, 210)
    print(f"Processing pages {start_page} to {end_page}...")
    
    images = convert_from_path(
        pdf_path,
        first_page=start_page,
        last_page=end_page,
        poppler_path=poppler_path,
        dpi=200
    )
    
    for i, image in enumerate(images):
        page_num = start_page + i
        image_filename = f"page_images/page_{page_num}.png"
        image.save(image_filename)
        all_images.append(image_filename)
        
        text = pytesseract.image_to_string(image, lang='eng')
        all_text.append(f"--- Page {page_num} ---\n{text}")
        print(f"  ✓ Page {page_num}: {len(text)} characters")

full_text = "\n\n".join(all_text)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(full_text)

print(f"\n✅ Created {len(chunks)} text chunks from {len(all_images)} page images")

docs = [Document(page_content=chunk) for chunk in chunks]

# ----------------------
# 5️⃣ Build vector store with LOCAL embeddings
# ----------------------
print("\n🔧 Building vector store with local embeddings...")

embeddings = LocalEmbeddings()

# Create new Chroma with explicit persist directory
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings,
    persist_directory=chroma_dir,
    collection_name="pdf_collection"
)
retriever = vectorstore.as_retriever()

print("✅ Vector store created successfully!")

# ----------------------
# 6️⃣ RAG chain setup
# ----------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the context below:

Context:
{context}

Question: {question}

Answer:"""
)

print("\n🤖 Initializing Gemini 2.0 Flash LLM...")
llm = GeminiLLM()

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ----------------------
# 7️⃣ Ask questions
# ----------------------
print("\n" + "="*60)
print("🎉 RAG System Ready! Using Gemini 2.0 Flash API")
print("="*60)

question = "What is this book about?"

print(f"\n📝 Question: {question}")
print("⏳ Getting answer from Gemini 2.0 Flash...")

try:
    answer = rag_chain.invoke(question)
    print(f"\n💡 Answer:\n{answer}")
except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "="*60)
print("✅ Done!")
print("="*60)
