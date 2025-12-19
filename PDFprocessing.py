import os
import subprocess

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
import pytesseract

TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\ProgramData\chocolatey\lib\poppler-25.12.0\Library\bin'

#PDF Processing and chunking logic

class PDFProcess:

    @staticmethod
    def get_pdf_page_count_fast(pdf_path, poppler_path):
        r"""
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
        
    @staticmethod
    @staticmethod
    def process_pdf(pdf_path, poppler_path, start_page=1, end_page=None):
        """
        Process PDF with OCR - automatically detects page count if not specified
        """
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        os.makedirs("page_images", exist_ok=True)
        
        # Auto-detect page count using pdfinfo if not specified
        if end_page is None:
            end_page = PDFProcess.get_pdf_page_count_fast(pdf_path, poppler_path)
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
                text = pytesseract.image_to_string(image, lang='ben+eng')  # ← CHANGED THIS
                all_text.append(f"--- Page {page_num} ---\n{text}")
                print(f"  ✓ Page {page_num}: {len(text)} characters")
        
        return all_text
    
    @staticmethod
    def create_chunks(all_text):
        """Split text into chunks"""
        print(f"\n🔧 Creating text chunks...")
        full_text = "\n\n".join(all_text)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(full_text)
        print(f"✅ Created {len(chunks)} text chunks")
        
        return chunks
