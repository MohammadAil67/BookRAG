import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
import pytesseract

TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\ProgramData\chocolatey\lib\poppler-25.12.0\Library\bin'

#PDF Processing and chunking logic with OCR cleanup and multithreading

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
                print(f"[ERROR] pdfinfo not found at: {pdfinfo_exe}")
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
                    print(f"[INFO] Detected {page_count} total pages in PDF (using pdfinfo)")
                    return page_count
            
            print("[WARN] Could not find 'Pages:' in pdfinfo output")
            return None
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] pdfinfo command failed: {e}")
            print(f"   stderr: {e.stderr}")
            return None
        except Exception as e:
            print(f"[ERROR] Error reading PDF metadata: {e}")
            return None

    @staticmethod
    def detect_language(text):
        """
        Detect if text contains Bangla or is English-only
        
        Returns:
            str: 'bangla', 'english', or 'mixed'
        """
        bangla_chars = sum(1 for ch in text if '\u0980' <= ch <= '\u09FF')
        english_chars = sum(1 for ch in text if ch.isalpha() and ord(ch) < 128)
        
        if bangla_chars > 10:
            return 'mixed' if english_chars > 10 else 'bangla'
        return 'english'

    @staticmethod
    def clean_ocr_text(text, language='mixed'):
        if not text or not text.strip():
            return ""
        
        # 1. Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 2. Handle Bangla-specific punctuation quirks BEFORE removing noise
        if language in ['bangla', 'mixed']:
            # Tesseract often mistakes the Bangla full stop (।) for a pipe (|) or double pipe (||)
            text = text.replace('||', '।')
            text = text.replace('|', '।') 
        
        # 3. Remove actual noise (Removed | from this list)
        noise_chars = r'[~^`´¨]' 
        text = re.sub(noise_chars, '', text)
        
        # 4. Fix broken Unicode
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            if not line: continue
            
            # Preserve page markers
            if line.startswith('--- Page'):
                cleaned_lines.append(line)
                continue
            
            # Relaxed length check
            if len(line) < 2: continue
            
            # Relaxed alpha ratio (0.15 is good for Bangla)
            valid_chars = sum(c.isalnum() or '\u0980' <= c <= '\u09FF' or c in '।.,' for c in line)
            if len(line) > 0:
                alpha_ratio = valid_chars / len(line)
                if alpha_ratio < 0.15: 
                    continue
            
            # Language-specific cleanup
            if language in ['english', 'mixed']:
                # Standard English fixes
                line = re.sub(r'\bl\b', 'I', line) 
                line = re.sub(r'\b0\b', 'O', line)
                
                # allow Bangla chars in the "standalone" check
                line = re.sub(r'\s+[^\w\s\u0980-\u09FF.,!?;:()\[\]{}"\'\-\।]\s+', ' ', line)
            
            # REMOVED the dangerous diacritic regex here
            
            # Fix spacing around punctuation (Included Bangla Danda ।)
            line = re.sub(r'\s+([.,!?;:।])', r'\1', line)
            line = re.sub(r'([.,!?;:।])\s*([^\s])', r'\1 \2', line)
            
            cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        cleaned_text = re.sub(r'([-_=*])\1{5,}', r'\1\1\1', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text

    @staticmethod
    def _process_single_page(image, page_num):
        """
        Process a single page with OCR (used by multithreading)
        
        Args:
            image: PIL Image object
            page_num: Page number
            
        Returns:
            Tuple of (page_num, cleaned_text, text_length, cleaned_length)
        """
        try:
                        
            text = pytesseract.image_to_string(image, lang='eng+ben')
            language = 'mixed'
            
            
            # 4. Clean OCR text
            cleaned_text = PDFProcess.clean_ocr_text(text, language)
            
            return (page_num, cleaned_text, len(text), len(cleaned_text))
            
        except Exception as e:
            print(f"  [ERROR] Page {page_num} failed: {e}")
            return (page_num, "", 0, 0)

    @staticmethod
    def process_pdf(pdf_path, poppler_path, start_page=1, end_page=None, max_workers=None):
        """
        Process PDF with OCR - dynamically detects English/Bangla per page
        Includes text cleanup and multithreading for faster processing
        
        Args:
            pdf_path: Path to PDF file
            poppler_path: Path to Poppler binaries
            start_page: First page to process (1-indexed)
            end_page: Last page to process (None = auto-detect)
            max_workers: Number of parallel threads (None = auto-detect optimal)
        
        Returns:
            List of text strings (one per page with page markers)
        """
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        os.makedirs("page_images", exist_ok=True)
        
        # Auto-detect optimal worker count if not specified
        if max_workers is None:
            cpu_threads = os.cpu_count() or 4
            # Use 80% of threads for OCR (OCR is CPU-bound and benefits from more threads)
            max_workers = max(4, int(cpu_threads * 0.8))
        
        # Auto-detect page count using pdfinfo if not specified
        if end_page is None:
            end_page = PDFProcess.get_pdf_page_count_fast(pdf_path, poppler_path)
            if end_page is None:
                raise ValueError("Could not determine PDF page count using pdfinfo")
        
        all_results = {}  # Store results with page numbers as keys
        
        print(f"\n[PDF] Processing PDF with OCR: {pdf_path}")
        print(f"   Pages: {start_page} to {end_page}")
        print(f"   CPU Info: {os.cpu_count()} threads detected")
        print(f"   Using {max_workers} parallel workers (optimized for OCR)")
        
        # OPTIMIZED: Smaller batches for better parallelization with high thread count
        # With 20 threads, we want to keep all threads busy with smaller batches
        batch_size = max(5, max_workers)  # At least as many pages as workers
        
        for batch_start in range(start_page, end_page + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, end_page)
            print(f"\n[BATCH] Processing pages {batch_start} to {batch_end}...")
            
            # Convert batch to images
            images = convert_from_path(
                pdf_path,
                first_page=batch_start,
                last_page=batch_end,
                poppler_path=poppler_path,
                dpi=200  # Consider increasing to 300 for better OCR quality if needed
            )
            
            # Process pages in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all pages in this batch
                futures = []
                for i, image in enumerate(images):
                    page_num = batch_start + i
                    future = executor.submit(PDFProcess._process_single_page, image, page_num)
                    futures.append(future)
                
                # Collect results as they complete
                for future in as_completed(futures):
                    page_num, cleaned_text, orig_len, clean_len = future.result()
                    all_results[page_num] = cleaned_text
                    print(f"  [OK] Page {page_num}: {orig_len} -> {clean_len} characters (cleaned)")
            
            # Memory cleanup
            del images
        
        # Reconstruct text in correct page order
        all_text = []
        for page_num in sorted(all_results.keys()):
            all_text.append(f"--- Page {page_num} ---\n{all_results[page_num]}")
        
        print(f"\n[DONE] Processed {len(all_text)} pages using {max_workers} workers")
        return all_text
    
    @staticmethod
    def create_chunks(all_text, chunk_size=1000, chunk_overlap=200):
        """
        Split text into chunks
        
        Args:
            all_text: List of text strings (one per page)
            chunk_size: Maximum size of each chunk (default: 1000)
            chunk_overlap: Overlap between chunks (default: 200)
        
        Returns:
            List of text chunks
        """
        print(f"\n[CHUNK] Creating text chunks...")
        full_text = "\n\n".join(all_text)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(full_text)
        print(f"[OK] Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
        
        return chunks