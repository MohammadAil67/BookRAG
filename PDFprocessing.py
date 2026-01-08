import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
import pytesseract

TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\ProgramData\chocolatey\lib\poppler-25.12.0\Library\bin'

# PDF Processing with metadata tracking (chapters, page numbers, sections)

class ChunkMetadata:
    """Metadata for each text chunk"""
    def __init__(self, 
                 pdf_page: int,
                 book_page: Optional[int] = None,
                 chapter: Optional[str] = None,
                 section: Optional[str] = None,
                 chunk_id: Optional[str] = None):
        self.pdf_page = pdf_page
        self.book_page = book_page  # Actual page number in the book
        self.chapter = chapter
        self.section = section
        self.chunk_id = chunk_id
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'pdf_page': self.pdf_page,
            'book_page': self.book_page,
            'chapter': self.chapter,
            'section': self.section,
            'chunk_id': self.chunk_id
        }
    
    def __repr__(self):
        book_info = f"Book p.{self.book_page}" if self.book_page else "Book p.?"
        chapter_info = f" | {self.chapter}" if self.chapter else ""
        return f"[PDF p.{self.pdf_page} | {book_info}{chapter_info}]"


class PDFProcess:

    @staticmethod
    def get_pdf_page_count_fast(pdf_path, poppler_path):
        """Fast page count using pdfinfo from Poppler"""
        try:
            pdfinfo_exe = os.path.join(poppler_path, 'pdfinfo.exe')
            
            if not os.path.exists(pdfinfo_exe):
                print(f"[ERROR] pdfinfo not found at: {pdfinfo_exe}")
                return None
            
            result = subprocess.run(
                [pdfinfo_exe, pdf_path],
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8'
            )
            
            for line in result.stdout.split('\n'):
                if line.startswith('Pages:'):
                    page_count = int(line.split(':')[1].strip())
                    print(f"[INFO] Detected {page_count} total pages in PDF (using pdfinfo)")
                    return page_count
            
            print("[WARN] Could not find 'Pages:' in pdfinfo output")
            return None
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] pdfinfo command failed: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Error reading PDF metadata: {e}")
            return None

    @staticmethod
    def detect_language(text):
        """Detect if text contains Bangla or is English-only"""
        bangla_chars = sum(1 for ch in text if '\u0980' <= ch <= '\u09FF')
        english_chars = sum(1 for ch in text if ch.isalpha() and ord(ch) < 128)
        
        if bangla_chars > 10:
            return 'mixed' if english_chars > 10 else 'bangla'
        return 'english'

    @staticmethod
    def extract_chapter_from_text(text: str) -> Optional[str]:
        """
        Extract chapter name from text using various patterns
        Works for both English and Bengali
        """
        if not text:
            return None
        
        # English patterns
        patterns = [
            r'(?:Chapter|CHAPTER)\s*(\d+|[IVXLCDM]+)[\s:]*([^\n]+)',
            r'(?:Part|PART)\s*(\d+|[IVXLCDM]+)[\s:]*([^\n]+)',
            r'(?:Unit|UNIT)\s*(\d+)[\s:]*([^\n]+)',
            # Bengali patterns
            r'(?:অধ্যায়|পরিচ্ছেদ)\s*(\d+|[০-৯]+)[\s:]*([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:500], re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:
                    chapter_num = match.group(1).strip()
                    chapter_name = match.group(2).strip()
                    return f"Chapter {chapter_num}: {chapter_name}"
                else:
                    return f"Chapter {match.group(1).strip()}"
        
        return None

    @staticmethod
    def extract_book_page_number(text: str) -> Optional[int]:
        """
        Extract actual book page number from text
        Looks for standalone numbers at top/bottom of page
        """
        if not text:
            return None
        
        # Look in first and last 200 characters
        search_regions = [text[:200], text[-200:]]
        
        for region in search_regions:
            # Pattern: standalone number (likely a page number)
            # Usually at start of line, possibly with some spacing
            patterns = [
                r'^\s*(\d{1,4})\s*$',  # Just a number on its own line
                r'^\s*-\s*(\d{1,4})\s*-\s*$',  # -123-
                r'^\s*\|\s*(\d{1,4})\s*\|\s*$',  # |123|
                r'Page\s+(\d{1,4})',  # Page 123
                # Bengali
                r'পৃষ্ঠা\s*[০-৯\d]{1,4}',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, region, re.MULTILINE | re.IGNORECASE)
                if match:
                    try:
                        page_num = int(match.group(1))
                        # Sanity check: reasonable page number
                        if 1 <= page_num <= 9999:
                            return page_num
                    except (ValueError, IndexError):
                        continue
        
        return None

    @staticmethod
    def extract_section_heading(text: str) -> Optional[str]:
        """
        Extract section heading from text
        Looks for numbered sections or bold headings
        """
        if not text:
            return None
        
        # Look in first 300 characters
        search_text = text[:300]
        
        patterns = [
            r'^(\d+\.\d+)\s+([^\n]+)',  # 1.1 Introduction
            r'^([A-Z][^\n]{10,50})$',  # ALL CAPS HEADING
            # Bengali
            r'^([\u0980-\u09FF\s]{10,50})$',  # Bengali heading
        ]
        
        for pattern in patterns:
            match = re.search(pattern, search_text, re.MULTILINE)
            if match:
                section = match.group(0).strip()
                # Filter out common false positives
                if len(section) > 5 and not section.startswith('---'):
                    return section
        
        return None

    @staticmethod
    def clean_ocr_text(text, language='mixed'):
        """Clean OCR text with improved handling"""
        if not text or not text.strip():
            return ""
        
        # 1. Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 2. Handle Bangla-specific punctuation
        if language in ['bangla', 'mixed']:
            text = text.replace('||', '।')
            text = text.replace('|', '।') 
        
        # 3. Remove noise
        noise_chars = r'[~^`´¨]' 
        text = re.sub(noise_chars, '', text)
        
        # 4. Fix broken Unicode
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Preserve page markers
            if line.startswith('--- Page'):
                cleaned_lines.append(line)
                continue
            
            # Relaxed length check
            if len(line) < 2:
                continue
            
            # Relaxed alpha ratio
            valid_chars = sum(c.isalnum() or '\u0980' <= c <= '\u09FF' or c in '।.,' for c in line)
            if len(line) > 0:
                alpha_ratio = valid_chars / len(line)
                if alpha_ratio < 0.15: 
                    continue
            
            # Language-specific cleanup
            if language in ['english', 'mixed']:
                line = re.sub(r'\bl\b', 'I', line) 
                line = re.sub(r'\b0\b', 'O', line)
                line = re.sub(r'\s+[^\w\s\u0980-\u09FF.,!?;:()\[\]{}"\'\-।]\s+', ' ', line)
            
            # Fix spacing around punctuation
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
        """Process a single page with OCR and metadata extraction"""
        try:
            text = pytesseract.image_to_string(image, lang='eng+ben')
            language = 'mixed'
            
            # Extract metadata before cleaning
            chapter = PDFProcess.extract_chapter_from_text(text)
            book_page = PDFProcess.extract_book_page_number(text)
            section = PDFProcess.extract_section_heading(text)
            
            # Clean OCR text
            cleaned_text = PDFProcess.clean_ocr_text(text, language)
            
            return (page_num, cleaned_text, len(text), len(cleaned_text), chapter, book_page, section)
            
        except Exception as e:
            print(f"  [ERROR] Page {page_num} failed: {e}")
            return (page_num, "", 0, 0, None, None, None)

    @staticmethod
    def process_pdf(pdf_path, poppler_path, start_page=1, end_page=None, max_workers=None,
                   pdf_to_book_offset: int = 0):
        """
        Process PDF with OCR and metadata extraction
        
        Args:
            pdf_path: Path to PDF file
            poppler_path: Path to Poppler binaries
            start_page: First page to process (1-indexed)
            end_page: Last page to process (None = auto-detect)
            max_workers: Number of parallel threads
            pdf_to_book_offset: Offset between PDF pages and book pages
                               (e.g., if book starts at page 1 but PDF page 10, offset = -9)
        
        Returns:
            Tuple of (page_texts, page_metadata)
            - page_texts: List of text strings with page markers
            - page_metadata: Dict mapping page numbers to metadata
        """
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        os.makedirs("page_images", exist_ok=True)
        
        # Auto-detect optimal worker count
        if max_workers is None:
            cpu_threads = os.cpu_count() or 4
            max_workers = max(4, int(cpu_threads * 0.8))
        
        # Auto-detect page count
        if end_page is None:
            end_page = PDFProcess.get_pdf_page_count_fast(pdf_path, poppler_path)
            if end_page is None:
                raise ValueError("Could not determine PDF page count")
        
        all_results = {}
        page_metadata = {}  # Store metadata per page
        current_chapter = None
        
        print(f"\n[PDF] Processing PDF with OCR and Metadata Extraction: {pdf_path}")
        print(f"   Pages: {start_page} to {end_page}")
        print(f"   PDF-to-Book offset: {pdf_to_book_offset}")
        print(f"   Using {max_workers} parallel workers")
        
        batch_size = max(5, max_workers)
        
        for batch_start in range(start_page, end_page + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, end_page)
            print(f"\n[BATCH] Processing pages {batch_start} to {batch_end}...")
            
            # Convert batch to images
            images = convert_from_path(
                pdf_path,
                first_page=batch_start,
                last_page=batch_end,
                poppler_path=poppler_path,
                dpi=200
            )
            
            # Process pages in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, image in enumerate(images):
                    page_num = batch_start + i
                    future = executor.submit(PDFProcess._process_single_page, image, page_num)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    page_num, cleaned_text, orig_len, clean_len, chapter, book_page, section = future.result()
                    all_results[page_num] = cleaned_text
                    
                    # Update current chapter if found
                    if chapter:
                        current_chapter = chapter
                        print(f"  [📖] Page {page_num}: Found chapter '{chapter}'")
                    
                    # Calculate book page number
                    calculated_book_page = book_page if book_page else (page_num + pdf_to_book_offset)
                    
                    # Store metadata
                    page_metadata[page_num] = {
                        'pdf_page': page_num,
                        'book_page': calculated_book_page,
                        'chapter': current_chapter,
                        'section': section,
                        'extracted_page_num': book_page  # What we found in text
                    }
                    
                    page_info = f"Book p.{calculated_book_page}"
                    if chapter:
                        page_info += f" | {chapter}"
                    
                    print(f"  [OK] Page {page_num} ({page_info}): {orig_len} -> {clean_len} chars")
            
            del images
        
        # Reconstruct text in correct page order
        all_text = []
        for page_num in sorted(all_results.keys()):
            metadata = page_metadata[page_num]
            page_marker = f"--- Page {page_num} (Book p.{metadata['book_page']}"
            if metadata['chapter']:
                page_marker += f" | {metadata['chapter']}"
            page_marker += ") ---"
            
            all_text.append(f"{page_marker}\n{all_results[page_num]}")
        
        print(f"\n[DONE] Processed {len(all_text)} pages with metadata")
        print(f"   Chapters found: {len(set(m['chapter'] for m in page_metadata.values() if m['chapter']))}")
        
        return all_text, page_metadata
    
    @staticmethod
    def create_chunks_with_metadata(all_text, page_metadata, chunk_size=1000, chunk_overlap=200):
        """
        Split text into chunks while preserving metadata
        
        Args:
            all_text: List of text strings (one per page with markers)
            page_metadata: Dict mapping page numbers to metadata
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
        
        Returns:
            Tuple of (chunks, chunk_metadata)
            - chunks: List of text chunks
            - chunk_metadata: List of ChunkMetadata objects
        """
        print(f"\n[CHUNK] Creating text chunks with metadata...")
        
        # Create a mapping from character position to page metadata
        char_to_page = []
        current_pos = 0
        
        for page_text in all_text:
            # Extract page number from marker
            match = re.search(r'--- Page (\d+)', page_text)
            if match:
                page_num = int(match.group(1))
                page_len = len(page_text)
                char_to_page.append((current_pos, current_pos + page_len, page_num))
                current_pos += page_len + 2  # +2 for \n\n separator
        
        # Join all text
        full_text = "\n\n".join(all_text)
        
        # Create chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(full_text)
        
        # Assign metadata to each chunk
        chunk_metadata_list = []
        current_char = 0
        
        for i, chunk in enumerate(chunks):
            chunk_start = full_text.find(chunk, current_char)
            chunk_end = chunk_start + len(chunk)
            
            # Find which page(s) this chunk belongs to
            chunk_pages = []
            for start, end, page_num in char_to_page:
                # Check if chunk overlaps with this page
                if not (chunk_end <= start or chunk_start >= end):
                    chunk_pages.append(page_num)
            
            # Use the first (primary) page for metadata
            if chunk_pages:
                primary_page = chunk_pages[0]
                page_meta = page_metadata.get(primary_page, {})
                
                metadata = ChunkMetadata(
                    pdf_page=primary_page,
                    book_page=page_meta.get('book_page'),
                    chapter=page_meta.get('chapter'),
                    section=page_meta.get('section'),
                    chunk_id=f"chunk_{i:04d}"
                )
            else:
                # Fallback metadata if page not found
                metadata = ChunkMetadata(
                    pdf_page=0,
                    book_page=None,
                    chapter=None,
                    section=None,
                    chunk_id=f"chunk_{i:04d}"
                )
            
            chunk_metadata_list.append(metadata)
            current_char = chunk_start + 1
        
        print(f"[OK] Created {len(chunks)} chunks with metadata")
        print(f"   Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")
        
        # Show sample metadata
        if chunk_metadata_list:
            print(f"   Sample metadata: {chunk_metadata_list[0]}")
        
        return chunks, chunk_metadata_list

    @staticmethod
    def create_chunks(all_text, chunk_size=1000, chunk_overlap=200):
        """
        Backward compatibility: Split text into chunks without metadata
        (Old method signature)
        """
        print(f"\n[CHUNK] Creating text chunks (legacy mode)...")
        full_text = "\n\n".join(all_text)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(full_text)
        print(f"[OK] Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
        
        return chunks