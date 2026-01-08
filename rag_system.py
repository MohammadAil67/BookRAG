"""
Enhanced RAG System - With Metadata and Local Model Fallback

Features:
- Chapter, page, and section metadata tracking
- Automatic fallback to local GGUF model when offline
- Citation validation and semantic topic tracking
- Smart metadata-aware retrieval
- Clean, accurate page citations
"""

import os
import re
import torch
import json
import pickle
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from config import Config, SystemUtils, HistoryObject
from models import GroqLLM, Embedder, Reranker
from retrieval import HybridRetriever, MultiQueryRetriever
from processing import PromptBuilder, AdvancedQueryRefiner, MultiQueryGenerator
from query_decomposition import QueryDecomposer, SmartContextApplier
from topics import EnhancedTopicTracker, CitationValidator


class RAGSystem:
    def __init__(self, config: Config):
        print("🚀 Initializing RAG System...")
        print("   🏠 Fallback: Local model support enabled")
        self.config = config
        
        # Initialize LLM with local model fallback support
        self.llm = GroqLLM(config.GROQ_API_KEY, local_model_path=config.LOCAL_MODEL_PATH)
        
        # 1. Load/Process PDF with metadata
        self.chunks, self.embeddings, self.chunk_metadata = self._load_data()
        
        # 2. Init Models
        self.embedder = Embedder(config)
        self.reranker = Reranker(config)
        
        # 3. Init Retrieval Stack
        base_retriever = HybridRetriever(
            self.chunks, self.embeddings, self.embedder, self.reranker, config
        )
        
        self.query_generator = MultiQueryGenerator(self.llm)
        self.retriever = MultiQueryRetriever(
            base_retriever=base_retriever,
            query_generator=self.query_generator,
            reranker=self.reranker,
            config=self.config,
            chunks=self.chunks
        )
        
        # 4. Query Intelligence
        self.decomposer = QueryDecomposer(self.llm)
        self.refiner = AdvancedQueryRefiner(self.llm)
        
        # 5. ENHANCED: Topic Management with semantic merging
        self.topic_tracker = EnhancedTopicTracker(self.embedder)
        self.context_applier = SmartContextApplier(self.topic_tracker)
        
        # 6. NEW: Citation Validator
        self.citation_validator = CitationValidator(self.embedder)
        
        # 7. State
        self.chat_history: List[Dict] = []
        self.history = HistoryObject()
        
        print("✅ Enhanced System Ready")

    def _load_data(self):
        """
        Loads chunks, embeddings, and metadata from cache or processes PDF
        """
        # Define cache files
        metadata_file = self.config.CHUNKS_FILE.replace('.pkl', '_metadata.pkl')
        
        # Check if cache exists
        if (os.path.exists(self.config.CHUNKS_FILE) and 
            os.path.exists(self.config.EMBEDDINGS_FILE) and
            os.path.exists(metadata_file)):
            try:
                print(f"📦 Loading cached data...")
                with open(self.config.CHUNKS_FILE, 'rb') as f: 
                    chunks = pickle.load(f)
                with open(self.config.EMBEDDINGS_FILE, 'rb') as f: 
                    embeddings = pickle.load(f)
                with open(metadata_file, 'rb') as f:
                    chunk_metadata = pickle.load(f)
                
                print(f"📚 Loaded {len(chunks)} chunks with metadata from cache")
                
                # Show sample metadata
                if chunk_metadata:
                    print(f"   Sample: {chunk_metadata[0]}")
                
                return chunks, embeddings, chunk_metadata
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}. Re-processing...")

        # Process PDF
        print("\n📄 No cache found. Processing PDF with metadata extraction...")
        try:
            from PDFprocessing import PDFProcess
            
            # Get Poppler path
            poppler_path = SystemUtils.find_poppler()
            if not poppler_path:
                raise ValueError("Poppler not found. Please install Poppler.")
            
            print(f"🔍 Using Poppler at: {poppler_path}")
            
            # Ask user for PDF-to-book page offset
            print("\n" + "="*60)
            print("📖 PDF TO BOOK PAGE MAPPING")
            print("="*60)
            print("If your book starts at page 1 but the PDF starts at page 10,")
            print("the offset would be -9 (to map PDF page 10 to book page 1).")
            print("\nExamples:")
            print("  - Book starts at PDF page 1: offset = 0")
            print("  - Book starts at PDF page 10: offset = -9")
            print("  - Book starts before PDF: offset = +5 (if book p.6 = PDF p.1)")
            
            try:
                offset_input = input("\nEnter PDF-to-Book page offset [default: 0]: ").strip()
                pdf_to_book_offset = int(offset_input) if offset_input else 0
            except ValueError:
                print("⚠️ Invalid input, using offset = 0")
                pdf_to_book_offset = 0
            
            print(f"✓ Using offset: {pdf_to_book_offset}")
            print("="*60 + "\n")
            
            # Process PDF with metadata extraction
            text_pages, page_metadata = PDFProcess.process_pdf(
                self.config.PDF_PATH, 
                poppler_path,
                pdf_to_book_offset=pdf_to_book_offset
            )
            
            # Create chunks with metadata
            chunks, chunk_metadata = PDFProcess.create_chunks_with_metadata(
                text_pages, 
                page_metadata
            )
            
            print(f"✅ Extracted {len(chunks)} chunks with metadata")
            
        except ImportError as e:
            print(f"⚠️ PDFprocessing module error: {e}")
            print("⚠️ Using dummy data for testing...")
            text = ["Dummy text content for testing purposes."]
            chunks = text
            chunk_metadata = [None] * len(chunks)
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            raise
        
        try:
            print("⚙️ Generating embeddings (Quantized CPU Optimized)...")
            
            # Load and quantize model
            temp_embedder = SentenceTransformer(
                'BAAI/bge-m3',
                cache_folder=self.config.MODEL_CACHE_DIR,
                local_files_only=False 
            )
            
            temp_embedder[0].auto_model = torch.quantization.quantize_dynamic(
                temp_embedder[0].auto_model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            
            print("   ⚡ Model quantized to INT8 for speed")

            # Encode
            embeddings = temp_embedder.encode(
                chunks,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            print(f"✅ Generated {len(embeddings)} embeddings")

        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            raise
        
        # Save to cache
        print("\n💾 Saving to cache...")
        try:
            os.makedirs(os.path.dirname(self.config.CHUNKS_FILE) or '.', exist_ok=True)
            
            with open(self.config.CHUNKS_FILE, 'wb') as f: 
                pickle.dump(chunks, f)
            print(f"   ✓ Saved chunks to {self.config.CHUNKS_FILE}")
            
            with open(self.config.EMBEDDINGS_FILE, 'wb') as f: 
                pickle.dump(embeddings, f)
            print(f"   ✓ Saved embeddings to {self.config.EMBEDDINGS_FILE}")
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(chunk_metadata, f)
            print(f"   ✓ Saved metadata to {metadata_file}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save cache: {e}")
        
        return chunks, embeddings, chunk_metadata

    def _format_chunk_with_metadata(self, chunk_idx: int) -> str:
        """Format a chunk with its metadata for display in prompts"""
        chunk = self.chunks[chunk_idx]
        metadata = self.chunk_metadata[chunk_idx] if chunk_idx < len(self.chunk_metadata) else None
        
        if metadata and metadata.book_page:
            # Simple, clean header with just page number
            header = f"[Page {metadata.book_page}]\n"
            return header + chunk
        else:
            return f"[Chunk {chunk_idx}]\n" + chunk

    def _get_primary_pages(self, chunk_indices: List[int], limit: int = 5) -> List[int]:
        """Get the primary page numbers from retrieved chunks"""
        if not self.chunk_metadata:
            return []
        
        pages = []
        for idx in chunk_indices[:limit]:  # Only check top chunks
            if idx < len(self.chunk_metadata):
                meta = self.chunk_metadata[idx]
                if meta and meta.book_page and meta.book_page > 0:
                    pages.append(meta.book_page)
        
        return sorted(set(pages))

    def _get_metadata_summary(self, chunk_indices: List[int]) -> str:
        """Get a clean summary of metadata from retrieved chunks"""
        if not self.chunk_metadata:
            return ""
        
        chapters = set()
        pages = set()
        
        for idx in chunk_indices:
            if idx < len(self.chunk_metadata):
                meta = self.chunk_metadata[idx]
                if meta:
                    # Only add valid chapter names (filter out fragments)
                    if meta.chapter and len(meta.chapter) > 15:  # Ignore short fragments
                        # Clean up chapter names
                        chapter = meta.chapter.strip()
                        # Remove common OCR artifacts
                        if not any(x in chapter.lower() for x in ['|', 'ici:', 'ic:', 'famousas', 'best movie']):
                            chapters.add(chapter)
                    
                    # Add book page numbers
                    if meta.book_page and meta.book_page > 0:
                        pages.add(meta.book_page)
        
        summary_parts = []
        
        # Format chapters
        if chapters:
            clean_chapters = sorted(chapters)[:3]  # Limit to top 3 chapters
            summary_parts.append(f"Chapter(s): {', '.join(clean_chapters)}")
        
        # Format pages
        if pages:
            page_list = sorted(pages)
            if len(page_list) <= 5:
                summary_parts.append(f"Page(s): {', '.join(map(str, page_list))}")
            else:
                # Show range for many pages
                summary_parts.append(f"Pages: {page_list[0]}-{page_list[-1]}")
        
        return " | ".join(summary_parts) if summary_parts else ""

    def generate_quiz(self, topic: str, difficulty: str, num_questions: int) -> List[Dict]:
        """
        Generates a structured quiz in JSON format based on the topic and PDF content
        Now includes source citations with metadata
        """
        print(f"🎲 Generating {difficulty} quiz for topic: '{topic}'")
        
        # 1. Retrieve relevant content
        search_query = f"facts concepts details about {topic}"
        chunk_indices = self.retriever.retrieve(search_query, [])
        
        # Format chunks with metadata
        formatted_chunks = [self._format_chunk_with_metadata(i) for i in chunk_indices[:5]]
        context_text = "\n\n".join(formatted_chunks)
        
        # Get primary pages
        primary_pages = self._get_primary_pages(chunk_indices[:5])
        if primary_pages:
            print(f"   📖 Retrieved from pages: {', '.join(map(str, primary_pages))}")
        
        # 2. Construct the strict JSON prompt
        prompt = f"""
Based strictly on the provided text context, generate {num_questions} multiple-choice questions (MCQs) about "{topic}".
Difficulty Level: {difficulty}

CONTEXT:
{context_text}

OUTPUT FORMAT:
You must return a valid JSON array of objects. Do not wrap in markdown code blocks. Do not add introductory text.

JSON Structure:
[
    {{
        "id": 1,
        "question": "Question text here?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer": "Option A",
        "explanation": "Brief explanation from text."
    }}
]

RULES:
1. "correct_answer" must be an exact string match to one of the "options".
2. "explanation" must explain WHY the answer is correct based on the text.
3. Do not use outside knowledge; strictly use the provided context.
4. If the topic is mathematical and there are mathematical data in the context, include at least one math-related question.
5. For math questions, the answer should involve calculation, not direct text lookup.
"""

        try:
            response = self.llm.generate(prompt)
            cleaned_response = re.sub(r'```json\s*|\s*```', '', response).strip()
            quiz_data = json.loads(cleaned_response)
            return quiz_data
            
        except json.JSONDecodeError:
            print("⚠️ Failed to parse Quiz JSON. Returning empty list.")
            print(f"Raw output: {response[:200]}...")
            return []
        except Exception as e:
            print(f"⚠️ Quiz generation error: {e}")
            return []

    # --- MAIN PIPELINE ---
    def ask(self, query: str) -> str:
        """Main query pipeline with intelligent routing and metadata"""
        print(f"\n🔍 Processing: '{query}'")
        
        should_decompose, decomp_type = self.decomposer.should_decompose(query)
        
        if should_decompose:
            print(f"  🔀 Complex query detected: {decomp_type}")
            return self._ask_with_decomposition(query)
        else:
            print(f"  📝 Simple query - using standard pipeline")
            return self._ask_simple(query)
    
    def _ask_simple(self, query: str) -> str:
        """Standard pipeline with citation validation and clean metadata"""
        topic_context = self.context_applier.get_smart_context(query)
        
        if topic_context:
            print(f"  📌 Applying context: '{topic_context}'")
        
        refined_query = self.refiner.refine(query, self.chat_history, topic_context)
        
        if refined_query != query:
            print(f"  🔧 Refined to: '{refined_query}'")
        
        chunk_indices = self.retriever.retrieve(refined_query, self.chat_history)
        retrieved_chunks = [self.chunks[i] for i in chunk_indices]
        
        # Get clean page numbers from top chunks
        primary_pages = self._get_primary_pages(chunk_indices, limit=5)
        
        print(f"  📦 Retrieved {len(retrieved_chunks)} chunks")
        if primary_pages:
            print(f"  📖 Primary sources: Pages {', '.join(map(str, primary_pages))}")
        
        if not retrieved_chunks:
            return "I couldn't find any relevant information in the document."
        
        # Format chunks with simple page headers
        formatted_chunks = [self._format_chunk_with_metadata(i) for i in chunk_indices]
        
        prompt = PromptBuilder.build(refined_query, formatted_chunks, self.chat_history)
        answer = self.llm.generate(prompt)
        
        is_grounded, supporting_chunks, confidence = \
            self.citation_validator.validate_answer(answer, retrieved_chunks)
        
        print(f"  📊 Grounding Score: {confidence:.2%}")
        
        if not is_grounded:
            print(f"  ⚠️ Low grounding detected. Regenerating...")
            strict_prompt = prompt + "\n\nCRITICAL INSTRUCTION: The previous answer was rejected because it included information not found in the text. You must cite specific details from the context provided above."
            answer = self.llm.generate(strict_prompt)
        
        # Append clean source citation
        show_sources = getattr(self.config, 'SHOW_SOURCES', True)
        if primary_pages and show_sources:
            if len(primary_pages) == 1:
                answer += f"\n\n*Source: Page {primary_pages[0]}*"
            elif len(primary_pages) <= 3:
                answer += f"\n\n*Sources: Pages {', '.join(map(str, primary_pages))}*"
            else:
                answer += f"\n\n*Sources: Pages {primary_pages[0]}-{primary_pages[-1]}*"
        
        self._update_history(query, answer)
        self.topic_tracker.update(query, answer)
        
        return answer
    
    def _ask_with_decomposition(self, query: str) -> str:
        """Handle complex queries with citation validation and clean metadata"""
        decomp_result = self.decomposer.decompose(query)
        sub_queries = decomp_result.sub_queries
        
        print(f"  📋 Decomposed into {len(sub_queries)} sub-queries:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"     {i}. {sq}")
        
        all_chunk_indices = set()
        for i, sub_q in enumerate(sub_queries, 1):
            print(f"  🔎 Retrieving for sub-query {i}/{len(sub_queries)}...")
            chunk_indices = self.retriever.retrieve(sub_q, self.chat_history)
            all_chunk_indices.update(chunk_indices)
        
        unique_chunks = [self.chunks[i] for i in all_chunk_indices]
        primary_pages = self._get_primary_pages(list(all_chunk_indices), limit=5)
        
        print(f"  📦 Total unique chunks: {len(unique_chunks)}")
        if primary_pages:
            print(f"  📖 Sources: Pages {', '.join(map(str, primary_pages))}")
        
        if not unique_chunks:
            return "I couldn't find any relevant information in the document."
        
        if len(unique_chunks) > self.config.FINAL_TOP_K:
            print(f"  🎯 Reranking {len(unique_chunks)} chunks...")
            reranked = self.reranker.rerank(query, unique_chunks, self.config.FINAL_TOP_K)
            final_indices = [list(all_chunk_indices)[idx] for idx, score in reranked]
            final_chunks = [unique_chunks[idx] for idx, score in reranked]
        else:
            final_indices = list(all_chunk_indices)
            final_chunks = unique_chunks
        
        # Format with metadata
        formatted_chunks = [self._format_chunk_with_metadata(i) for i in final_indices]
        
        prompt = self._build_decomposed_prompt(
            original_query=query,
            sub_queries=sub_queries,
            chunks=formatted_chunks,
            decomp_type=decomp_result.decomposition_type
        )
        
        answer = self.llm.generate(prompt)
        
        is_grounded, supporting_chunks, cite_confidence = \
            self.citation_validator.validate_answer(answer, final_chunks)
        
        print(f"  ✅ Citation confidence: {cite_confidence:.2%}")
        
        if not is_grounded:
            print(f"  ⚠️ Low grounding. Regenerating...")
            prompt += "\n\nCRITICAL: Answer ONLY using provided context. Cite specific details."
            answer = self.llm.generate(prompt)
        
        # Append clean source citation
        show_sources = getattr(self.config, 'SHOW_SOURCES', True)
        if primary_pages and show_sources:
            if len(primary_pages) == 1:
                answer += f"\n\n*Source: Page {primary_pages[0]}*"
            elif len(primary_pages) <= 3:
                answer += f"\n\n*Sources: Pages {', '.join(map(str, primary_pages))}*"
            else:
                answer += f"\n\n*Sources: Pages {primary_pages[0]}-{primary_pages[-1]}*"
        
        self._update_history(query, answer)
        self.topic_tracker.update(query, answer)
        
        return answer
    
    def _build_decomposed_prompt(self, original_query: str, 
                                  sub_queries: List[str], 
                                  chunks: List[str],
                                  decomp_type: str) -> str:
        """Build specialized prompt for decomposed queries"""
        
        context_text = "\n\n".join([
            f"[Context {i+1}]\n{chunk}" 
            for i, chunk in enumerate(chunks)
        ])
        
        history_text = ""
        if self.chat_history:
            recent = self.chat_history[-3:]
            history_text = "\n".join([
                f"User: {h['user']}\nAssistant: {h['ai']}" 
                for h in recent
            ])
        
        if decomp_type == 'comparison':
            special_instructions = """
For comparison questions:
1. Clearly describe each entity being compared
2. Highlight key similarities
3. Highlight key differences
4. Provide specific examples from the context
5. Organize your answer logically"""
        
        elif decomp_type == 'analytical':
            special_instructions = """
For analytical questions:
1. Break down the topic systematically
2. Address each aspect mentioned in the sub-questions
3. Provide evidence from the context
4. Draw connections between different aspects
5. Conclude with a synthesis"""
        
        else:
            special_instructions = """
1. Address all aspects comprehensively
2. Organize information logically
3. Use specific details from the context
4. Ensure all sub-questions are answered"""
        
        prompt = f"""You are answering a complex question broken down for analysis.

Original Question: {original_query}

Sub-questions:
{chr(10).join(f"{i+1}. {sq}" for i, sq in enumerate(sub_queries))}

Context from the document (with source metadata):
{context_text}

Previous conversation:
{history_text if history_text else "None"}

Instructions:
{special_instructions}

CRITICAL RULES:
- Use ONLY information from the provided context
- Do not make up or infer information not in context
- If context lacks information, acknowledge this
- Be specific and cite relevant details
- Maintain clear, organized structure

Answer:"""
        
        return prompt
    
    def _update_history(self, question: str, answer: str):
        """Update conversation history"""
        self.chat_history.append({"user": question, "ai": answer})
        self.history.history.append({"question": question, "answer": answer})
        
        if len(self.chat_history) > self.config.MAX_CONVERSATION_HISTORY:
            self.chat_history = self.chat_history[-self.config.MAX_CONVERSATION_HISTORY:]

    def clear_history(self):
        """Clear conversation history and reset enhanced topic tracker"""
        self.chat_history = []
        self.history.history = []
        self.topic_tracker = EnhancedTopicTracker(self.embedder)
        self.context_applier = SmartContextApplier(self.topic_tracker)
        print("🗑️ History cleared")

    def get_system_stats(self) -> Dict:
        """More comprehensive statistics including model status and metadata"""
        current_topic = self.topic_tracker.get_current_topic()
        
        # Check which model is being used
        model_status = "🌐 Online (Groq API)" if not self.llm.using_local else "🏠 Local (Offline)"
        
        # Count chapters
        unique_chapters = set()
        if self.chunk_metadata:
            for meta in self.chunk_metadata:
                if meta and meta.chapter:
                    unique_chapters.add(meta.chapter)
        
        return {
            'total_chunks': len(self.chunks),
            'conversation_turns': len(self.chat_history),
            'topics_discussed': len(self.topic_tracker.get_all_topics()),
            'current_topic': current_topic.name if current_topic else None,
            'current_topic_confidence': current_topic.confidence if current_topic else 0.0,
            'entity_mentions': dict(current_topic.entity_mentions) if current_topic else {},
            'last_intent': self.topic_tracker.last_intent,
            'total_entities_tracked': len(self.topic_tracker.entity_history),
            'model_status': model_status,
            'local_model_available': self.llm.local_model_path is not None,
            'chapters_in_document': len(unique_chapters),
            'has_metadata': bool(self.chunk_metadata)
        }
    
    def switch_to_local(self):
        """Manually switch to local model"""
        return self.llm.force_local()
    
    def switch_to_online(self):
        """Manually switch to online model"""
        self.llm.force_online()
    
    def search_by_metadata(self, chapter: str = None, page_range: Tuple[int, int] = None) -> List[int]:
        """
        Search chunks by metadata
        
        Args:
            chapter: Chapter name to filter by
            page_range: Tuple of (start_page, end_page) to filter by
        
        Returns:
            List of chunk indices matching criteria
        """
        if not self.chunk_metadata:
            return []
        
        matching_indices = []
        
        for i, meta in enumerate(self.chunk_metadata):
            if not meta:
                continue
            
            # Check chapter match
            if chapter and meta.chapter and chapter.lower() in meta.chapter.lower():
                matching_indices.append(i)
                continue
            
            # Check page range match
            if page_range and meta.book_page:
                start, end = page_range
                if start <= meta.book_page <= end:
                    matching_indices.append(i)
        
        return matching_indices