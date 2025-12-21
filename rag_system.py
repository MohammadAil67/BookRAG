"""
Enhanced RAG System - Tier 2+ Implementation

IMPROVEMENTS OVER ORIGINAL:
============================
1. ✅ Embedding-based topic merging (prevents duplicate topics)
2. ✅ Entity-weighted topic scoring (better relevance detection)
3. ✅ Intent-based hard reset (detects topic switches)
4. ✅ Citation validation (prevents hallucination)
5. ✅ Dynamic retrieval K (adapts based on confidence)
6. ✅ Topic embeddings cached (performance boost)
7. ✅ Enhanced logging and debugging
"""

import os
import re
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
        print("🚀 Initializing Enhanced Tier 2+ RAG System...")
        print("   ✨ Features: Semantic topic merging, Citation validation, Dynamic K")
        self.config = config
        self.llm = GroqLLM(config.GROQ_API_KEY)
        
        # 1. Load/Process PDF
        self.chunks, self.embeddings = self._load_data()
        
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
        """Loads chunks and embeddings from cache or processes PDF."""
        if os.path.exists(self.config.CHUNKS_FILE) and os.path.exists(self.config.EMBEDDINGS_FILE):
            try:
                with open(self.config.CHUNKS_FILE, 'rb') as f: 
                    chunks = pickle.load(f)
                with open(self.config.EMBEDDINGS_FILE, 'rb') as f: 
                    embeddings = pickle.load(f)
                print(f"📚 Loaded {len(chunks)} chunks from cache")
                return chunks, embeddings
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}. Re-processing...")

        try:
            from PDFprocessing import PDFProcess 
            text = PDFProcess.process_pdf(self.config.PDF_PATH, SystemUtils.find_poppler())
            chunks = PDFProcess.create_chunks(text)
        except ImportError:
            print("⚠️ PDFprocessing module not found. Using dummy data.")
            text = "Dummy text content for testing purposes."
            chunks = [text]

        print("⚙️ Generating embeddings (this may take a moment)...")
        temp_embedder = SentenceTransformer('BAAI/bge-m3')
        embeddings = temp_embedder.encode(chunks, show_progress_bar=True)
        
        with open(self.config.CHUNKS_FILE, 'wb') as f: 
            pickle.dump(chunks, f)
        with open(self.config.EMBEDDINGS_FILE, 'wb') as f: 
            pickle.dump(embeddings, f)
        
        return chunks, embeddings

    def _extract_entities(self, text: str) -> Set[str]:
        """Extract Named Entities"""
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities = re.findall(pattern, text)
        stopwords = {
            'According', 'The', 'This', 'That', 'These', 'Those', 'There', 'Here', 
            'What', 'When', 'Where', 'Who', 'How', 'Answer', 
            'Liberation', 'Language', 'War'
        }
        return set([e for e in entities if e not in stopwords])
    
    def generate_quiz(self, topic: str, difficulty: str, num_questions: int) -> List[Dict]:
        """
        Generates a structured quiz in JSON format based on the topic and PDF content.
        """
        print(f"🎲 Generating {difficulty} quiz for topic: '{topic}'")
        
        # 1. Retrieve relevant content specifically for the quiz
        # We use a broad retrieval here to get enough context
        search_query = f"facts concepts details about {topic}"
        chunk_indices = self.retriever.retrieve(search_query, [])
        # Take top 5 chunks to ensure enough material for questions
        context_chunks = [self.chunks[i] for i in chunk_indices[:5]]
        
        context_text = "\n\n".join(context_chunks)
        
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
        """

        # 3. Generate
        try:
            response = self.llm.generate(prompt)
            
            # 4. Clean and Parse JSON
            # Remove Markdown code blocks if present (```json ... ```)
            cleaned_response = re.sub(r'```json\s*|\s*```', '', response).strip()
            
            # Attempt to parse
            quiz_data = json.loads(cleaned_response)
            return quiz_data
            
        except json.JSONDecodeError:
            print("⚠️ Failed to parse Quiz JSON. Returning empty list.")
            print(f"Raw output: {response}")
            return []
        except Exception as e:
            print(f"⚠️ Quiz generation error: {e}")
            return []

    # --- MAIN PIPELINE ---
    def ask(self, query: str) -> str:
        """
        Main query pipeline with intelligent routing.
        Uses enhanced topic tracking and citation validation.
        """
        print(f"\n🔍 Processing: '{query}'")
        
        # Route based on query complexity
        should_decompose, decomp_type = self.decomposer.should_decompose(query)
        
        if should_decompose:
            print(f"  🔀 Complex query detected: {decomp_type}")
            return self._ask_with_decomposition(query)
        else:
            print(f"  📝 Simple query - using standard pipeline")
            return self._ask_simple(query)
    
    def _ask_simple(self, query: str) -> str:
        """
        ENHANCED: Standard pipeline with citation validation and dynamic K.
        """
        # 1. Apply smart context
        topic_context = self.context_applier.get_smart_context(query)
        
        if topic_context:
            print(f"  📌 Applying context: '{topic_context}'")
        
        # 2. Refine query
        refined_query = self.refiner.refine(query, self.chat_history, topic_context)
        
        if refined_query != query:
            print(f"  🔧 Refined to: '{refined_query}'")
        
        # 3. Retrieve chunks
        chunk_indices = self.retriever.retrieve(refined_query, self.chat_history)
        retrieved_chunks = [self.chunks[i] for i in chunk_indices]
        
        print(f"  📦 Retrieved {len(retrieved_chunks)} chunks")
        
        if not retrieved_chunks:
            return "I couldn't find any relevant information in the document."
        
        # 4. Generate answer
        prompt = PromptBuilder.build(refined_query, retrieved_chunks, self.chat_history)
        answer = self.llm.generate(prompt)
        
        # 5. Validate using Embeddings (Fast)
        is_grounded, supporting_chunks, confidence = \
            self.citation_validator.validate_answer(answer, retrieved_chunks)
        
        print(f"  📊 Grounding Score: {confidence:.2%}")
        
        # If validation fails, try ONE regeneration with stricter instructions
        if not is_grounded:
            print(f"  ⚠️ Low grounding detected. Regenerating...")
            strict_prompt = prompt + "\n\nCRITICAL INSTRUCTION: The previous answer was rejected because it included information not found in the text. You must cite specific details from the context provided above."
            answer = self.llm.generate(strict_prompt)
        
        # 6. Update state (Happens for both good and bad answers)
        self._update_history(query, answer)
        self.topic_tracker.update(query, answer)
        
        return answer
    
    def _ask_with_decomposition(self, query: str) -> str:
        """
        ENHANCED: Handle complex queries with citation validation.
        """
        # 1. Decompose query
        decomp_result = self.decomposer.decompose(query)
        sub_queries = decomp_result.sub_queries
        
        print(f"  📋 Decomposed into {len(sub_queries)} sub-queries:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"     {i}. {sq}")
        
        # 2. Retrieve chunks for each sub-query
        all_chunk_indices = set()
        for i, sub_q in enumerate(sub_queries, 1):
            print(f"  🔎 Retrieving for sub-query {i}/{len(sub_queries)}...")
            chunk_indices = self.retriever.retrieve(sub_q, self.chat_history)
            all_chunk_indices.update(chunk_indices)
        
        # 3. Get all unique chunks
        unique_chunks = [self.chunks[i] for i in all_chunk_indices]
        print(f"  📦 Total unique chunks: {len(unique_chunks)}")
        
        if not unique_chunks:
            return "I couldn't find any relevant information in the document."
        
        # 4. Rerank combined chunks
        if len(unique_chunks) > self.config.FINAL_TOP_K:
            print(f"  🎯 Reranking {len(unique_chunks)} chunks...")
            reranked = self.reranker.rerank(query, unique_chunks, self.config.FINAL_TOP_K)
            final_chunks = [unique_chunks[idx] for idx, score in reranked]
        else:
            final_chunks = unique_chunks
        
        # 5. Generate comprehensive answer
        prompt = self._build_decomposed_prompt(
            original_query=query,
            sub_queries=sub_queries,
            chunks=final_chunks,
            decomp_type=decomp_result.decomposition_type
        )
        
        answer = self.llm.generate(prompt)
        
        # 6. Validate (Citation Validator)
        is_grounded, supporting_chunks, cite_confidence = \
            self.citation_validator.validate_answer(answer, final_chunks)
        
        print(f"  ✅ Citation confidence: {cite_confidence:.2%}")
        
        if not is_grounded:
            print(f"  ⚠️ Low grounding. Regenerating...")
            prompt += "\n\nCRITICAL: Answer ONLY using provided context. Cite specific details."
            answer = self.llm.generate(prompt)
        
        # 7. Update state
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

Context from the document:
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

    def get_topic_status(self) -> Dict:
        """ENHANCED: Get topic information with entity tracking"""
        status = self.topic_tracker.get_topic_hints()
        status['context_would_apply'] = self.context_applier.should_apply_context(
            "What about it?"
        )
        return status
    
    def get_system_stats(self) -> Dict:
        """ENHANCED: More comprehensive statistics"""
        current_topic = self.topic_tracker.get_current_topic()
        
        return {
            'total_chunks': len(self.chunks),
            'conversation_turns': len(self.chat_history),
            'topics_discussed': len(self.topic_tracker.get_all_topics()),
            'current_topic': current_topic.name if current_topic else None,
            'current_topic_confidence': current_topic.confidence if current_topic else 0.0,
            'entity_mentions': dict(current_topic.entity_mentions) if current_topic else {},
            'last_intent': self.topic_tracker.last_intent,
            'total_entities_tracked': len(self.topic_tracker.entity_history)
        }