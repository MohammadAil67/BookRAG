import os
import re
import pickle
import numpy as np
from typing import List, Dict, Set
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from config import Config, SystemUtils, HistoryObject
from models import GroqLLM, Embedder, Reranker
from retrieval import HybridRetriever, TopicAwareRetriever, MultiQueryRetriever
from processing import PromptBuilder, AdvancedQueryRefiner, SelfReflector, MultiQueryGenerator
from topics import ConversationTopicTracker
from query_decomposition import QueryDecomposer, SmartContextApplier

class RAGSystem:
    def __init__(self, config: Config):
        print("🚀 Initializing 9.7/10 RAG System (Query Decomposition Enabled)...")
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
        
        self.topic_retriever = TopicAwareRetriever(
            base_retriever=base_retriever,
            embedder=self.embedder,
            config=self.config
        )
        
        self.query_generator = MultiQueryGenerator(self.llm)
        self.multi_query_retriever = MultiQueryRetriever(
            base_retriever=self.topic_retriever,
            query_generator=self.query_generator,
            reranker=self.reranker,
            config=self.config,
            chunks=self.chunks
        )
        self.retriever = self.multi_query_retriever
        
        # 4. Query Intelligence
        self.decomposer = QueryDecomposer(self.llm)
        self.refiner = AdvancedQueryRefiner(self.llm)
        self.verifier = SelfReflector(self.llm)
        
        # 5. Topic Management
        self.topic_tracker = ConversationTopicTracker()
        self.context_applier = SmartContextApplier(self.topic_tracker)
        
        # 6. State
        self.chat_history: List[Dict] = []
        self.history = HistoryObject()
        
        print("✅ System Ready with Query Decomposition")

    def _load_data(self):
        """Loads chunks and embeddings from cache or processes PDF if missing."""
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

    # --- HELPER METHODS ---
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

    # --- MAIN PIPELINE ---
    def ask(self, query: str) -> str:
        """
        Main query pipeline with intelligent routing:
        - Complex queries → Decomposition
        - Follow-ups with pronouns → Context injection
        - Direct queries → Standard retrieval
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
        """Standard pipeline for simple queries"""
        
        # 1. Apply smart context if needed
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
        
        # 5. Verify with self-reflection
        is_valid, reason = self.verifier.verify(answer, retrieved_chunks)
        
        if not is_valid:
            print(f"  ⚠️ Answer rejected by Verifier: {reason}")
            print(f"  🔄 Regenerating with stricter constraints...")
            
            prompt += "\n\nCRITICAL: The previous answer was rejected. Answer ONLY using the context provided above. Do not add information from outside the context."
            answer = self.llm.generate(prompt)
        
        # 6. Update state
        self._update_history(query, answer)
        self.topic_tracker.update(query, answer)
        
        return answer
    
    def _ask_with_decomposition(self, query: str) -> str:
        """Handle complex queries with decomposition"""
        
        # 1. Decompose query
        decomp_result = self.decomposer.decompose(query)
        sub_queries = decomp_result.sub_queries
        
        print(f"  📋 Decomposed into {len(sub_queries)} sub-queries:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"     {i}. {sq}")
        
        # 2. Retrieve chunks for each sub-query
        all_chunk_indices = set()
        sub_query_chunks = {}
        
        for i, sub_q in enumerate(sub_queries, 1):
            print(f"  🔎 Retrieving for sub-query {i}/{len(sub_queries)}...")
            
            # Retrieve for this sub-query
            chunk_indices = self.retriever.retrieve(sub_q, self.chat_history)
            all_chunk_indices.update(chunk_indices)
            sub_query_chunks[sub_q] = chunk_indices
            
            print(f"     → Found {len(chunk_indices)} chunks")
        
        # 3. Get all unique chunks
        unique_chunks = [self.chunks[i] for i in all_chunk_indices]
        
        print(f"  📦 Total unique chunks: {len(unique_chunks)}")
        
        if not unique_chunks:
            return "I couldn't find any relevant information in the document to answer this question."
        
        # 4. Rerank combined chunks for the original query
        if len(unique_chunks) > self.config.FINAL_TOP_K:
            print(f"  🎯 Reranking {len(unique_chunks)} chunks to top {self.config.FINAL_TOP_K}...")
            
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
        
        # 6. Verify
        is_valid, reason = self.verifier.verify(answer, final_chunks)
        
        if not is_valid:
            print(f"  ⚠️ Answer rejected: {reason}")
            print(f"  🔄 Regenerating...")
            
            prompt += "\n\nCRITICAL: Answer ONLY using the provided context. Be specific and cite details."
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
        
        # Format context
        context_text = "\n\n".join([
            f"[Context {i+1}]\n{chunk}" 
            for i, chunk in enumerate(chunks)
        ])
        
        # Format conversation history
        history_text = ""
        if self.chat_history:
            recent = self.chat_history[-3:]
            history_text = "\n".join([
                f"User: {h['user']}\nAssistant: {h['ai']}" 
                for h in recent
            ])
        
        # Build specialized instructions based on decomposition type
        if decomp_type == 'comparison':
            special_instructions = """
For comparison questions:
1. Clearly describe each entity being compared
2. Highlight key similarities
3. Highlight key differences
4. Provide specific examples from the context
5. Organize your answer logically (e.g., similarities first, then differences)"""
        
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
1. Address all aspects of the question comprehensively
2. Organize information logically
3. Use specific details from the context
4. Ensure all sub-questions are answered"""
        
        # Build full prompt
        prompt = f"""You are answering a complex question that has been broken down into sub-questions for better analysis.

Original Question: {original_query}

The question was broken into these sub-questions:
{chr(10).join(f"{i+1}. {sq}" for i, sq in enumerate(sub_queries))}

Context from the document:
{context_text}

Previous conversation:
{history_text if history_text else "None"}

Instructions:
{special_instructions}

CRITICAL RULES:
- Use ONLY information from the provided context
- Do not make up or infer information not present in the context
- If the context doesn't contain information to fully answer the question, acknowledge this
- Be specific and cite relevant details
- Maintain a clear, organized structure

Answer:"""
        
        return prompt
    
    def _update_history(self, question: str, answer: str):
        """Update conversation history"""
        self.chat_history.append({"user": question, "ai": answer})
        self.history.history.append({"question": question, "answer": answer})
        
        # Keep history manageable
        if len(self.chat_history) > self.config.MAX_CONVERSATION_HISTORY:
            self.chat_history = self.chat_history[-self.config.MAX_CONVERSATION_HISTORY:]

    def clear_history(self):
        """Clear conversation history and reset topic tracker"""
        self.chat_history = []
        self.history.history = []
        self.topic_tracker = ConversationTopicTracker()
        self.context_applier = SmartContextApplier(self.topic_tracker)
        print("🗑️ History cleared")

    def get_topic_status(self) -> Dict:
        """Get current topic information for debugging/UI"""
        status = self.topic_tracker.get_topic_hints()
        status['context_would_apply'] = self.context_applier.should_apply_context(
            "What about it?"  # Test with vague query
        )
        return status
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'total_chunks': len(self.chunks),
            'conversation_turns': len(self.chat_history),
            'topics_discussed': len(self.topic_tracker.get_all_topics()),
            'current_topic': self.topic_tracker.get_current_topic().name 
                           if self.topic_tracker.get_current_topic() else None,
        }