# Copyright 2024
# Directory: HelixRAG/app/services/rag.py

"""
RAG (Retrieval-Augmented Generation) service implementation.
Orchestrates the complete RAG pipeline: chunk → embed → search → generate.
"""

import logging
import time
import re
from typing import List, Dict, Any, Tuple
from ..core.database import db
from .embedding import embedding_service
from .chat import chat_service
from .chunker import chunker
from .rerank import rerank_service
from .document_parser import document_parser
from ..data.default_documents import DEFAULT_DOCUMENTS

logger = logging.getLogger(__name__)


class RAGService:
    """Main RAG service orchestrating the complete pipeline."""
    
    def __init__(self):
        """Initialize RAG service."""
        self.db = db
        self.embedding_service = embedding_service
        self.chat_service = chat_service
        self.chunker = chunker
        self.rerank_service = rerank_service
        self.document_parser = document_parser
    
    async def seed_documents(self, documents: List[Dict[str, str]] = None) -> int:
        """
        Seed the knowledge base with documents.
        
        Args:
            documents: Optional list of documents. If None, uses default documents.
            
        Returns:
            Number of chunks successfully inserted
        """
        start_time = time.time()
        
        # Use default documents if none provided
        if documents is None:
            documents = DEFAULT_DOCUMENTS
            logger.info("Using default documents for seeding")
        
        try:
            # Step 1: Chunk documents
            chunks = self.chunker.chunk_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            
            # Step 2: Generate embeddings for all chunks
            texts = [chunk['text'] for chunk in chunks]
            embeddings = await self.embedding_service.embed_texts(texts)
            
            # Step 3: Combine chunks with embeddings
            for chunk, embedding in zip(chunks, embeddings):
                chunk['embedding'] = embedding
            
            # Step 4: Store in database
            inserted_count = await self.db.upsert_chunks(chunks)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Seeding completed in {elapsed_ms}ms: {inserted_count} chunks inserted")
            
            return inserted_count
            
        except Exception as e:
            logger.error(f"Seeding failed: {e}")
            raise
    
    async def answer_query(self, query: str, top_k: int = 6) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer, citations, and debug info
        """
        start_time = time.time()
        
        try:
            # Step 1: Generate query embedding (+ bilingual augmentation)
            candidate_queries = self._augment_queries(query)
            aggregated: Dict[str, Dict[str, Any]] = {}
            for q in candidate_queries:
                q_emb = await self.embedding_service.embed_query(q)
                # Step 2: Vector similarity search (retrieve more for reranking)
                partial = await self.db.vector_search(q_emb, top_k * 2)
                for r in partial or []:
                    cid = r.get('chunk_id')
                    if cid and cid not in aggregated:
                        aggregated[cid] = r
            search_results = list(aggregated.values())
            # Heuristic: for policy/refund/return intents, drop obvious tmp/demo sources
            policy_signals = ['退款','退货','政策','return','refund','exchange']
            if any(sig in query.lower() for sig in ['return','refund','exchange','policy']) or any(sig in query for sig in ['退款','退货','退换','政策']):
                filtered = []
                for r in search_results:
                    cid = (r.get('chunk_id') or '').lower()
                    src = (r.get('source') or '').lower()
                    if src.startswith('/tmp/'):
                        continue
                    filtered.append(r)
                if filtered:
                    search_results = filtered
            
            if not search_results:
                return {
                    'text': "I don't have enough information to answer your question. Could you please rephrase or provide more context?",
                    'citations': [],
                    'debug': {
                        'top_doc_ids': [],
                        'latency_ms': int((time.time() - start_time) * 1000)
                    }
                }
            
            # Step 3: Apply reranking for improved accuracy
            reranked_results = self.rerank_service.rerank_results_sync(query, search_results)
            
            # Step 4: Deduplicate and prepare context
            context_blocks = self._prepare_context(reranked_results)
            
            # Step 5: Generate answer
            answer_text = await self.chat_service.generate_answer(query, context_blocks)
            
            # Step 6: Extract citations from answer
            citations = self._extract_citations(answer_text, context_blocks)
            if not citations:
                # Fallback: show top_doc_ids as citations to guarantee sources
                citations = [block['chunk_id'] for block in context_blocks]
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            result = {
                'text': answer_text,
                'citations': citations,
                'debug': {
                    'top_doc_ids': [block['chunk_id'] for block in context_blocks],
                    'latency_ms': elapsed_ms
                }
            }
            
            logger.info(f"Query processed in {elapsed_ms}ms with {len(citations)} citations")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                'text': f"I encountered an error while processing your question: {str(e)}",
                'citations': [],
                'debug': {
                    'top_doc_ids': [],
                    'latency_ms': int((time.time() - start_time) * 1000)
                }
            }

    def _augment_queries(self, query: str) -> List[str]:
        """
        Simple bilingual augmentation to improve recall on CN/EN mixes.
        Expands common policy domain terms.
        """
        zh2en = {
            '退货': 'return',
            '退款': 'refund',
            '退换': 'exchange',
            '配送': 'shipping',
            '运费': 'shipping',
            '物流': 'shipping',
            '运输': 'shipping',
            '尺码': 'size',
            '大小': 'size',
            '政策': 'policy',
        }
        en2zh = {
            'return': '退货',
            'refund': '退款',
            'exchange': '退换',
            'shipping': '配送',
            'size': '尺码',
            'policy': '政策',
        }
        zh_synonyms = {
            '退款': ['退货'],
            '退货': ['退款'],
        }
        aug = [query]
        # If contains Chinese characters, append mapped English keywords
        if any('\u4e00' <= ch <= '\u9fff' for ch in query):
            extra_terms = []
            for zh, en in zh2en.items():
                if zh in query:
                    extra_terms.append(en)
                    # add zh synonyms to improve recall
                    for syn in zh_synonyms.get(zh, []):
                        extra_terms.append(syn)
            if extra_terms:
                aug.append(' '.join(extra_terms))
            if ('退款' in query) or ('退货' in query) or ('政策' in query):
                aug.append('退货 政策 退款')
        else:
            # English query: append Chinese keywords
            extra_terms = []
            for en, zh in en2zh.items():
                if en in query.lower():
                    extra_terms.append(zh)
            if extra_terms:
                aug.append(' '.join(extra_terms))
            if any(k in query.lower() for k in ['return','refund','policy']):
                aug.append('退货 政策 退款')
        return aug
    
    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare context blocks from search results with simple deduplication.
        
        Args:
            search_results: Raw search results from database
            
        Returns:
            Processed context blocks
        """
        # Simple deduplication by chunk_id prefix (MMR-lite)
        seen_prefixes = set()
        context_blocks = []
        
        for result in search_results:
            chunk_id = result.get('chunk_id', '')
            
            # Extract base chunk ID (before #)
            base_id = chunk_id.split('#')[0] if '#' in chunk_id else chunk_id
            
            if base_id not in seen_prefixes:
                context_blocks.append(result)
                seen_prefixes.add(base_id)
            
            # Limit to reasonable context size
            if len(context_blocks) >= 4:
                break
        
        logger.info(f"Prepared {len(context_blocks)} unique context blocks")
        return context_blocks
    
    def _extract_citations(self, answer_text: str, context_blocks: List[Dict[str, Any]]) -> List[str]:
        """
        Extract citation chunk_ids from the generated answer.
        
        Args:
            answer_text: Generated answer text
            context_blocks: Context blocks that were provided
            
        Returns:
            List of chunk_ids that were cited
        """
        # Find all citations in format [chunk_id]
        citation_pattern = r'\[([^\]]+)\]'
        found_citations = re.findall(citation_pattern, answer_text)
        
        # Filter to only include valid chunk_ids from context
        valid_chunk_ids = {block['chunk_id'] for block in context_blocks}
        valid_citations = [cite for cite in found_citations if cite in valid_chunk_ids]
        
        # Remove duplicates while preserving order
        unique_citations = []
        for cite in valid_citations:
            if cite not in unique_citations:
                unique_citations.append(cite)
        
        return unique_citations
    
    async def process_uploaded_files(self, uploaded_files: List[Any]) -> int:
        """
        Process uploaded files and add them to the knowledge base.
        
        Args:
            uploaded_files: List of uploaded file objects
            
        Returns:
            Number of chunks successfully inserted
        """
        start_time = time.time()
        
        try:
            # Step 1: Parse uploaded files using enhanced parser
            documents = self.document_parser.parse_uploaded_files(uploaded_files)
            
            if not documents:
                logger.warning("No documents were parsed from uploaded files")
                return 0
            
            # Step 2: Convert documents to chunks
            chunks = self.document_parser.convert_to_chunks(documents)
            
            # Step 3: Generate embeddings for all chunks
            texts = [chunk['text'] for chunk in chunks]
            embeddings = await self.embedding_service.embed_texts(texts)
            
            # Step 4: Combine chunks with embeddings
            for chunk, embedding in zip(chunks, embeddings):
                chunk['embedding'] = embedding
            
            # Step 5: Store in database
            inserted_count = await self.db.upsert_chunks(chunks)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"File processing completed in {elapsed_ms}ms: {inserted_count} chunks inserted")
            
            return inserted_count
            
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            raise
    
    async def process_directory(self, directory_path: str, file_extensions: List[str] = None) -> int:
        """
        Process all files in a directory and add them to the knowledge base.
        
        Args:
            directory_path: Path to directory containing files
            file_extensions: Optional list of file extensions to include
            
        Returns:
            Number of chunks successfully inserted
        """
        start_time = time.time()
        
        try:
            # Step 1: Parse directory using enhanced parser
            documents = self.document_parser.parse_directory(directory_path, file_extensions)
            
            if not documents:
                logger.warning(f"No documents were found in directory: {directory_path}")
                return 0
            
            # Step 2: Convert documents to chunks
            chunks = self.document_parser.convert_to_chunks(documents)
            
            # Step 3: Generate embeddings for all chunks
            texts = [chunk['text'] for chunk in chunks]
            embeddings = await self.embedding_service.embed_texts(texts)
            
            # Step 4: Combine chunks with embeddings
            for chunk, embedding in zip(chunks, embeddings):
                chunk['embedding'] = embedding
            
            # Step 5: Store in database
            inserted_count = await self.db.upsert_chunks(chunks)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Directory processing completed in {elapsed_ms}ms: {inserted_count} chunks inserted")
            
            return inserted_count
            
        except Exception as e:
            logger.error(f"Directory processing failed: {e}")
            raise


# Global RAG service instance
rag_service = RAGService()
