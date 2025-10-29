# Copyright 2024
# Directory: HelixRAG/app/services/rerank.py

"""
Reranking service using DashScope Rerank for improved retrieval accuracy.
Enhances search results by reordering them based on query relevance.
"""

import logging
from typing import List, Dict, Any, Optional
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
import dashscope
from llama_index.core.schema import NodeWithScore, TextNode
from ..core.config import get_settings, get_effective_dashscope_key

logger = logging.getLogger(__name__)
settings = get_settings()


class RerankService:
    """DashScope Rerank service for improving retrieval accuracy."""
    
    def __init__(self, top_n: int = 6):
        """
        Initialize rerank service.
        
        Args:
            top_n: Number of top results to return after reranking
        """
        self.top_n = top_n
        # Ensure DashScope SDK sees the API key globally
        try:
            dashscope.api_key = get_effective_dashscope_key()
        except Exception:
            pass
        self.reranker = DashScopeRerank(
            top_n=top_n,
            return_documents=True,
            api_key=get_effective_dashscope_key()
        )
        logger.info(f"Initialized DashScope Rerank service with top_n={top_n}")
    
    async def rerank_results(
        self, 
        query: str, 
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using DashScope Rerank.
        
        Args:
            query: User query
            search_results: Initial search results from vector search
            
        Returns:
            Reranked results with improved relevance
        """
        if not search_results:
            return []
        
        try:
            # Convert search results to LlamaIndex NodeWithScore format
            nodes = []
            for result in search_results:
                # Create TextNode from search result
                node = TextNode(text=result.get('text', ''))
                node.metadata = {
                    'chunk_id': result.get('chunk_id', ''),
                    'source': result.get('source', ''),
                    'similarity': result.get('similarity', 0.0)
                }
                
                # Create NodeWithScore
                node_with_score = NodeWithScore(
                    node=node,
                    score=result.get('similarity', 0.0)
                )
                nodes.append(node_with_score)
            
            # Apply reranking
            reranked_nodes = self.reranker.postprocess_nodes(nodes, query_str=query)
            
            # Convert back to our format
            reranked_results = []
            for node_with_score in reranked_nodes:
                result = {
                    'chunk_id': node_with_score.node.metadata.get('chunk_id', ''),
                    'source': node_with_score.node.metadata.get('source', ''),
                    'text': node_with_score.node.text,
                    'similarity': node_with_score.score,
                    'reranked': True  # Flag to indicate this was reranked
                }
                reranked_results.append(result)
            
            logger.info(f"Reranked {len(search_results)} results to {len(reranked_results)} top results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            logger.warning("Falling back to original search results")
            # Return original results if reranking fails
            return search_results[:self.top_n]
    
    def rerank_results_sync(
        self, 
        query: str, 
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Synchronous version of rerank_results for compatibility.
        
        Args:
            query: User query
            search_results: Initial search results from vector search
            
        Returns:
            Reranked results with improved relevance
        """
        if not search_results:
            return []
        
        try:
            # Convert search results to LlamaIndex NodeWithScore format
            nodes = []
            for result in search_results:
                # Create TextNode from search result
                node = TextNode(text=result.get('text', ''))
                node.metadata = {
                    'chunk_id': result.get('chunk_id', ''),
                    'source': result.get('source', ''),
                    'similarity': result.get('similarity', 0.0)
                }
                
                # Create NodeWithScore
                node_with_score = NodeWithScore(
                    node=node,
                    score=result.get('similarity', 0.0)
                )
                nodes.append(node_with_score)
            
            # Apply reranking
            reranked_nodes = self.reranker.postprocess_nodes(nodes, query_str=query)
            
            # Convert back to our format
            reranked_results = []
            for node_with_score in reranked_nodes:
                result = {
                    'chunk_id': node_with_score.node.metadata.get('chunk_id', ''),
                    'source': node_with_score.node.metadata.get('source', ''),
                    'text': node_with_score.node.text,
                    'similarity': node_with_score.score,
                    'reranked': True  # Flag to indicate this was reranked
                }
                reranked_results.append(result)
            
            logger.info(f"Reranked {len(search_results)} results to {len(reranked_results)} top results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            logger.warning("Falling back to original search results")
            # Return original results if reranking fails
            return search_results[:self.top_n]


# Global rerank service instance
rerank_service = RerankService()
