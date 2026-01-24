"""
FocusedRAG Implementation

This module implements the FocusedRAG methodology for academic document retrieval,
featuring variance-maximization section localization and density-guided sentence-level
evidence selection.

Key Components:
1. Section-Aware Document Partitioning
2. Intent-Driven Section Localization with Variance-Maximization Adaptive Thresholding
3. Density-Guided Sentence-Level Evidence Selection
4. Budget-Constrained Evidence Aggregation
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/retrieval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FocusedRAG:
    """
    FocusedRAG: A precision-focused retrieval system for academic documents.
    
    Uses variance-maximization for section selection and density-guided 
    mechanisms for sentence-level evidence extraction.
    """
    
    def __init__(self, rerank_function, token_budget: int = 2000):
        """
        Initialize FocusedRAG.
        
        Args:
            rerank_function: Async function for computing relevance scores
            token_budget: Maximum token budget for retrieved evidence
        """
        self.rerank_function = rerank_function
        self.token_budget = token_budget
        
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences using improved sentence boundary detection.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Handle common abbreviations and edge cases
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split on sentence boundaries while preserving structure
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Filter out empty sentences and very short fragments
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    @staticmethod
    def estimate_token_count(text: str) -> int:
        """
        Estimate token count for text (rough approximation).
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 characters per token on average
        return len(text) // 4
    
    def variance_maximization_threshold(self, scores: List[float]) -> int:
        """
        Compute optimal threshold using variance maximization (Otsu's method).
        
        This method finds the threshold that maximizes between-class variance,
        effectively separating high-relevance items from low-relevance ones.
        
        Args:
            scores: List of relevance scores (sorted in descending order)
            
        Returns:
            Optimal boundary index k*
        """
        if len(scores) <= 1:
            return len(scores)
        
        n = len(scores)
        scores_array = np.array(scores)
        
        max_variance = -1
        optimal_k = 1
        
        # Try all possible partition points
        for k in range(1, n):
            # High-relevance group: top k scores
            high_group = scores_array[:k]
            # Low-relevance group: remaining scores
            low_group = scores_array[k:]
            
            # Compute group statistics
            w_high = k / n
            w_low = (n - k) / n
            
            mu_high = np.mean(high_group)
            mu_low = np.mean(low_group)
            
            # Between-class variance
            variance_between = w_high * w_low * (mu_high - mu_low) ** 2
            
            # Track maximum
            if variance_between > max_variance:
                max_variance = variance_between
                optimal_k = k
        
        logger.info(f"Variance maximization: selected top {optimal_k} of {n} items (max variance: {max_variance:.4f})")
        return optimal_k
    
    async def section_localization(
        self, 
        query: str, 
        sections: List[Dict]
    ) -> List[Dict]:
        """
        Perform intent-driven section localization using variance-maximization.
        
        Args:
            query: User query
            sections: List of section dictionaries with 'title', 'path', 'content', etc.
            
        Returns:
            Selected sections with relevance scores
        """
        if not sections:
            return []
        
        logger.critical(f"FocusedRAG: Performing section-level intent alignment for query")
        
        # Construct section signatures (title + opening sentence)
        section_signatures = []
        for section in sections:
            title = section.get('title', '')
            content = section.get('content', '')
            
            # Extract opening sentence as context
            sentences = self.split_into_sentences(content)
            opening_sentence = sentences[0] if sentences else ""
            
            # Signature: title + opening sentence
            signature = f"{title}. {opening_sentence}"
            section_signatures.append(signature)
        
        # Compute intent alignment scores using rerank function
        if not section_signatures:
            return []
        
        rerank_results = await self.rerank_function(query, section_signatures)
        
        # Extract scores and sort in descending order
        section_scores = []
        for i, section in enumerate(sections):
            # Find matching rerank result
            score = 0.0
            for result in rerank_results:
                if result['index'] == i:
                    score = result['relevance_score']
                    break
            section_scores.append({'section': section, 'score': score})
        
        # Sort by relevance score (descending)
        section_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Extract sorted scores for variance maximization
        sorted_scores = [item['score'] for item in section_scores]
        
        # Apply variance-maximization threshold
        optimal_k = self.variance_maximization_threshold(sorted_scores)
        
        # Select top-k sections
        selected_sections = section_scores[:optimal_k]
        
        logger.critical(f"FocusedRAG: Selected {len(selected_sections)} sections from {len(sections)} candidates")
        
        return selected_sections
    
    def density_guided_sentence_selection(
        self,
        sentences: List[str],
        relevance_scores: List[float],
        preserve_order: bool = True
    ) -> List[Tuple[int, str, float]]:
        """
        Perform density-guided sentence-level evidence selection.
        
        This implements the surplus-based interval extraction with topological filtering
        to identify high-density evidence regions.
        
        Args:
            sentences: List of sentences
            relevance_scores: Corresponding relevance scores
            preserve_order: Whether to preserve original document order
            
        Returns:
            List of tuples (original_index, sentence, score)
        """
        if not sentences or not relevance_scores:
            return []
        
        n = len(sentences)
        assert len(relevance_scores) == n, "Sentences and scores must have same length"
        
        # Step 1: Compute adaptive baseline using variance maximization
        sorted_scores = sorted(relevance_scores, reverse=True)
        optimal_k = self.variance_maximization_threshold(sorted_scores)
        tau_sent = sorted_scores[optimal_k - 1] if optimal_k > 0 else sorted_scores[0]
        
        logger.info(f"FocusedRAG: Adaptive sentence baseline: {tau_sent:.4f}")
        
        # Step 2: Compute relevance surplus for each sentence
        delta = [score - tau_sent for score in relevance_scores]
        
        # Step 3: Extract positive-surplus intervals
        intervals = []
        i = 0
        while i < n:
            if delta[i] > 0:
                # Start of potential interval
                start = i
                cumulative_surplus = 0
                end = start
                
                # Extend interval while cumulative surplus remains positive
                for j in range(start, n):
                    cumulative_surplus += delta[j]
                    if cumulative_surplus > 0:
                        end = j
                    else:
                        break
                
                # Record interval
                total_surplus = sum(delta[start:end + 1])
                intervals.append({
                    'start': start,
                    'end': end,
                    'length': end - start + 1,
                    'total_surplus': total_surplus
                })
                
                i = end + 1
            else:
                i += 1
        
        if not intervals:
            logger.warning("FocusedRAG: No positive-surplus intervals found")
            return []
        
        # Step 4: Topological filtering - prefer multi-sentence intervals
        supported_intervals = [intv for intv in intervals if intv['length'] >= 2]
        
        if supported_intervals:
            # Use supported intervals (density-rich regions)
            final_intervals = supported_intervals
            logger.info(f"FocusedRAG: Using {len(supported_intervals)} supported intervals")
        else:
            # Fallback: select single highest-surplus interval
            best_interval = max(intervals, key=lambda x: x['total_surplus'])
            final_intervals = [best_interval]
            logger.info(f"FocusedRAG: Fallback to highest-surplus single interval")
        
        # Step 5: Sort intervals by total surplus (descending)
        final_intervals.sort(key=lambda x: x['total_surplus'], reverse=True)
        
        # Extract sentences from intervals
        selected_sentences = []
        for interval in final_intervals:
            for idx in range(interval['start'], interval['end'] + 1):
                selected_sentences.append((idx, sentences[idx], relevance_scores[idx]))
        
        # Optionally preserve original document order
        if preserve_order:
            selected_sentences.sort(key=lambda x: x[0])
        
        logger.critical(f"FocusedRAG: Selected {len(selected_sentences)} sentences from {n} candidates")
        
        return selected_sentences
    
    async def budget_constrained_aggregation(
        self,
        selected_sentences: List[Tuple[int, str, float]],
        token_budget: Optional[int] = None
    ) -> List[Dict]:
        """
        Aggregate selected sentences under token budget constraint.
        
        Args:
            selected_sentences: List of (index, sentence, score) tuples
            token_budget: Token budget (uses instance default if None)
            
        Returns:
            List of sentence dictionaries with metadata
        """
        if token_budget is None:
            token_budget = self.token_budget
        
        # Sort by relevance score (descending) for greedy selection
        sorted_sentences = sorted(selected_sentences, key=lambda x: x[2], reverse=True)
        
        aggregated = []
        total_tokens = 0
        
        for idx, sentence, score in sorted_sentences:
            token_count = self.estimate_token_count(sentence)
            
            if total_tokens + token_count <= token_budget:
                aggregated.append({
                    'original_index': idx,
                    'content': sentence,
                    'relevance_score': score,
                    'token_count': token_count
                })
                total_tokens += token_count
            else:
                # Budget exhausted
                break
        
        # Re-sort by original document order for coherent context
        aggregated.sort(key=lambda x: x['original_index'])
        
        logger.critical(f"FocusedRAG: Aggregated {len(aggregated)} sentences using {total_tokens}/{token_budget} tokens")
        
        return aggregated
    
    async def retrieve(
        self,
        query: str,
        sections: List[Dict],
        token_budget: Optional[int] = None
    ) -> Dict:
        """
        Full FocusedRAG retrieval pipeline.
        
        Args:
            query: User query
            sections: List of document sections
            token_budget: Token budget (uses instance default if None)
            
        Returns:
            Dictionary with selected sections and evidence sentences
        """
        if token_budget is None:
            token_budget = self.token_budget
        
        logger.critical(f"FocusedRAG: Starting retrieval pipeline for query")
        
        # Stage 1: Section Localization
        selected_sections = await self.section_localization(query, sections)
        
        if not selected_sections:
            logger.warning("FocusedRAG: No relevant sections found")
            return {'sections': [], 'sentences': [], 'total_tokens': 0}
        
        # Stage 2: Sentence-Level Evidence Selection
        all_sentences = []
        all_scores = []
        sentence_to_section = []
        
        for section_data in selected_sections:
            section = section_data['section']
            content = section.get('content', '')
            
            # Split into sentences
            sentences = self.split_into_sentences(content)
            
            if not sentences:
                continue
            
            # Compute sentence-level relevance scores
            rerank_results = await self.rerank_function(query, sentences)
            
            # Build score mapping
            sentence_scores = [0.0] * len(sentences)
            for result in rerank_results:
                idx = result['index']
                if 0 <= idx < len(sentences):
                    sentence_scores[idx] = result['relevance_score']
            
            # Accumulate across sections
            all_sentences.extend(sentences)
            all_scores.extend(sentence_scores)
            sentence_to_section.extend([section] * len(sentences))
        
        # Apply density-guided selection
        selected_sentence_data = self.density_guided_sentence_selection(
            all_sentences,
            all_scores,
            preserve_order=True
        )
        
        # Stage 3: Budget-Constrained Aggregation
        final_evidence = await self.budget_constrained_aggregation(
            selected_sentence_data,
            token_budget=token_budget
        )
        
        # Compile results
        result = {
            'sections': [s['section'] for s in selected_sections],
            'section_scores': [s['score'] for s in selected_sections],
            'sentences': final_evidence,
            'total_tokens': sum(s['token_count'] for s in final_evidence),
            'num_sections_selected': len(selected_sections),
            'num_sentences_selected': len(final_evidence)
        }
        
        logger.critical(
            f"FocusedRAG: Retrieval complete - "
            f"{result['num_sections_selected']} sections, "
            f"{result['num_sentences_selected']} sentences, "
            f"{result['total_tokens']} tokens"
        )
        
        return result
