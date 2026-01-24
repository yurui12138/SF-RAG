# FocusedRAG Integration Guide

## Overview

This document describes the integration of **FocusedRAG** methodology into the PT-RAG system. FocusedRAG significantly improves retrieval precision and recall through variance-maximization section localization and density-guided sentence-level evidence selection.

## What is FocusedRAG?

FocusedRAG is an advanced retrieval methodology specifically designed for academic documents that addresses the limitations of traditional RAG systems by:

1. **Section-Aware Document Partitioning**: Parsing document hierarchy to establish isolated retrieval subspaces
2. **Intent-Driven Section Localization**: Using variance-maximization adaptive thresholding to compress global search spaces
3. **Density-Guided Sentence Selection**: Leveraging local relevance density patterns to identify high-quality evidence
4. **Budget-Constrained Aggregation**: Efficiently managing token budgets while maximizing information quality

## Key Improvements

### 1. Variance-Maximization Threshold Selection

Unlike fixed thresholds, FocusedRAG adaptively determines optimal boundaries by maximizing between-class variance. This approach:

- **Adapts to query complexity**: Automatically adjusts selection based on relevance distribution
- **Handles multi-hop queries**: Expands selection when evidence spans multiple sections
- **Robust to noise**: Maintains performance even with imperfect relevance predictions

**Mathematical Foundation**:
```
σ²_B(k) = ω_H(k) · ω_L(k) · (μ_H(k) - μ_L(k))²

k* = argmax_{1 ≤ k < n} σ²_B(k)
```

### 2. Density-Guided Sentence Selection

FocusedRAG uses relevance surplus and cumulative density analysis to:

- **Improve recall**: Retain contextually supported sentences even if slightly below threshold
- **Improve precision**: Suppress isolated high-scoring fragments lacking contextual support
- **Preserve continuity**: Maintain coherent evidence regions

**Key Concepts**:
- **Relevance Surplus**: δ_j = v_j - τ_sent
- **Interval Extraction**: Longest continuous intervals with positive cumulative surplus
- **Topological Filtering**: Prefer multi-sentence intervals over isolated matches

### 3. Budget-Constrained Evidence Aggregation

Intelligently allocates token budget to maximize evidence quality:

- **Greedy selection by surplus**: Prioritize highest-relevance intervals
- **Document-order preservation**: Maintain coherent context for LLM
- **Efficient resource utilization**: Maximize information density within budget

## Architecture

### Module Structure

```
focused_rag.py
├── FocusedRAG (main class)
│   ├── __init__(rerank_function, token_budget)
│   ├── variance_maximization_threshold()
│   ├── section_localization()
│   ├── density_guided_sentence_selection()
│   ├── budget_constrained_aggregation()
│   └── retrieve()  # Main pipeline
```

### Integration Points

1. **LocalRetriever** (`local_retriever.py`):
   - Initialized with FocusedRAG instance in `__init__()`
   - Enhanced `retrieve_contexts()` method with FocusedRAG option
   - New `focused_rag_retrieve_contexts()` method for FocusedRAG pipeline

2. **PaperTreeRAG** (`paper_tree_rag.py`):
   - Configuration parameters in `retrieval_config`:
     - `use_focused_rag`: Enable/disable FocusedRAG (default: True)
     - `token_budget`: Maximum tokens for evidence (default: 2000)

## Configuration

### Basic Configuration

```python
from config import Config
from paper_tree_rag import PaperTreeRAG

config = Config.from_env(prefix="GPT_4o_mini")

# Default configuration (FocusedRAG enabled with 2000 token budget)
processor = PaperTreeRAG(config)
```

### Custom Configuration

```python
# Customize FocusedRAG parameters
custom_config = {
    "use_focused_rag": True,   # Enable FocusedRAG
    "token_budget": 3000,       # Increase token budget
    "top_n": 5,                 # More candidates for reranking
    "min_relevance_score": 0.15 # Higher quality threshold
}

processor = PaperTreeRAG(config, retrieval_config=custom_config)
```

### Disabling FocusedRAG

```python
# Revert to original retrieval method
legacy_config = {
    "use_focused_rag": False
}

processor = PaperTreeRAG(config, retrieval_config=legacy_config)
```

## Usage Examples

### Example 1: Single-Paper Detail Query with FocusedRAG

```python
import asyncio
from config import Config
from paper_tree_rag import PaperTreeRAG, always_get_an_event_loop

# Setup
config = Config.from_env(prefix="GPT_4o_mini")
loop = always_get_an_event_loop()
processor = PaperTreeRAG(config)

# Build paper tree (one-time setup)
loop.run_until_complete(processor.build_paper_tree("./files"))

# Perform local retrieval with FocusedRAG
result = loop.run_until_complete(
    processor.local_retrieval(
        'How is the embedding layer designed and implemented in the Vision Transformer?',
        'single',
        False  # pro=False for non-multihop queries
    )
)
print(f"Answer: {result}")
```

### Example 2: Multi-Hop Query with Enhanced Precision

```python
# Multi-hop query benefits from FocusedRAG's density-guided selection
result = loop.run_until_complete(
    processor.local_retrieval(
        'What datasets were used for pre-training ViT, and what are the titles of their original papers?',
        'single',
        True  # pro=True for multi-hop queries
    )
)
print(f"Answer: {result}")
```

### Example 3: Cross-Paper Comparison with Large Token Budget

```python
# Increase token budget for comprehensive cross-paper analysis
custom_processor = PaperTreeRAG(config, retrieval_config={
    "use_focused_rag": True,
    "token_budget": 4000  # More tokens for cross-paper queries
})

result = loop.run_until_complete(
    custom_processor.local_retrieval(
        'Please compare and contrast the embedding modules in BERT and ViT',
        'cross',
        False
    )
)
print(f"Answer: {result}")
```

## Performance Characteristics

### Advantages

1. **Higher Precision**: Density-guided selection reduces irrelevant fragments
2. **Better Recall**: Surplus-based intervals preserve contextually important sentences
3. **Adaptive Behavior**: Variance maximization adjusts to query complexity
4. **Efficient Token Usage**: Budget-constrained aggregation maximizes information density
5. **Improved Coherence**: Document-order preservation maintains context flow

### Trade-offs

1. **Computational Overhead**: Additional processing for variance maximization and density analysis
2. **Memory Usage**: Requires numpy for statistical computations
3. **Latency**: Slight increase due to multi-stage pipeline

### When to Use FocusedRAG

**Best suited for**:
- Complex academic queries requiring precise evidence
- Multi-hop questions spanning multiple document sections
- Scenarios where retrieval precision is critical
- Large documents with abundant distractor content

**Consider disabling for**:
- Simple keyword-based queries
- Time-critical applications requiring minimal latency
- Very small documents where full-context retrieval is feasible

## Algorithm Details

### Section Localization Pipeline

1. **Signature Construction**: Concatenate section title + opening sentence
2. **Intent Alignment**: Compute relevance scores via reranking
3. **Score Sorting**: Arrange sections in descending relevance order
4. **Variance Maximization**: Find optimal partition boundary k*
5. **Section Selection**: Return top-k sections

### Sentence Selection Pipeline

1. **Baseline Computation**: Apply variance maximization to sentence scores → τ_sent
2. **Surplus Calculation**: Compute δ_j = v_j - τ_sent for each sentence
3. **Interval Extraction**: Identify maximal positive-cumulative-surplus intervals
4. **Topological Filtering**: Prefer multi-sentence intervals (length ≥ 2)
5. **Surplus Ranking**: Sort intervals by total surplus
6. **Budget Aggregation**: Greedily select until token budget exhausted

## Implementation Notes

### Dependencies

- **numpy**: Required for variance computation and statistical analysis
- **OpenAI API**: For LLM-based answer generation
- **Reranking API**: For computing relevance scores

### File Modifications

1. **focused_rag.py** (new): Core FocusedRAG implementation
2. **local_retriever.py** (modified): Integration with LocalRetriever
3. **paper_tree_rag.py** (modified): Configuration parameters
4. **requirements.txt** (modified): Added numpy dependency

### Backward Compatibility

The implementation maintains full backward compatibility:
- Original retrieval methods remain available
- FocusedRAG can be disabled via configuration
- Existing test scripts work without modification

## Testing and Validation

### Running Tests

```bash
# Test with FocusedRAG enabled (default)
cd /home/user/webapp && python test.py

# Test with original method (FocusedRAG disabled)
# Modify test.py to pass custom retrieval_config
```

### Validation Checklist

- [ ] Section localization selects relevant sections
- [ ] Sentence selection produces coherent evidence
- [ ] Token budget is respected
- [ ] Answers maintain accuracy and quality
- [ ] Performance is acceptable for use case

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'numpy'**
   ```bash
   pip install numpy
   ```

2. **FocusedRAG retrieval fails**
   - Check logs in `output/retrieval.log`
   - Verify reranking API is accessible
   - System automatically falls back to original method

3. **Token budget exceeded warnings**
   - Increase `token_budget` in configuration
   - Reduce `top_n` to limit candidate selection

4. **Poor retrieval quality**
   - Adjust `min_relevance_score` threshold
   - Experiment with different `token_budget` values
   - Consider query reformulation

## References

This implementation is based on the FocusedRAG methodology described in academic literature, featuring:

- Variance-maximization adaptive thresholding (inspired by Otsu's method)
- Density-guided evidence selection
- Budget-constrained aggregation

For detailed mathematical foundations and evaluation results, refer to the original research paper.

## Future Enhancements

Potential improvements for future versions:

1. **Dynamic token budgets**: Adaptive allocation based on query complexity
2. **Multi-modal support**: Integration with figure and table retrieval
3. **Caching**: Store frequently-accessed section partitions
4. **Parallel processing**: Concurrent section and sentence analysis
5. **Fine-tuned thresholds**: Query-specific parameter optimization

## Support

For questions, issues, or contributions related to FocusedRAG integration:

1. Check the main README.md for general PT-RAG usage
2. Review this document for FocusedRAG-specific guidance
3. Examine `focused_rag.py` for implementation details
4. Consult `output/retrieval.log` for diagnostic information
