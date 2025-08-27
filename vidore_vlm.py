"""
Vision-Based Document Retrieval Evaluation using ViDoRe Benchmark

This module evaluates vision-based document retrieval models (ColQwen2, ColPali) 
using the ViDoRe benchmark framework. Unlike OCR-based approaches, these models
directly process document images without text extraction, demonstrating superior
robustness to document degradation.

Key Components:
- ColQwen2/ColPali retriever initialization
- ViDoRe benchmark evaluation
- Direct visual document encoding
- Comparison with OCR-based methods

Author: Alex Most
Date: 2025
Paper: Lost in OCR Translation (arXiv:2505.05666)
"""

from vidore_benchmark.retrievers import ColQwen2Retriever, ColPaliRetriever
from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorQA
from datasets import load_dataset
import torch

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

print("🤖 Loading ColQwen2Retriever")

# Initialize vision-based retriever
# ColQwen2 is a vision-language model that directly encodes document images
# without requiring OCR, making it robust to text degradation
retriever = ColQwen2Retriever(
    pretrained_model_name_or_path="vidore/colqwen2-v1.0",
    device="cuda" if torch.cuda.is_available() else "cpu",
    num_workers=0  # Set to 0 to avoid multiprocessing issues
)

def patched_get_scores(query_embeddings, passage_embeddings, **kwargs):
    """
    Patched scoring function for batch size control.
    
    This patch ensures the retriever processes embeddings with a controlled
    batch size to prevent memory issues on limited GPU resources.
    
    Args:
        query_embeddings: Embedded query representations
        passage_embeddings: Embedded document representations
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Similarity scores between queries and passages
    
    Note:
        This patch is necessary due to a known issue in the ViDoRe library
        where batch size isn't properly controlled in the default implementation
    """
    return ColQwen2Retriever.get_scores(
        retriever, 
        query_embeddings, 
        passage_embeddings, 
        batch_size=1  # Force batch size of 1 for stability
    )

# Apply the patch to the retriever instance
retriever.get_scores = patched_get_scores

# ============================================================================
# DATASET LOADING
# ============================================================================

print("📚 Loading ViDoRe dataset")

# Load the DocVQA test subset from ViDoRe benchmark
# Two options available:
# 1. "vidore/docvqa_test_subsampled_tesseract" - includes OCR baseline
# 2. "vidore/docvqa_test_subsampled" - vision-only evaluation
dataset = load_dataset("vidore/docvqa_test_subsampled", split="test")

print(f"Dataset loaded: {len(dataset)} samples")

# ============================================================================
# EVALUATOR SETUP
# ============================================================================

print("⚙️ Initializing evaluator")

# Create ViDoRe evaluator for question-answering tasks
# This evaluator handles the complete evaluation pipeline including:
# - Query encoding
# - Document retrieval
# - Metric calculation
evaluator = ViDoReEvaluatorQA(retriever)

# ============================================================================
# EVALUATION EXECUTION
# ============================================================================

print("🔍 Running evaluation...")
print("This may take several minutes depending on dataset size and GPU...")

# Run the evaluation with specified batch sizes
# These parameters control memory usage and processing efficiency
results = evaluator.evaluate_dataset(
    ds=dataset,
    batch_query=16,              # Number of queries to process together
    batch_passage=1,             # Number of passages to encode together
    dataloader_prebatch_query=1,    # Dataloader batch size for queries
    dataloader_prebatch_passage=2,   # Adjust to 1 if OOM occurs
    k=5,                         # Top-k documents to retrieve
    batch_size=1                 # Overall batch size for scoring
)

# ============================================================================
# RESULTS DISPLAY
# ============================================================================

print("\n" + "="*50)
print("📊 EVALUATION RESULTS")
print("="*50)

# Display all calculated metrics
for metric, value in results.items():
    # Format metric name for better readability
    metric_display = metric.replace("_", " ").title()
    print(f"{metric_display}: {value:.4f}")

# Interpret key metrics
print("\n💡 METRIC INTERPRETATION:")
print("- NDCG@5: Normalized Discounted Cumulative Gain (quality of ranking)")
print("- MRR: Mean Reciprocal Rank (average rank of first relevant document)")
print("- Recall@5: Proportion of queries with relevant doc in top 5")
print("- Higher values indicate better performance")

# Compare with typical OCR-based performance
print("\n📈 PERFORMANCE CONTEXT:")
print("Vision-based models like ColQwen2 typically maintain performance")
print("across document degradation levels, unlike OCR-based approaches")
print("which show significant performance drops with increased distortion.")

print("\n✅ Evaluation complete!")
