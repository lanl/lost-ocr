"""
Nougat OCR Retrieval Evaluation with Distortion Analysis

This module evaluates retrieval performance for Nougat-extracted documents,
with special focus on how text distortion levels affect retrieval quality.
It implements NDCG@5 metrics and provides detailed breakdowns by distortion level.

Key Features:
- Document retrieval with text distortion tracking
- NDCG (Normalized Discounted Cumulative Gain) calculation
- Per-question and per-distortion-level analysis
- CSV output for detailed result analysis

Author: Alex Most, Manish Bhattarai
Date: 2025
Paper: Lost in OCR Translation (arXiv:2505.05666)
"""

import os
import json
import pickle
import pandas as pd
import numpy as np 
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ============================================================================
# SECTION 1: DOCUMENT LOADING WITH DISTORTION LEVELS
# ============================================================================

def gather_docs_with_distortion(filepath):
    """
    Load documents with their associated text distortion levels.
    
    This function reads Nougat-processed documents that have been annotated
    with distortion levels, allowing for analysis of OCR quality impact.
    
    Args:
        filepath (str): Path to JSONL file containing documents with distortion info
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - text: Normalized document text
            - file_name: Original filename
            - text_distortion_level: Distortion score (0-3)
    
    Text Processing:
        - Joins split text with single spaces
        - Removes extra whitespace for consistency
        - Preserves distortion metadata for analysis
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Normalize text by joining with single spaces
            text = " ".join(data["text"].split())
            
            # Extract metadata
            file_name = data.get("filename")
            distortion = data.get("text_distortion_level", -1)

            records.append({
                "text": text,
                "file_name": file_name,
                "text_distortion_level": distortion
            })
    
    return pd.DataFrame(records)

# ============================================================================
# SECTION 2: DOCUMENT EMBEDDING WITH CACHING
# ============================================================================

def embed_docs(df_docs, embedding_model, embeddings_file="doc_embeds.pkl"):
    """
    Generate or load cached document embeddings.
    
    This function handles the embedding process for documents, with intelligent
    caching to avoid redundant computation. It's optimized for large document
    sets with batch processing.
    
    Args:
        df_docs (pd.DataFrame): Documents to embed
        embedding_model: SentenceTransformer model instance
        embeddings_file (str): Path for embedding cache
    
    Returns:
        np.ndarray: Document embeddings (num_docs x embedding_dim)
    
    Caching Logic:
        1. Check if cached embeddings exist
        2. Verify cache validity (document count match)
        3. Re-embed if cache is invalid or missing
        4. Save new embeddings for future use
    """
    # Check for existing embeddings
    if os.path.exists(embeddings_file):
        print(f"Loading document embeddings from {embeddings_file}...")
        with open(embeddings_file, 'rb') as f:
            doc_embeddings = pickle.load(f)
        
        # Validate cache
        if len(doc_embeddings) == len(df_docs):
            return doc_embeddings
        else:
            print("Embedding size mismatch. Re-embedding documents...")

    # Generate new embeddings
    print("Embedding documents...")
    texts = df_docs["text"].fillna("").tolist()
    
    # Encode with progress bar and optimized batch size
    doc_embeddings = embedding_model.encode(
        texts, 
        show_progress_bar=True, 
        convert_to_numpy=True, 
        batch_size=2  # Slightly larger batch for Nougat text
    )
    
    # Save embeddings
    with open(embeddings_file, 'wb') as f:
        pickle.dump(doc_embeddings, f)
    
    return doc_embeddings

# ============================================================================
# SECTION 3: Q&A PAIR LOADING AND MATCHING
# ============================================================================

def gather_qa_pairs_from_matched_file(filepath, df_docs):
    """
    Load Q&A pairs and match them to documents with sanity checking.
    
    This function reads pre-matched Q&A pairs and links them to the correct
    documents in the corpus. It includes a sanity check to ensure proper
    filename matching before processing.
    
    Args:
        filepath (str): Path to matched Q&A pairs JSONL
        df_docs (pd.DataFrame): Document corpus DataFrame
    
    Returns:
        tuple: (questions, answers, ground_truth_indices)
            - questions: List of query strings
            - answers: List of ground truth answers
            - ground_truth_indices: Document indices for each Q&A pair
    
    Filename Normalization:
        The function handles different filename formats:
        - Removes .txt extensions
        - Converts .pdf- to .pdf.pdf-page- format
        This ensures compatibility across different processing pipelines
    """
    questions, answers, ground_truth_indices = [], [], []

    # Build filename lookup dictionary
    doc_filenames = df_docs["file_name"].tolist()
    file_name_to_idx = {fname: i for i, fname in enumerate(doc_filenames)}

    # Load Q&A data
    with open(filepath, 'r') as f:
        qa_data = [json.loads(line.strip()) for line in f]

    if not qa_data:
        print("No QA data found!")
        return [], [], []

    # --- Sanity check: Verify first file matches ---
    raw_file_name = qa_data[0].get("file_name", "")

    # Normalize filename for matching
    file_base = raw_file_name.replace(".txt", "")
    file_base = file_base.replace(".pdf-", ".pdf.pdf-page-")
    file_name = file_base

    print(f"Testing matching for QA file_name: {raw_file_name} -> normalized: {file_name}")

    if file_name in file_name_to_idx:
        print("✅ Sanity check: Match found.")
    else:
        print("❌ Sanity check: No match found!")
        print("First few doc filenames:")
        print(doc_filenames[:5])
        print("Exiting to avoid wasting time...")
        exit(1)

    # --- Process all Q&A pairs ---
    for data in qa_data:
        raw_file_name = data.get("file_name", "")

        # Normalize filename
        file_base = raw_file_name.replace(".txt", "")
        file_base = file_base.replace(".pdf-", ".pdf.pdf-page-")
        file_name = file_base

        qa_pairs = data.get("qa_pairs", [])

        # Skip if document not found
        if file_name not in file_name_to_idx:
            continue

        doc_idx = file_name_to_idx[file_name]

        # Extract individual Q&A pairs
        for qa in qa_pairs:
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            if q and a:
                questions.append(q)
                answers.append(a)
                ground_truth_indices.append(doc_idx)

    return questions, answers, ground_truth_indices

# ============================================================================
# SECTION 4: NDCG METRIC CALCULATION
# ============================================================================

def dcg_at_k(relevances, k):
    """
    Calculate Discounted Cumulative Gain at position k.
    
    DCG is a ranking quality metric that uses graded relevance scores,
    giving higher weight to relevant items at top positions.
    
    Args:
        relevances (array-like): Binary relevance scores (0 or 1)
        k (int): Position cutoff
    
    Returns:
        float: DCG@k score
    
    Formula:
        DCG@k = Σ(rel_i / log2(i+1)) for i=1 to k
        where rel_i is relevance at position i
    """
    relevances = np.asarray(relevances, dtype=float)[:k]
    if relevances.size:
        # Discount factor increases logarithmically with position
        return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
    return 0.

def ndcg_at_k(true_index, retrieved_indices, k=5):
    """
    Calculate Normalized Discounted Cumulative Gain at position k.
    
    NDCG normalizes DCG by the ideal DCG, providing a score between 0 and 1
    that's comparable across different queries.
    
    Args:
        true_index (int): Index of the relevant document
        retrieved_indices (list): Indices of retrieved documents (ranked)
        k (int): Position cutoff
    
    Returns:
        float: NDCG@k score (0 to 1, where 1 is perfect ranking)
    
    Interpretation:
        - 1.0: Perfect ranking (relevant doc at position 1)
        - 0.0: Relevant doc not in top-k
        - 0.5-0.8: Good retrieval (relevant doc in top positions)
    """
    # Create binary relevance list (1 for correct doc, 0 for others)
    relevances = [1 if idx == true_index else 0 for idx in retrieved_indices[:k]]
    
    # Ideal ranking would have the relevant doc first
    ideal_relevances = sorted(relevances, reverse=True)
    
    # Calculate actual and ideal DCG
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(ideal_relevances, k)
    
    # Normalize (avoid division by zero)
    return dcg / idcg if idcg > 0 else 0.

# ============================================================================
# SECTION 5: RETRIEVAL EVALUATION WITH DISTORTION ANALYSIS
# ============================================================================

def evaluate_question_retrieval(questions, ground_truth_indices, doc_embeddings, 
                               df_docs, embedding_model, top_k=5):
    """
    Comprehensive retrieval evaluation with distortion-level breakdown.
    
    This function evaluates retrieval performance across all questions and
    provides detailed analysis by text distortion level. It generates both
    aggregate metrics and fine-grained per-question results.
    
    Args:
        questions (list): Query questions
        ground_truth_indices (list): Correct document index for each question
        doc_embeddings (np.ndarray): Document embeddings
        df_docs (pd.DataFrame): Document metadata
        embedding_model: Model for encoding questions
        top_k (int): Number of documents to retrieve
    
    Returns:
        dict: Comprehensive metrics including:
            - Overall MRR, Recall@k, NDCG@5
            - Distortion-level breakdown
            - Question counts per distortion level
    
    Output Files:
        - per_question_retrieval_results.csv: Detailed per-question results
        - distortion_level_summary.csv: Aggregated metrics by distortion
    
    Analysis Levels:
        1. Overall: Aggregate metrics across all questions
        2. Per-distortion: Metrics grouped by distortion level
        3. Per-question: Individual question performance
    """
    print("Embedding questions...")
    # Encode all questions with optimized batch size
    q_embeddings = embedding_model.encode(
        questions, 
        show_progress_bar=True, 
        convert_to_numpy=True, 
        batch_size=1
    )

    # Extract document metadata
    texts_list = df_docs["text"].tolist()
    distortion_levels = df_docs["text_distortion_level"].tolist()

    # Initialize tracking variables
    ranks = []                    # Rank positions of correct documents
    ndcg_scores = []              # NDCG@5 scores
    distortion_ranks = {}         # Ranks grouped by distortion level
    distortion_ndcg = {}          # NDCG scores grouped by distortion
    question_records = []         # Detailed per-question results

    # Evaluate each question
    for i, q_emb in enumerate(q_embeddings):
        # Compute similarity scores with all documents
        sims = cosine_similarity([q_emb], doc_embeddings)[0]
        
        # Get top-k documents by similarity
        ranked_idx = sims.argsort()[::-1][:top_k]

        # Get ground truth document info
        gt_idx = ground_truth_indices[i]
        dist = distortion_levels[gt_idx]

        # Calculate rank position of correct document
        if gt_idx in ranked_idx:
            rank_pos = list(ranked_idx).index(gt_idx) + 1
        else:
            rank_pos = float('inf')  # Not in top-k

        # Calculate NDCG@5 for this question
        ndcg = ndcg_at_k(gt_idx, ranked_idx, k=top_k)

        # Store results
        ranks.append(rank_pos)
        ndcg_scores.append(ndcg)

        # Group by distortion level
        if dist not in distortion_ranks:
            distortion_ranks[dist] = []
            distortion_ndcg[dist] = []
        distortion_ranks[dist].append(rank_pos)
        distortion_ndcg[dist].append(ndcg)

        # Store detailed record for CSV output
        file_name = df_docs.iloc[gt_idx]["file_name"]
        question_records.append({
            "file": file_name,
            "RankPosition": rank_pos,
            "NDCG@5": ndcg,
            "DistortionLevel": dist
        })

    # Calculate overall metrics
    overall_mrr = sum(1/r for r in ranks if r != float('inf')) / len(ranks)
    overall_recall = sum(1 for r in ranks if r <= top_k) / len(ranks)
    overall_ndcg = sum(ndcg_scores) / len(ndcg_scores)

    # Calculate per-distortion metrics
    distortion_metrics = {}
    distortion_counts = {}

    for dist, rks in distortion_ranks.items():
        mrr = sum(1/r for r in rks if r != float('inf')) / len(rks)
        recall = sum(1 for r in rks if r <= top_k) / len(rks)
        avg_ndcg = sum(distortion_ndcg[dist]) / len(distortion_ndcg[dist])
        distortion_metrics[dist] = (mrr, recall, avg_ndcg)
        distortion_counts[dist] = len(rks)

    # Save per-question results
    question_df = pd.DataFrame(question_records)
    question_df.to_csv("per_question_retrieval_results.csv", index=False)
    print("📊 Saved per-question results to per_question_retrieval_results.csv")

    # Save distortion-level summary
    distortion_summary = pd.DataFrame([
        {
            "DistortionLevel": dist, 
            "Count": distortion_counts[dist], 
            "MRR": mrr, 
            "Recall@5": recall, 
            "NDCG@5": ndcg
        }
        for dist, (mrr, recall, ndcg) in distortion_metrics.items()
    ])
    distortion_summary.to_csv("distortion_level_summary.csv", index=False)
    print("📊 Saved distortion summary to distortion_level_summary.csv")

    return {
        "MRR": overall_mrr,
        "Recall@k": overall_recall,
        "NDCG@5": overall_ndcg,
        "Distortion Breakdown": distortion_metrics,
        "Distortion Counts": distortion_counts
    }

# ============================================================================
# SECTION 6: MAIN PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline for Nougat OCR retrieval evaluation.
    
    This function orchestrates the complete evaluation process with focus on
    analyzing how text distortion affects retrieval quality. It provides both
    overall metrics and detailed breakdowns by distortion level.
    
    Pipeline Steps:
        1. Load Nougat documents with distortion metadata
        2. Generate/load document embeddings
        3. Load matched Q&A pairs
        4. Evaluate retrieval with NDCG metrics
        5. Generate reports by distortion level
    
    Output:
        - Console output with overall and per-level metrics
        - CSV files for detailed analysis
        - Cached embeddings for reproducibility
    """
    # File paths
    doc_file = "nougat_text_with_distortion.jsonl"
    qa_file = "matched_docs.jsonl"

    # Step 1: Load documents with distortion info
    print("📚 Loading documents with distortion levels...")
    df_docs = gather_docs_with_distortion(doc_file)
    print(f"Loaded {len(df_docs)} documents.")
    
    # Display distortion distribution
    distortion_counts = df_docs['text_distortion_level'].value_counts().sort_index()
    print("\nDistortion level distribution:")
    for level, count in distortion_counts.items():
        print(f"  Level {level}: {count} documents")

    # Step 2: Initialize embedding model
    print("\n🤖 Initializing embedding model...")
    embedding_model = SentenceTransformer(
        "Alibaba-NLP/gte-Qwen2-7B-instruct", 
        trust_remote_code=True
    )
    
    # Step 3: Generate/load document embeddings
    print("\n🔢 Processing document embeddings...")
    doc_embeddings = embed_docs(
        df_docs, 
        embedding_model, 
        embeddings_file="doc_embeds_text_distortion.pkl"
    )

    # Step 4: Load Q&A pairs
    print("\n❓ Loading Q&A pairs...")
    questions, answers, ground_truth_indices = gather_qa_pairs_from_matched_file(
        qa_file, df_docs
    )
    print(f"Loaded {len(questions)} questions.")

    # Step 5: Evaluate retrieval
    print("\n🔍 Evaluating retrieval performance...")
    retrieval_metrics = evaluate_question_retrieval(
        questions, 
        ground_truth_indices, 
        doc_embeddings, 
        df_docs, 
        embedding_model, 
        top_k=5
    )

    # Step 6: Display results
    print("\n" + "="*60)
    print("📊 OVERALL RETRIEVAL METRICS")
    print("="*60)
    print(f"MRR: {retrieval_metrics['MRR']:.4f}")
    print(f"Recall@5: {retrieval_metrics['Recall@k']:.4f}")
    print(f"NDCG@5: {retrieval_metrics['NDCG@5']:.4f}")

    print("\n" + "="*60)
    print("📊 BREAKDOWN BY TEXT DISTORTION LEVEL")
    print("="*60)
    
    for dist_level, (mrr, recall, ndcg) in retrieval_metrics["Distortion Breakdown"].items():
        count = retrieval_metrics["Distortion Counts"].get(dist_level, 0)
        print(f"\n📈 Distortion Level {dist_level} ({count} questions):")
        print(f"   MRR: {mrr:.4f}")
        print(f"   Recall@5: {recall:.4f}")
        print(f"   NDCG@5: {ndcg:.4f}")
    
    # Analyze trend
    print("\n💡 TREND ANALYSIS:")
    levels = sorted(retrieval_metrics["Distortion Breakdown"].keys())
    if len(levels) > 1:
        first_level_ndcg = retrieval_metrics["Distortion Breakdown"][levels[0]][2]
        last_level_ndcg = retrieval_metrics["Distortion Breakdown"][levels[-1]][2]
        degradation = (first_level_ndcg - last_level_ndcg) / first_level_ndcg * 100
        print(f"NDCG degradation from lowest to highest distortion: {degradation:.1f}%")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
