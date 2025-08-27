"""
RAG (Retrieval-Augmented Generation) Pipeline Evaluation for LLaMA OCR

This module implements a complete RAG evaluation pipeline specifically for LLaMA-extracted
text documents. It measures retrieval performance, answer quality, and hallucination metrics
while tracking computational efficiency (embedding/retrieval times and GPU memory usage).

Key Components:
1. Document embedding using sentence transformers
2. Semantic retrieval with cosine similarity
3. LLM-based answer generation
4. Multi-metric evaluation (MRR, BLEU, ROUGE, Exact Match)
5. GPU memory monitoring for resource optimization

Author: Alex Most, Manish Bhattarai
Date: 2025
Paper: Lost in OCR Translation (arXiv:2505.05666)
"""

import os
import glob
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import time

# ============================================================================
# GPU MEMORY MONITORING
# ============================================================================

def log_gpu_memory_usage(note=""):
    """
    Log current GPU memory usage for performance monitoring.
    
    This function helps track memory consumption during different stages of the
    pipeline, which is crucial for optimizing batch sizes and preventing OOM errors.
    
    Args:
        note (str): Optional note to identify the current operation stage
    
    Output:
        Prints allocated and reserved memory for each available GPU
    """
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)  # Convert to MB
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)    # Convert to MB
        print(f"[GPU {i}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB ({note})")

# ============================================================================
# LLM CLIENT CONFIGURATION
# ============================================================================

# Initialize OpenAI-compatible client for SambaNova API
# This client is used for the generation component of RAG
client = OpenAI(
    api_key="your-api-key-here",  # Replace with actual API key
    base_url="https://api.sambanova.ai/v1",
)

# ============================================================================
# EMBEDDING MANAGEMENT
# ============================================================================

def save_embeddings(embeddings, file_path):
    """
    Save embeddings to disk for caching.
    
    Args:
        embeddings: Numpy array of document embeddings
        file_path (str): Path to save pickle file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(file_path):
    """
    Load cached embeddings from disk.
    
    Args:
        file_path (str): Path to pickle file
    
    Returns:
        Numpy array of embeddings
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# ============================================================================
# DOCUMENT GATHERING AND PREPROCESSING
# ============================================================================

def gather_docs(folder_path, min_words=40):
    """
    Build a DataFrame of valid documents from JSONL files.
    
    This function reads Q&A generation output files and extracts documents,
    applying normalization and filtering to ensure quality.
    
    Args:
        folder_path (str): Directory containing JSONL files
        min_words (int): Minimum word count threshold for valid documents
    
    Returns:
        pd.DataFrame: DataFrame with 'abstract' and 'file_name' columns
    
    Processing Steps:
        1. Read all JSONL files in folder
        2. Normalize abstracts (collapse whitespace)
        3. Filter by minimum word count
        4. Remove duplicates
    """
    records = []
    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))

    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())

                # Normalize abstract by collapsing multiple spaces
                raw_abstract = " ".join(data.get("abstract", "").split())

                # Skip if too short or empty
                if len(raw_abstract.split()) < min_words:
                    continue

                records.append({
                    "abstract": raw_abstract,
                    "file_name": data.get("file_title", os.path.basename(jsonl_file))
                })

    # Remove duplicate abstracts to avoid redundant embeddings
    df_docs = pd.DataFrame(records).drop_duplicates(subset=["abstract"])
    return df_docs

# ============================================================================
# DOCUMENT EMBEDDING
# ============================================================================

def embed_docs(df_docs, embedding_model, embeddings_file="doc_embeds2.pkl"):
    """
    Generate or load embeddings for documents with timing analysis.
    
    This function handles the compute-intensive embedding generation process,
    including caching, GPU memory management, and performance timing.
    
    Args:
        df_docs (pd.DataFrame): Documents to embed
        embedding_model: SentenceTransformer model instance
        embeddings_file (str): Path for embedding cache
    
    Returns:
        tuple: (embeddings array, average time per document or None if cached)
    
    Features:
        - Automatic caching to avoid re-computation
        - GPU memory tracking for optimization
        - Timing analysis for performance benchmarking
        - Batch size optimization (batch_size=1 for memory efficiency)
    """
    texts = df_docs["abstract"].fillna("").tolist()

    # Check for cached embeddings
    if os.path.exists(embeddings_file):
        print(f"Loading doc embeddings from {embeddings_file}...")
        with open(embeddings_file, 'rb') as f:
            doc_embeddings = pickle.load(f)
        
        # Verify cache validity
        if len(doc_embeddings) == len(df_docs):
            log_gpu_memory_usage("After loading embeddings")
            return doc_embeddings, None
        else:
            print("Mismatch in number of docs. Re-embedding...")

    # Generate new embeddings with performance monitoring
    print("Embedding documents...")
    log_gpu_memory_usage("Before embedding documents")
    start_time = time.time()
    
    # Clear GPU cache to ensure accurate memory measurements
    torch.cuda.empty_cache()
    
    # Generate embeddings with progress bar
    doc_embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=1  # Conservative batch size to prevent OOM
    )

    # Calculate timing statistics
    total_time = time.time() - start_time
    avg_time_per_doc = total_time / len(texts)
    print(f"Total Embedding Time: {total_time:.2f} seconds")
    print(f"Average Embedding Time per document: {avg_time_per_doc:.4f} seconds")

    # Save embeddings for future use
    with open(embeddings_file, 'wb') as f:
        pickle.dump(doc_embeddings, f)
    
    log_gpu_memory_usage("After embedding documents")
    return doc_embeddings, avg_time_per_doc

# ============================================================================
# Q&A PAIR GATHERING
# ============================================================================

def gather_qa_pairs(folder_path, df_docs):
    """
    Extract Q&A pairs and map them to documents.
    
    This function reads Q&A generation outputs and matches them to the
    normalized abstracts in df_docs, handling format variations and errors.
    
    Args:
        folder_path (str): Directory containing Q&A JSONL files
        df_docs (pd.DataFrame): Document DataFrame for matching
    
    Returns:
        tuple: (questions list, answers list, ground truth indices list)
    
    Processing:
        - Normalizes abstracts for consistent matching
        - Handles JSON parsing errors gracefully
        - Skips unmatched abstracts with counting
        - Maps each Q&A pair to its source document index
    """
    questions = []
    answers = []
    ground_truth_indices = []
    unmatched_count = 0

    # Build lookup table for efficient matching
    abstracts_list = df_docs["abstract"].tolist()
    abstract_to_idx = {ab: i for i, ab in enumerate(abstracts_list)}

    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
    
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())

                # Normalize abstract for matching
                raw_abstract = " ".join(data.get("abstract", "").split())

                # Skip if abstract not in document set
                if raw_abstract not in abstract_to_idx:
                    unmatched_count += 1
                    continue

                doc_idx = abstract_to_idx[raw_abstract]
                
                # Parse Q&A pairs (handle both dict and string formats)
                qa_pairs = data.get("qa_pairs", [])
                if isinstance(qa_pairs, str):
                    try:
                        qa_pairs = json.loads(qa_pairs.strip())
                    except:
                        qa_pairs = []

                # Extract individual Q&A pairs
                for qa in qa_pairs:
                    q = qa.get("question", "").strip()
                    a = qa.get("answer", "").strip()
                    if not q or not a:
                        continue
                    questions.append(q)
                    answers.append(a)
                    ground_truth_indices.append(doc_idx)

    print(f"Skipped {unmatched_count} entries with unmatched abstracts.")
    return questions, answers, ground_truth_indices

# ============================================================================
# RETRIEVAL EVALUATION
# ============================================================================

def evaluate_retrieval(questions, ground_truth_indices, doc_embeddings, df, 
                      embedding_model, top_k=5):
    """
    Evaluate retrieval performance with detailed metrics and timing.
    
    This function performs semantic retrieval for each question and calculates
    various performance metrics including MRR, Recall@k, and timing statistics.
    
    Args:
        questions (list): List of query questions
        ground_truth_indices (list): Correct document index for each question
        doc_embeddings: Numpy array of document embeddings
        df (pd.DataFrame): Document DataFrame
        embedding_model: Model for encoding questions
        top_k (int): Number of documents to retrieve
    
    Returns:
        dict: Comprehensive evaluation metrics including:
            - MRR: Mean Reciprocal Rank
            - Recall@k: Proportion of queries with correct doc in top-k
            - Avg Retrieval Time: Average time per query
            - Results DataFrame: Detailed results for analysis
    
    Output Files:
        - retrieval_results_text.csv: Detailed results for each query
    """
    abstracts_list = df["abstract"].tolist()
    file_names = df["file_name"].tolist()

    log_gpu_memory_usage("Before question embedding")

    print("Embedding questions for retrieval...")
    retrieval_times = []
    ranks = []
    retrieved_docs = []
    results = []

    # Encode all questions at once for efficiency
    q_embeddings = embedding_model.encode(
        questions, 
        show_progress_bar=True, 
        convert_to_numpy=True, 
        batch_size=1
    )

    log_gpu_memory_usage("After question embedding")

    # Evaluate each question
    for i, q_emb in enumerate(q_embeddings):
        start_time = time.time()

        # Calculate cosine similarity with all documents
        sims = cosine_similarity([q_emb], doc_embeddings)[0]
        
        # Get top-k most similar documents
        ranked_idx = sims.argsort()[::-1][:top_k]

        retrieval_time = time.time() - start_time
        retrieval_times.append(retrieval_time)

        # Extract retrieved document texts
        top_docs = [abstracts_list[idx] for idx in ranked_idx]
        retrieved_docs.append(top_docs)

        # Calculate rank of correct document
        gt_idx = ground_truth_indices[i]
        rank_pos = list(ranked_idx).index(gt_idx) + 1 if gt_idx in ranked_idx else float('inf')
        
        # Calculate reciprocal rank score
        score = 1 / rank_pos if rank_pos != float('inf') else 0

        ranks.append(rank_pos)

        # Store detailed results for analysis
        results.append({
            "Question": questions[i],
            "Ground Truth File": file_names[gt_idx],
            "Rank Position": rank_pos,
            "Score": score,
            "Retrieval Time (sec)": retrieval_time,
            "Top-k Retrieved Files": [file_names[idx] for idx in ranked_idx]
        })

    # Calculate aggregate metrics
    avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
    print(f"Average Retrieval Time per query: {avg_retrieval_time:.4f} seconds")

    log_gpu_memory_usage("After retrieval process completed")

    # Calculate MRR and Recall@k
    mrr = sum(1/r for r in ranks if r != float('inf')) / len(ranks)
    recall_at_k = sum(1 for r in ranks if r <= top_k) / len(ranks)

    # Save detailed results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("retrieval_results_text.csv", index=False)

    return {
        "MRR": mrr,
        "Recall@k": recall_at_k,
        "Avg Retrieval Time": avg_retrieval_time,
        "Ranks": ranks,
        "Retrieved Docs": retrieved_docs,
        "Results DataFrame": results_df
    }

# ============================================================================
# LLM QUERY WITH RAG
# ============================================================================

def query_llm_with_rag(query, retrieved_docs):
    """
    Generate answer using LLM with retrieved context (RAG).
    
    This function implements the generation component of RAG, using retrieved
    documents as context for the LLM to generate accurate answers.
    
    Args:
        query (str): User question
        retrieved_docs (list): Top-k retrieved document texts
    
    Returns:
        str: Generated answer or 'NO RESPONSE' on failure
    
    Prompt Engineering:
        - Structures context clearly with numbered documents
        - Uses system message for role definition
        - Low temperature (0.3) for factual consistency
        - Limited tokens (150) for concise answers
    """
    # Build structured prompt with retrieved context
    context = "\n\n".join(f"Context {i+1}: {doc}" for i, doc in enumerate(retrieved_docs))
    prompt = (
        f"Given the following context, answer the query.\n\n"
        f"Query: {query}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:"
    )

    # Call LLM API with defensive error handling
    resp = client.chat.completions.create(
        model="Meta-Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for answering user queries."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,  # Low temperature for consistency
        max_tokens=150    # Limit response length
    )

    # Defensive checks for API response structure
    if resp is None:
        print("Warning: LLM response is None.")
        return "NO RESPONSE"

    if not resp.choices or len(resp.choices) == 0:
        print("Warning: LLM response has no choices.")
        return "NO RESPONSE"

    choice = resp.choices[0]
    if choice.message is None:
        print("Warning: LLM choice has no 'message' field.")
        return "NO RESPONSE"

    content = choice.message.content
    if content is None:
        print("Warning: LLM choice.message has no 'content'.")
        return "NO RESPONSE"

    return content.strip()

# ============================================================================
# SPECIALIZED DOCUMENT LOADING FOR MATCHED FILES
# ============================================================================

def gather_docs_from_matched_file(filepath):
    """
    Load documents from a pre-matched JSONL file.
    
    This specialized function reads documents that have already been matched
    with Q&A pairs, ensuring consistency in evaluation.
    
    Args:
        filepath (str): Path to matched_docs.jsonl
    
    Returns:
        pd.DataFrame: Documents with normalized abstracts
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            abstract = " ".join(data["abstract"].split())
            file_name = data.get("file_name")
            records.append({
                "abstract": abstract,
                "file_name": file_name
            })
    return pd.DataFrame(records).drop_duplicates(subset=["abstract"])

def gather_qa_pairs_from_matched_file(filepath, df_docs):
    """
    Extract Q&A pairs from pre-matched file.
    
    Args:
        filepath (str): Path to matched_docs.jsonl
        df_docs (pd.DataFrame): Document DataFrame
    
    Returns:
        tuple: (questions, answers, ground_truth_indices)
    """
    questions, answers, ground_truth_indices = [], [], []

    abstracts_list = df_docs["abstract"].tolist()
    abstract_to_idx = {ab: i for i, ab in enumerate(abstracts_list)}

    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            raw_abstract = " ".join(data["abstract"].split())
            qa_pairs = data["qa_pairs"]

            if raw_abstract not in abstract_to_idx:
                continue

            doc_idx = abstract_to_idx[raw_abstract]

            for qa in qa_pairs:
                q = qa.get("question", "").strip()
                a = qa.get("answer", "").strip()
                if q and a:
                    questions.append(q)
                    answers.append(a)
                    ground_truth_indices.append(doc_idx)

    return questions, answers, ground_truth_indices

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline for RAG evaluation.
    
    This function orchestrates the complete evaluation process:
    1. Load documents and Q&A pairs
    2. Generate/load embeddings
    3. Evaluate retrieval performance
    4. Generate answers using LLM
    5. Calculate final metrics
    
    Features:
        - Resumable processing (saves progress to CSV)
        - Comprehensive metrics tracking
        - GPU memory monitoring
        - Timing analysis for all components
    """
    # Configuration
    matched_docs_file = "matched_docs.jsonl"

    # Step 1: Load documents
    print("📚 Loading documents...")
    df_docs = gather_docs_from_matched_file(matched_docs_file)
    print(f"Loaded {len(df_docs)} documents")

    # Step 2: Initialize embedding model
    print("🤖 Initializing embedding model...")
    embedding_model = SentenceTransformer(
        "Alibaba-NLP/gte-Qwen2-7B-instruct", 
        trust_remote_code=True
    )

    # Step 3: Generate/load embeddings
    print("🔢 Processing document embeddings...")
    doc_embeddings, avg_doc_embed_time = embed_docs(
        df_docs, 
        embedding_model, 
        embeddings_file="doc_embeds_llama_text.pkl"
    )

    if avg_doc_embed_time is not None:
        print(f"Avg embedding time per document: {avg_doc_embed_time:.4f} sec")
    else:
        print("Embeddings loaded from disk, no embedding time computed.")

    # Step 4: Load Q&A pairs
    print("❓ Loading Q&A pairs...")
    questions, answers, ground_truth_indices = gather_qa_pairs_from_matched_file(
        matched_docs_file, df_docs
    )

    if not questions:
        print("No Q&A pairs found.")
        return

    print(f"# Documents: {len(df_docs)}")
    print(f"# Questions: {len(questions)}")
    print(f"Avg questions per doc: {len(questions)/len(df_docs):.2f}")

    # Step 5: Check for existing responses (for resumable processing)
    responses_file = "llm_responses_text.csv"
    if os.path.exists(responses_file):
        df_saved = pd.read_csv(responses_file)
        llm_responses = df_saved["LLM Response"].tolist()
        start_idx = len(llm_responses)
        print(f"Resuming from saved responses: {start_idx}/{len(questions)}")
    else:
        llm_responses = []
        start_idx = 0

    # Step 6: Evaluate retrieval
    print("🔍 Evaluating retrieval performance...")
    retrieval = evaluate_retrieval(
        questions, ground_truth_indices, doc_embeddings, df_docs, 
        embedding_model, top_k=5
    )

    avg_query_retrieval_time = retrieval["Avg Retrieval Time"]
    print(f"Average Query Retrieval Time: {avg_query_retrieval_time:.4f} sec/query")
    print(f"MRR: {retrieval['MRR']:.4f}")
    print(f"Recall@5: {retrieval['Recall@k']:.4f}")

    # Step 7: Generate answers using RAG (if needed)
    print("💬 Generating answers with LLM...")
    from tqdm import trange

    for i in trange(start_idx, len(questions), desc="Querying LLaMA"):
        top_docs = retrieval["Retrieved Docs"][i]
        resp = query_llm_with_rag(questions[i], top_docs)
        llm_responses.append(resp)

        # Save progress every 100 questions or at the end
        if i % 100 == 0 or i == len(questions) - 1:
            pd.DataFrame({
                "Question": questions[:i + 1],
                "LLM Response": llm_responses,
                "Retrieved Context": [
                    "\n\n".join(retrieval["Retrieved Docs"][j]) for j in range(i + 1)
                ]
            }).to_csv(responses_file, index=False)
            print(f"Progress saved: {i+1}/{len(questions)}")

    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
