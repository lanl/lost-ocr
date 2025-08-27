"""
Nougat + ViDoRe Utilities Module

Bridges Nougat OCR and ViDoRe benchmark formats. Provides conversion, evaluation helpers, and reporting for OCR-based baselines on ViDoRe.

Key Features:
- Dataset preparation and adapters
- Metric reporting compatible with ViDoRe
- Lightweight CLI-style entry points

Author: Alex Most, Manish Bhattarai
Date: 2025
Paper: Lost in OCR Translation (arXiv:2505.05666)
"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from transformers import AutoTokenizer

# ─── CONFIGURATION ────────────────────────────────────────────────
OCR_DIR       = "vidore_texts_nougat_final"
DATASET_NAME  = "vidore/docvqa_test_subsampled_tesseract"
MODEL_NAME    = "Alibaba-NLP/gte-Qwen2-7B-instruct"
MAX_TOKENS    = 512
DEVICE        = "cpu"  # safer to start on CPU
TOP_K         = 5
# ──────────────────────────────────────────────────────────────────

# 1) Load OCR texts
print(" Loading OCR texts...")
ocr_texts = {}
for fname in os.listdir(OCR_DIR):
    if fname.endswith(".txt"):
        file_id = fname[:-4]
        with open(os.path.join(OCR_DIR, fname), 'r', encoding='utf-8') as f:
            ocr_texts[file_id] = f.read().strip()
print(f"→ Loaded {len(ocr_texts)} OCR documents.")

# 2) Load DocVQA dataset and match OCR docs
print("Loading and matching DocVQA questions...")
ds = load_dataset(DATASET_NAME, split='test')
queries, gt_file_ids = [], []
for ex in ds:
    file_id = ex["image_filename"]
    if file_id in ocr_texts:
        queries.append(ex["query"])
        gt_file_ids.append(file_id)
print(f" Matched {len(queries)} queries to OCR docs.")

# 3) Prepare tokenizer and truncate texts
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def truncate(text, max_tokens=MAX_TOKENS):
    tokens = tokenizer.encode(text, max_length=max_tokens, truncation=True)
    return tokenizer.decode(tokens, skip_special_tokens=True)

truncated_texts = [truncate(ocr_texts[fid]) for fid in gt_file_ids]
truncated_queries = [truncate(q) for q in queries]

# 4) Embed documents and queries (batch=1, CPU)
print(" Embedding model loading...")
embedder = SentenceTransformer(MODEL_NAME, device=DEVICE)

print(" Embedding OCR documents...")
doc_embeds = embedder.encode(
    truncated_texts,
    convert_to_numpy=True,
    batch_size=1,
    show_progress_bar=True
)

print("Embedding queries...")
query_embeds = embedder.encode(
    truncated_queries,
    convert_to_numpy=True,
    batch_size=1,
    show_progress_bar=True
)

# 5) Cosine similarity retrieval (top-K)
print(f" Computing top-{TOP_K} retrieval...")
cos_sim = cosine_similarity(query_embeds, doc_embeds)
topk_indices = np.argsort(-cos_sim, axis=1)[:, :TOP_K]

# 6) Evaluate Retrieval with NDCG@5 and Recall@5
def dcg(rank):  # rank is zero-based
    return 1 / np.log2(rank + 2)

ideal_dcg = dcg(0)
ndcgs, hits = [], 0
for idx, retrieved_idxs in enumerate(topk_indices):
    if idx in retrieved_idxs:
        rank = list(retrieved_idxs).index(idx)
        ndcgs.append(dcg(rank) / ideal_dcg)
        hits += 1
    else:
        ndcgs.append(0.0)

avg_ndcg = np.mean(ndcgs)
recall_at_5 = hits / len(queries)

print("\n Evaluation Results:")
print(f"NDCG@5   : {avg_ndcg:.4f}")
print(f"Recall@5 : {recall_at_5:.4f}")

