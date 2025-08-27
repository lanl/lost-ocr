"""
LLaMA ViDoRe Evaluation Module

Runs LLaMA-based evaluation on ViDoRe tasks, integrating retrieval outputs to produce answers and measure performance across visual document datasets.

Key Features:
- Support for hosted LLaMA endpoints
- Batch inference with rate-limit resilience
- Structured outputs for downstream scoring

Author: Alex Most, Manish Bhattarai
Date: 2025
Paper: Lost in OCR Translation (arXiv:2505.05666)
"""
import os
import json
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import torch

from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoProcessor
from sentence_transformers import SentenceTransformer
from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorQA


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load ViDoRe dataset ===
print(" Loading ViDoRe dataset...")

dataset = load_dataset("vidore/docvqa_test_subsampled_tesseract", split="test")

# === Step 2: Load LLaMA-generated text files ===
import time
start_time = time.time()

TEXT_FOLDER = "/vast/home/amost/docVQA_txt/docVQA_txt_4096"

print(" Loading LLaMA text outputs from directory...")

ocr_results = {}
missing_files = []

for item in tqdm(dataset):
    image_filename = item["image_filename"]  # e.g., "fsdw0217.png"
    text_filename = os.path.splitext(image_filename)[0] + ".txt"
    text_file_path = os.path.join(TEXT_FOLDER, text_filename)

    if os.path.exists(text_file_path):
        with open(text_file_path, "r", encoding="utf-8") as f:
            text = f.read()
        ocr_results[image_filename] = text
    else:
        missing_files.append(image_filename)
        ocr_results[image_filename] = "[MISSING TEXT FILE]"

if missing_files:
    print(f" WARNING: {len(missing_files)} files were missing corresponding .txt files.")
    print(f" Missing examples: {missing_files[:5]}...")

print(f" Loaded all text documents in {time.time() - start_time:.2f} seconds.")

first_doc_key = list(ocr_results.keys())[0]
print("Example doc content:", ocr_results[first_doc_key][:500])

# === Step 3: Build document list from OCR outputs ===
print(" Building documents from OCR results...")
documents = [ocr_results[item["image_filename"]] for item in dataset]
questions = [item["query"] for item in dataset]

# === Step 4: Define Retriever ===
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class MyRetriever:
    def __init__(self):
        print(" Loading sentence-embedding model …")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model  = SentenceTransformer(
            "Alibaba-NLP/gte-Qwen2-7B-instruct",
            trust_remote_code=True
        ).to(self.device)
        self.use_visual_embedding = False                         # kept for API parity

    # ---------- fit -----------------------------------------------------------------
    def fit(self, corpus_texts, batch_size: int = 2):
        """Encode and L2-normalise the corpus once."""
        print(" Encoding corpus …")
        self.corpus = corpus_texts
        with torch.inference_mode():
            emb = self.model.encode(
                corpus_texts,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=batch_size,
                device=self.device
            )
            self.embeddings = F.normalize(emb, p=2, dim=1)        # <- unit length

    # ---------- retrieve -------------------------------------------------------------
    def retrieve(self, queries, k: int = 5, batch_size: int = 2):
        """Return top-k passages for each query using cosine similarity."""
        print(" Retrieving top-k passages …")
        with torch.inference_mode():
            q_emb = self.model.encode(
                queries,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=batch_size,
                device=self.device
            )
            q_emb = F.normalize(q_emb, p=2, dim=1)                # <- unit length

        # cosine similarity == dot product for unit vectors
        scores = q_emb @ self.embeddings.T                        # shape (B, |corp|)
        top_k = torch.topk(scores, k, dim=1).indices.cpu().tolist()

        return [[self.corpus[idx] for idx in row] for row in top_k]

    # ---------- helpers used by ViDoReEvaluatorQA -----------------------------------
    def forward_queries(self, queries, batch_size: int = 2):
        with torch.inference_mode():
            q = self.model.encode(
                queries, convert_to_tensor=True,
                show_progress_bar=True, batch_size=batch_size,
                device=self.device
            )
        return F.normalize(q, p=2, dim=1)

    def forward_passages(self, passages, batch_size: int = 2):
        with torch.inference_mode():
            p = self.model.encode(
                passages, convert_to_tensor=True,
                show_progress_bar=True, batch_size=batch_size,
                device=self.device
            )
        return F.normalize(p, p=2, dim=1)

    def get_scores(self, query_embeddings, passage_embeddings, batch_size=None):
        """Return cosine-similarity matrix."""
        if isinstance(query_embeddings, list):
            query_embeddings = torch.stack(query_embeddings)
        if isinstance(passage_embeddings, list):
            passage_embeddings = torch.stack(passage_embeddings)

        # both sets are already normalised
        return query_embeddings @ passage_embeddings.T

# === Step 5: Run Evaluation ===



print(" Running RAG evaluation on OCR outputs...")
retriever = MyRetriever()
retriever.fit(documents)

# Sanity check embeddings are changing:
print("Sample embedding for first document (first 5 values):",
      retriever.embeddings[0][:5])


evaluator = ViDoReEvaluatorQA(retriever)

results = evaluator.evaluate_dataset(
    ds=dataset,
    batch_query=1,
    batch_passage=1,
    dataloader_prebatch_query=1,
    dataloader_prebatch_passage=1,
    k=5
)

# === Results ===
print("\n === Evaluation Results ===")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

