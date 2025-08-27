"""
RAG Pipeline (Index, Retrieve, Generate) Module

End-to-end Retrieval-Augmented Generation pipeline for visually rich documents. Builds indices, retrieves top-k passages, and generates answers with an LLM. Designed for experiments comparing OCR-based and vision-only retrieval.

Key Features:
- Flexible indexing with dense embeddings
- Top-k retrieval and reranking hooks
- Pluggable generator (LLM) interface
- Batch evaluation and result logging
- Deterministic runs with seed control

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
import openai
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# For embeddings, use Hugging Face Sentence Transformers
from sentence_transformers import SentenceTransformer

##########################################################
# 1) OpenAI-like LLM Client (SambaNova)
##########################################################
client = openai.OpenAI(
    api_key="api key here",
    base_url="https://api.sambanova.ai/v1",
)

##########################################################
# 2) Save / Load Embeddings
##########################################################
def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

##########################################################
# 3) Step: Gather All JSON Lines => Build a Doc Table
##########################################################

def gather_docs(folder_path, min_words=40):
    """
    Builds a DataFrame of valid documents from .jsonl in `folder_path`.
    Each line is something like:
      {
        "file_name": "...",
        "abstract": "...some text...",
        "qa_pairs": [...]
      }

    We skip lines that have an invalid or short `abstract`.
    """
    import glob, json, os
    records = []
    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))

    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())

                raw_abstract = data.get("abstract", "").strip()
                # Skip if no abstract or it's too short
                word_count = len(raw_abstract.split())
                if word_count < min_words:
                    continue

                # We also skip if 'abstract' is effectively empty
                if not raw_abstract:
                    continue

                # We'll store the doc in 'records'; 
                # We do NOT parse qa_pairs here, just building doc list.
                # If the same abstract appears multiple times, 
                # you can decide if you want to unify them or keep duplicates.
                records.append({
                    "abstract": raw_abstract,
                    "file_name": data.get("file_name", "")
                })

    df_docs = pd.DataFrame(records).drop_duplicates(subset=["abstract"])
    return df_docs


##########################################################
# 4) Embedding the Documents
##########################################################
def embed_docs(df_docs, embedding_model, embeddings_file="doc_embeds.pkl"):
    """
    Use huggingface embedding_model to encode each doc's abstract.
    """
    if os.path.exists(embeddings_file):
        print(f"Loading doc embeddings from {embeddings_file}...")
        with open(embeddings_file, 'rb') as f:
            doc_embeddings = pickle.load(f)
        # Optional consistency check:
        if len(doc_embeddings) == len(df_docs):
            return doc_embeddings
        else:
            print("Mismatch in number of docs. Re-embedding...")
    
    print("Embedding documents...")
    texts = df_docs["abstract"].fillna("").tolist()
    doc_embeddings = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size = 1)
    with open(embeddings_file, 'wb') as f:
        pickle.dump(doc_embeddings, f)
    return doc_embeddings


def gather_qa_pairs(folder_path, df_docs):
    """
    Goes through the same .jsonl files, looks for QA pairs,
    and tries to match them to the doc's abstract in df_docs.

    We skip lines that reference an abstract we didn't keep in df_docs
    (e.g. short or invalid ones).
    """
    import glob, json, os
    questions = []
    answers = []
    ground_truth_indices = []

    abstracts_list = df_docs["abstract"].tolist()
    abstract_to_idx = {ab: i for i, ab in enumerate(abstracts_list)}

    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                raw_abstract = data.get("abstract", "").strip()

                # If we didn't keep this doc in df_docs, skip
                if raw_abstract not in abstract_to_idx:
                    continue

                doc_idx = abstract_to_idx[raw_abstract]

                qa_pairs = data.get("qa_pairs", [])
                if isinstance(qa_pairs, str):
                    # parse if it's a string
                    try:
                        qa_pairs = json.loads(qa_pairs.strip())
                    except:
                        qa_pairs = []

                for qa in qa_pairs:
                    questions.append(qa.get("question", ""))
                    answers.append(qa.get("answer", ""))
                    ground_truth_indices.append(doc_idx)

    return questions, answers, ground_truth_indices


##########################################################
# 5) Evaluate Retrieval
##########################################################
def evaluate_retrieval(questions, ground_truth_indices, doc_embeddings, df, embedding_model, top_k=5):
    """
    Convert questions to embeddings, do top-k retrieval, measure MRR, Recall@k.
    Returns a dict with metrics + 'Retrieved Docs' for each Q.
    """
    print("Embedding questions for retrieval...")
    q_embeddings = embedding_model.encode(questions, show_progress_bar=True, convert_to_numpy=True, batch_size = 1)



    abstracts_list = df["abstract"].tolist()

    ranks = []
    retrieved_docs = []
    not_found_count = 0

    for i, q_emb in enumerate(q_embeddings):
        sims = cosine_similarity([q_emb], doc_embeddings)[0]
        ranked_idx = sims.argsort()[::-1][:top_k]

        # store the top doc texts
        top_docs = [abstracts_list[idx] for idx in ranked_idx]
        retrieved_docs.append(top_docs)

        # check rank for ground truth
        gt_idx = ground_truth_indices[i]
        if gt_idx in ranked_idx:
            rank_pos = list(ranked_idx).index(gt_idx) + 1
        else:
            rank_pos = float('inf')
            not_found_count += 1

        ranks.append(rank_pos)

    # MRR + Recall
    mrr = sum(1/r for r in ranks if r != float('inf')) / len(ranks)
    recall_at_k = sum(1 for r in ranks if r <= top_k) / len(ranks)

    print("Retrieval Stats:")
    print(f"Not found correct doc = {not_found_count}/{len(questions)}")
    print(f"MRR: {mrr:.4f}, Recall@{top_k}: {recall_at_k:.4f}")

    return {
        "MRR": mrr,
        "Recall@k": recall_at_k,
        "Ranks": ranks,
        "Retrieved Docs": retrieved_docs
    }

##########################################################
# 6) LLM Query (RAG)
##########################################################
def query_llm_with_rag(query, retrieved_docs):
    """
    Query LLM with retrieved documents as context.
    Returns a string response or 'NO RESPONSE' if something is missing.
    """
    # Build your prompt
    context = "\n\n".join(f"Context {i+1}: {doc}" for i, doc in enumerate(retrieved_docs))
    prompt = (
        f"Given the following context, answer the query.\n\n"
        f"Query: {query}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:"
    )

    # Call the SambaNova chat endpoint
    resp = client.chat.completions.create(
        model="Meta-Llama-3.1-70B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for answering user queries."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=150
    )

    # Defensive checks
    if resp is None:
        print("Warning: LLM response is None.")
        return "NO RESPONSE"

    if not resp.choices or len(resp.choices) == 0:
        print("Warning: LLM response has no choices.")
        return "NO RESPONSE"

    # Typically each choice has a `.message` object
    choice = resp.choices[0]
    if choice.message is None:
        print("Warning: LLM choice has no 'message' field.")
        return "NO RESPONSE"

    # The `content` can also be None
    content = choice.message.content
    if content is None:
        print("Warning: LLM choice.message has no 'content'.")
        return "NO RESPONSE"

    return content.strip()


##########################################################
# 7) Evaluate Hallucination
##########################################################
def evaluate_hallucination(questions, answers, df, ground_truth_indices, retrieved_docs, llm_responses):
    """
    Similar to your original code:
    - Check if the correct doc was retrieved
    - Compare final LLM response to ground-truth 'answers'
    - BLEU, ROUGE, etc.
    """
    abstracts_list = df["abstract"].tolist()

    ranks = []
    exact_matches = 0
    bleu_scores = []
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    for i, q in enumerate(questions):
        # 1. Did we retrieve the correct doc?
        correct_abstract = abstracts_list[ground_truth_indices[i]]
        rank = float('inf')
        if correct_abstract in retrieved_docs[i]:
            rank = retrieved_docs[i].index(correct_abstract) + 1
        ranks.append(rank)

        # 2. Compare LLM response to ground-truth answer
        gt_answer = answers[i].strip().lower()
        llm_answer = llm_responses[i].strip().lower()
        if gt_answer == llm_answer:
            exact_matches += 1

        # BLEU
        bleu = sentence_bleu([gt_answer.split()], llm_answer.split())
        bleu_scores.append(bleu)

        # ROUGE
        rouge = scorer.score(gt_answer, llm_answer)
        rouge_scores.append({
            "rouge1": rouge["rouge1"].fmeasure,
            "rougeL": rouge["rougeL"].fmeasure
        })

    mrr = sum(1/r for r in ranks if r != float('inf')) / len(ranks)
    recall_at_5 = sum(1 for r in ranks if r <= 5) / len(ranks)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge1 = sum(r["rouge1"] for r in rouge_scores) / len(rouge_scores)
    avg_rougeL = sum(r["rougeL"] for r in rouge_scores) / len(rouge_scores)
    exact_match_rate = exact_matches / len(questions)

    return {
        "MRR": mrr,
        "Recall@5": recall_at_5,
        "Exact Match Rate": exact_match_rate,
        "Average BLEU": avg_bleu,
        "Average ROUGE-1": avg_rouge1,
        "Average ROUGE-L": avg_rougeL
    }

##########################################################
# 8) Main Pipeline
##########################################################
def main():
    folder_path = "qa_pairs_output"

    # 1) Gather docs, skipping short abstracts
    df_docs = gather_docs(folder_path, min_words=40)
    if df_docs.empty:
        print("No valid docs found. Exiting.")
        return

    # 2) Load or create doc embeddings
    embedding_model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)
    doc_embeddings = embed_docs(df_docs, embedding_model, embeddings_file="doc_embeds.pkl")

    # 3) Gather Q&A pairs referencing these filtered docs
    questions, answers, ground_truth_indices = gather_qa_pairs(folder_path, df_docs)
    if not questions:
        print("No Q&A pairs found.")
        return

    # 4) Evaluate retrieval, etc.
    retrieval = evaluate_retrieval(
        questions=questions,
        ground_truth_indices=ground_truth_indices,
        doc_embeddings=doc_embeddings,
        df=df_docs,
        embedding_model=embedding_model,
        top_k=5
    )
    
    # 5) RAG
    llm_responses = []
    for i, q in tqdm(enumerate(questions)):
        top_docs = retrieval["Retrieved Docs"][i]
        resp = query_llm_with_rag(q, top_docs)
        llm_responses.append(resp)

    # 6) Evaluate hallucination
    metrics = evaluate_hallucination(
        questions=questions,
        answers=answers,
        df=df_docs,
        ground_truth_indices=ground_truth_indices,
        retrieved_docs=retrieval["Retrieved Docs"],
        llm_responses=llm_responses
    )
    print("Hallucination Metrics:", metrics)

if __name__ == "__main__":
    main()

