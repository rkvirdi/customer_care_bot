# ============================================
# src/evaluation/evaluate_rag.py
# ============================================

import sys, os
import pandas as pd
from tqdm import tqdm

# ✅ Fix import path so Python can locate src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ✅ Import your real RAG pipeline
from src.main import run_pipeline

# ✅ Import your evaluation tools
from src.evaluation.retrieval_metrics import *
from src.evaluation.generation_metrics import *
from src.evaluation.visualization import plot_scores
from src.evaluation.utils import load_ground_truth

from pathlib import Path

# ------------------------------------------------------------
# 1️⃣ Prepare dataset and knowledgebase file paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
KB_DIR   = BASE_DIR / "knowledgebase"


# All  Tesla manuals go here
FILE_PATHS = [
    str(KB_DIR / "model_S_owners_manual.pdf"),
    str(KB_DIR / "model_X_owners_manual.pdf"),
]

# ------------------------------------------------------------
# 2️⃣ Load evaluation dataset
# ------------------------------------------------------------
data = load_ground_truth("ground_truth.json")

results = []

# ------------------------------------------------------------
# 3️⃣ Evaluate each test question
# ------------------------------------------------------------
for row in tqdm(data, desc="Evaluating RAG pipeline"):
    q = row["query"]
    gold_answer = row["answer"]

    # ✅ Call your real pipeline
    try:
        model_answer, context_chunks = run_pipeline(FILE_PATHS, q)
    except Exception as e:
        print(f"⚠️ Error during pipeline for '{q}': {e}")
        model_answer, context_chunks = "", []

    
    retrieved_ids = [f"chunk_{i}" for i in range(len(context_chunks))]
    relevant_docs = {retrieved_ids[0]: 1} if retrieved_ids else {}

    # --------------------------------------------------------
    # 4️⃣ Compute metrics
    # --------------------------------------------------------
    results.append({
        "Question": q,
        "Precision@3": precision_at_k(retrieved_ids, relevant_docs, 3),
        "Recall@3": recall_at_k(retrieved_ids, relevant_docs, 3),
        "MRR@3": mrr_at_k(retrieved_ids, relevant_docs, 3),
        "EM": exact_match(model_answer, gold_answer),
        "F1": token_f1(model_answer, gold_answer),
        "ROUGE-L": rouge_l(model_answer, gold_answer),
    })

# ------------------------------------------------------------
# 5️⃣ Save results + plot
# ------------------------------------------------------------
df = pd.DataFrame(results)
os.makedirs("src/evaluation/results", exist_ok=True)
df.to_csv("src/evaluation/results/eval_report.csv", index=False)
print(df)
print("\nAVERAGE SCORES:\n", df.mean(numeric_only=True))

plot_scores(df)
print("\n✅ Evaluation completed! Results saved to src/evaluation/results/\n")


# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# import os, json
# from tqdm import tqdm
# from src.evaluation.utils import load_ground_truth
# from src.evaluation.retrieval_metrics import *
# from src.evaluation.generation_metrics import *
# from src.evaluation.visualization import plot_scores

# # ---- Load dataset ----
# data = load_ground_truth("ground_truth.json")

# # Dummy retriever/generator (replace with your real RAG modules)
# def retrieve(query): return ["d1", "d2", "d3"]
# def generate(query): return "mock answer"

# results = []
# for row in tqdm(data):
#     q, gold, doc = row["query"], row["answer"], row["ground_truth_document"]
#     retrieved = retrieve(q)                     # Get top documents (IDs)
#     generated = generate(q)                     # Simulated answer
#     rel = {retrieved[0]: 1}                     # Mark first doc as 'relevant'

#     # Compute all metrics
#     metrics = {
#         "Question": q,
#         "Precision@3": precision_at_k(retrieved, rel, 3),
#         "Recall@3": recall_at_k(retrieved, rel, 3),
#         "MRR@3": mrr_at_k(retrieved, rel, 3),
#         "nDCG@3": ndcg_at_k(retrieved, rel, 3),
#         "EM": exact_match(generated, gold),
#         "F1": token_f1(generated, gold),
#         "ROUGE-L": rouge_l(generated, gold),
#     }
#     results.append(metrics)

# import pandas as pd
# df = pd.DataFrame(results)
# print(df)
# # Create the folder automatically if it doesn't exist
# os.makedirs("evaluation/results", exist_ok=True)

# # Now safe to save
# df.to_csv("evaluation/results/eval_report.csv", index=False)

# # Optional: visualize
# os.makedirs("evaluation/results", exist_ok=True)
# plot_scores(df)
