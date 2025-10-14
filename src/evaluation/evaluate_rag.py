# ============================================
# src/evaluation/evaluate_rag.py
# ============================================

import sys, os, json
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
    # Use "answer" field from ground truth as shown in the JSON structure
    gold_answer = row.get("answer", "")
    reference = row.get("reference", "")

    # ✅ Call your real pipeline with error handling
    try:
        model_answer, context_chunks = run_pipeline(FILE_PATHS, q)
        
        # Validate context chunks
        if not context_chunks:
            print(f"⚠️ Warning: No context chunks returned for query: '{q}'")
        
        # Validate model answer
        if not model_answer or model_answer.strip() == "":
            print(f"⚠️ Warning: Empty answer generated for query: '{q}'")
            
        # Log context sources for debugging
        sources = [chunk.metadata.get('source', 'unknown') for chunk in context_chunks]
        print(f"Context sources for '{q}': {sources}")
        
    except Exception as e:
        print(f"⚠️ Error during pipeline for '{q}': {e}")
        model_answer, context_chunks = "", []

    
    # Create retrieved document IDs and check relevance based on reference
    retrieved_ids = []
    relevant_docs = {}
    
    for i, chunk in enumerate(context_chunks):
        chunk_id = f"chunk_{i}"
        retrieved_ids.append(chunk_id)
        
        # Check if chunk contains reference content or matches source
        is_relevant = False
        if reference:
            if reference.lower() in chunk.page_content.lower():  # Direct reference match
                is_relevant = True
            elif any(ref_part in chunk.metadata.get('source', '').lower() 
                    for ref_part in reference.lower().split()):  # Source match
                is_relevant = True
        
        if is_relevant:
            relevant_docs[chunk_id] = 1

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

# Create results directory
results_dir = Path("src/evaluation/results")
results_dir.mkdir(parents=True, exist_ok=True)

# Save detailed evaluation results
df.to_csv(results_dir / "eval_report.csv", index=False)

# Save per-query detailed results with examples
detailed_results = []
for i, row in enumerate(results):
    q = row["Question"]
    detailed_results.append({
        "query": q,
        "gold_answer": data[i]["answer"],
        "model_answer": results[i].get("model_answer", ""),
        "reference": data[i]["reference"],
        "metrics": {
            k: v for k, v in row.items() 
            if k not in ["Question", "model_answer"]
        }
    })

with open(results_dir / "detailed_results.json", "w") as f:
    json.dump(detailed_results, f, indent=2)

# Print summary statistics
print("\nEvaluation Results:")
print("=" * 50)
print(df)
print("\nAverage Scores:")
print("-" * 50)
means = df.mean(numeric_only=True)
for metric, value in means.items():
    print(f"{metric:10s}: {value:.3f}")

# Create visualizations
plot_scores(df)

print(f"\n✅ Evaluation completed! Results saved to {results_dir}/")
print("Files generated:")
print("  - eval_report.csv: Summary metrics")
print("  - detailed_results.json: Per-query detailed analysis")
print("  - metrics_chart.png: Visualization of results")


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
