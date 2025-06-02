import sys
import json
from pathlib import Path
import re
# Allow imports from src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.data_loading import load_data
from utils.chunking import chunk_text
from utils.embedding import embedding_model
from utils.vector_store import embed_chunks
from utils.retrieval_augment import retrieve_similar

def is_relevant(doc, ground_truth, threshold=0.3):
    # Remove punctuation and split into words
    def tokenize(text):
        return set(re.findall(r'\w+', text.lower()))
    gt_tokens = tokenize(ground_truth)
    doc_tokens = tokenize(doc.page_content)
    # Avoid division by zero
    if not gt_tokens:
        return False
    overlap = gt_tokens & doc_tokens
    # Proportion of ground truth words present in the doc
    return (len(overlap) / len(gt_tokens)) >= threshold

def precision_at_k(retrieved, ground_truth, k):
    hits = sum([is_relevant(doc, ground_truth) for doc in retrieved[:k]])
    return hits / k

def recall_at_k(retrieved, ground_truth, k):
    # Returns 1 if any doc in top-k matches, else 0
    return 1.0 if any(is_relevant(doc, ground_truth) for doc in retrieved[:k]) else 0.0

def mean_reciprocal_rank(retrieved, ground_truth):
    for idx, doc in enumerate(retrieved, 1):
        if is_relevant(doc, ground_truth):
            return 1.0 / idx
    return 0.0

def main():
    # Load ground truth data
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    KB_DIR = PROJECT_ROOT / "knowledgebase"
    GT_FILE = PROJECT_ROOT / "ground_truth.json"
    file_paths = [
        str(KB_DIR / "model_S_owners_manual.pdf"),
        str(KB_DIR / "model_X_owners_manual.pdf"),
    ]
    with open(GT_FILE, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Build vector store only once
    docs = load_data(file_paths)
    chunks = chunk_text(docs, chunk_size=500, chunk_overlap=50)
    model = embedding_model()
    vectorstore = embed_chunks(chunks, model)

    k = 5
    precisions, recalls, mrrs = [], [], []

    for example in gt_data:
        query = example["query"]
        ground_truth = example["ground_truth_document"]

        # Always return a list of docs for metrics
        results = retrieve_similar(docs, vectorstore, query, k)
        # If you have a chain, always use .invoke(query)
        if hasattr(results, "invoke"):
            # For modern LangChain retriever chains
            result_dict = results.invoke(query)
            results = result_dict.get("context", [])

        print(f"Query: {query}")
        for i, doc in enumerate(results):
            print(f"Retrieved Doc {i+1}: {doc.page_content[:120]} ...")
        print("---")

        precisions.append(precision_at_k(results, ground_truth, k))
        recalls.append(recall_at_k(results, ground_truth, k))
        mrrs.append(mean_reciprocal_rank(results, ground_truth))

        print(f"Precision@{k}: {precisions[-1]:.2f}, Recall@{k}: {recalls[-1]:.2f}, MRR: {mrrs[-1]:.2f}")
        print("-" * 60)

    print("\n==== AVERAGE METRICS ACROSS ALL QUERIES ====")
    print(f"Avg Precision@{k}: {sum(precisions)/len(precisions):.2f}")
    print(f"Avg Recall@{k}: {sum(recalls)/len(recalls):.2f}")
    print(f"Avg MRR: {sum(mrrs)/len(mrrs):.2f}")

if __name__ == "__main__":
    main()
