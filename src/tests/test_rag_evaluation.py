# tests/test_rag_evaluation.py

import os
import json
import math
import pytest
from sentence_transformers import CrossEncoder

# — Replace these imports with the real paths in your repo —
from src.utils.data_loading       import load_data
from src.utils.chunking           import chunk_text
from src.utils.pinecone_store     import load_vectorstore
from src.utils.retrieval          import hybrid_retrieve
# — end replace —

# ─── Helper functions ──────────────────────────────────────────────────────────

def rerank(query: str, docs: list, top_k: int):
    """Rescore with a cross‐encoder and return top_k."""
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2")
    inputs   = [(query, d.page_content) for d in docs]
    scores   = reranker.predict(inputs)
    ranked   = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return ranked[:top_k]

def mean_reciprocal_rank(retrieved_ids, relevant_ids, k):
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def dcg_at_k(retrieved_ids, relevant_ids, k):
    return sum(
        (1 if doc_id in relevant_ids else 0) / math.log2(idx+2)
        for idx, doc_id in enumerate(retrieved_ids[:k])
    )

def idcg_at_k(relevant_ids, k):
    ideal = min(len(relevant_ids), k)
    # positions 1…ideal
    return sum(1 / math.log2(pos+1) for pos in range(1, ideal+1))

def ndcg_at_k(retrieved_ids, relevant_ids, k):
    idcg = idcg_at_k(relevant_ids, k)
    return (dcg_at_k(retrieved_ids, relevant_ids, k) / idcg) if idcg > 0 else 0.0

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def chunks():
    pdfs = [
        "/mnt/data/model_X_owners_manual.pdf",
        "/mnt/data/model_S_owners_manual.pdf",
    ]
    docs = load_data(pdfs)
    return chunk_text(docs, chunk_size=500, chunk_overlap=20)

@pytest.fixture(scope="session")
def vectorstore(chunks):
    # Should either create or load your Pinecone index
    return load_vectorstore(chunks)

@pytest.fixture(scope="session")
def ground_truth():
    path = os.path.join(os.path.dirname(__file__), "ground_truth.json")
    with open(path) as f:
        return json.load(f)

# ─── Tests ────────────────────────────────────────────────────────────────────

def test_rag_retrieval_and_rerank_metrics(chunks, vectorstore, ground_truth):
    k = 5
    precisions, recalls, mrrs, ndcgs = [], [], [], []

    for record in ground_truth:
        query   = record["query"]
        relevant = record["ground_truth_doc_ids"]

        # 1) initial hybrid retrieve (SS + BM25 + graph)
        candidates = hybrid_retrieve(query, vectorstore, k=10)

        # 2) rerank down to k
        final_docs = rerank(query, candidates, top_k=k)
        retrieved_ids = [d.id for d in final_docs]

        # 3) compute metrics
        tp = len(set(retrieved_ids) & set(relevant))
        precisions.append(tp / k)
        recalls.append(tp / len(relevant))
        mrrs.append(mean_reciprocal_rank(retrieved_ids, relevant, k))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant, k))

    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls)    / len(recalls)
    avg_mrr = sum(mrrs)     / len(mrrs)
    avg_ndcg = sum(ndcgs)   / len(ndcgs)

    print(f"\nAvg Precision@{k}: {avg_p:.3f}")
    print(f"Avg Recall@{k}:    {avg_r:.3f}")
    print(f"Avg MRR@{k}:       {avg_mrr:.3f}")
    print(f"Avg nDCG@{k}:      {avg_ndcg:.3f}\n")

    # enforce whatever thresholds make sense for you
    assert avg_p   >= 0.5, "Precision below target"
    assert avg_r   >= 0.5, "Recall below target"
    assert avg_mrr >= 0.5, "MRR below target"
    assert avg_ndcg>= 0.5, "nDCG below target"
