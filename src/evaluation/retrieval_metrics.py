#Evaluates the retriever part of your RAG system.
#Each function measures how effectively the retriever finds the right context for the LLM.

import math

# Precision@k: How many of the top-k retrieved docs are actually relevant
def precision_at_k(retrieved, relevant, k=3):
    return sum(relevant.get(d, 0) for d in retrieved[:k]) / max(1, len(retrieved[:k]))

# Recall@k: How many of all relevant docs were retrieved in top-k
def recall_at_k(retrieved, relevant, k=3):
    total_rel = sum(relevant.values())
    return sum(relevant.get(d, 0) for d in retrieved[:k]) / max(1, total_rel)

# Mean Reciprocal Rank: ranks matter — earlier correct docs are better
def mrr_at_k(retrieved, relevant, k=10):
    for i, d in enumerate(retrieved[:k], 1):
        if relevant.get(d, 0) > 0:
            return 1.0 / i
    return 0.0

