import json
import os
from src.main import run_pipeline
#from utils.retrieval_augment import retrieve_similar


# Load the dataset for evaluation
with open("ground_truth.json", "r") as f:
    ground_truth = json.load(f)

# Function to calculate evaluation metrics
def evaluate_retrieval(retriever, ground_truth, k=10):
    precision_list = []
    recall_list = []
    reciprocal_ranks = []

    for item in ground_truth:
        query = item["query"]
        ground_truth_doc = item["ground_truth_document"]

        # Retrieve top-k documents
        retrieved_docs = retriever.get_relevant_documents(query)[:k]

        # Extract content for comparison
        retrieved_content = [doc.page_content for doc in retrieved_docs]

        # Precision@k
        relevant_retrieved = sum(1 for doc in retrieved_content if ground_truth_doc in doc)
        precision = relevant_retrieved / k
        precision_list.append(precision)

        # Recall@k
        total_relevant = 1  # Assuming one ground truth document
        recall = relevant_retrieved / total_relevant
        recall_list.append(recall)

        # Mean Reciprocal Rank (MRR)
        try:
            rank = next(i + 1 for i, doc in enumerate(retrieved_content) if ground_truth_doc in doc)
            reciprocal_ranks.append(1 / rank)
        except StopIteration:
            reciprocal_ranks.append(0)

    # Calculate averages
    precision_avg = sum(precision_list) / len(precision_list)
    recall_avg = sum(recall_list) / len(recall_list)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return {
        "Precision@k": precision_avg,
        "Recall@k": recall_avg,
        "MRR": mrr
    }




# Evaluate the retriever
evaluation_metrics = evaluate_retrieval(run_pipeline, ground_truth, k=10)
print("Retrieval Evaluation Metrics:", evaluation_metrics)

