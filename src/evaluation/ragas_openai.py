import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from src.main import run_pipeline

# Load ground truth
with open("ground_truth.json", "r", encoding="utf-8") as f:
    gt = json.load(f)["ground_truth"]

# Collect results
records = []
for item in gt:
    query = item["query"]
    expected_answer = item["answer"]
    # Run your RAG pipeline
    answer, contexts = run_pipeline(query)
    context_texts = [c.page_content if hasattr(c, "page_content") else str(c) for c in contexts]

    records.append({
        "question": query,
        "answer": answer,
        "contexts": context_texts,
        "ground_truth": expected_answer,
    })

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(pd.DataFrame(records))

# Evaluate
scores = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

# Convert results to DataFrame
df = scores.to_pandas()
df.to_csv("src/evaluation/results/ragas_report.csv", index=False)
print(df)
print("Detailed report saved to src/evaluation/results/ragas_report.csv")