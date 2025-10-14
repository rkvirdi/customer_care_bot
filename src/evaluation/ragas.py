import os
os.environ["RAGAS_LLM_N_SAMPLES"] = "1"  # prevent Groq 'n must be at most 1' errors
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from src.main import run_pipeline

# --------------- Initialize Groq LLM ----------------
def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("❌ Please set GROQ_API_KEY environment variable.")
    print("✅ Using Groq (llama-3.1-8b-instant)")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    return LangchainLLMWrapper(llm)

# --------------- Initialize Embeddings ----------------
def get_embedding_model():
    hf_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(hf_model)

# --------------- Load ground truth ----------------
with open("ground_truth.json", "r", encoding="utf-8") as f:
    gt = json.load(f)["ground_truth"][:3]  # use only first 3

records = []
for item in gt:
    query = item["query"]
    expected = item["answer"]
    answer, contexts = run_pipeline(query)
    ctx_texts = [c.page_content if hasattr(c, "page_content") else str(c) for c in contexts]
    records.append({
        "question": query,
        "answer": answer,
        "contexts": ctx_texts,
        "ground_truth": expected,
    })

dataset = Dataset.from_pandas(pd.DataFrame(records))
llm = get_llm()
embeddings= get_embedding_model()
# --------------- Evaluate ----------------
print("⚙️ Evaluating RAGAS metrics (Groq backend)...")

scores = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm,
    embeddings=embeddings,
)

df = scores.to_pandas()
df.to_csv("src/evaluation/results/ragas_report.csv", index=False)
print("\n📊 Average scores:\n", df.mean(numeric_only=True))
print("✅ Saved to src/evaluation/results/ragas_report.csv")
