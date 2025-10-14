import os
import json
import asyncio
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

from src.main import run_pipeline

# ---------------- Hugging Face Hub LLM ----------------
def get_llm():
    # print("✅ Using Hugging Face Inference API (Phi-3-Mini)")
    # return LangchainLLMWrapper(HuggingFaceHub(
    #     repo_id="microsoft/Phi-3-mini-4k-instruct",
    #     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    #     model_kwargs={"temperature": 0, "max_new_tokens": 256}
    # ))


    print("✅ Using local Hugging Face model (Phi-3-Mini)")
    pipe = pipeline(
        "text-generation",
        model="microsoft/Phi-3-mini-4k-instruct",
        max_new_tokens=256,
        temperature=0.01,
        device_map="auto"
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return LangchainLLMWrapper(llm)

# ---------------- Embeddings ----------------
def get_embeddings():
    print("✅ Using Hugging Face Embeddings (MiniLM-L6-v2)")
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(model)

# ---------------- Load Ground Truth ----------------
with open("ground_truth.json", "r", encoding="utf-8") as f:
    gt = json.load(f)["ground_truth"][:3]  # test subset

records = []
for item in gt:
    query = item["query"]
    expected_answer = item["answer"]
    answer, contexts = run_pipeline(query)
    context_texts = [c.page_content if hasattr(c, "page_content") else str(c) for c in contexts]
    records.append({
        "question": query,
        "answer": answer,
        "contexts": context_texts,
        "ground_truth": expected_answer,
    })

dataset = Dataset.from_pandas(pd.DataFrame(records))
llm = get_llm()
embeddings = get_embeddings()

# ---------------- Async batching ----------------
async def async_evaluate_batches(dataset, batch_size=2):
    tasks = []
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        subset = dataset.select(range(start, end))
        tasks.append(asyncio.to_thread(
            evaluate,
            subset,
            [faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings
        ))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    results = [r for r in results if not isinstance(r, Exception)]
    return results

print("⚙️ Evaluating RAGAS metrics (async HF Inference API)...")

batch_results = asyncio.run(async_evaluate_batches(dataset, batch_size=2))

dfs = [res.to_pandas() for res in batch_results]
df = pd.concat(dfs, ignore_index=True)
df.to_csv("src/evaluation/results/ragas_hf_report_async.csv", index=False)

print("\n📊 Average scores:\n", df.mean(numeric_only=True))
print("✅ Saved to src/evaluation/results/ragas_hf_report_async.csv")
