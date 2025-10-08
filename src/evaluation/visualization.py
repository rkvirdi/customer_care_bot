import matplotlib.pyplot as plt
import numpy as np
import os

# Draw bar chart comparing metrics per query
def plot_scores(df):
    metrics = ["Precision@3", "Recall@3", "EM", "F1"]
    x = np.arange(len(df))
    plt.figure(figsize=(10,6))
    for i, m in enumerate(metrics):
        plt.bar(x + i*0.15, df[m], width=0.15, label=m)
    plt.xticks(x, df["Question"], rotation=30, ha="right")
    plt.ylabel("Score (0–1)")
    plt.title("RAG Evaluation Metrics per Query")
    plt.legend()
    os.makedirs("evaluation/results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("evaluation/results/metrics_chart.png")
    plt.show()
