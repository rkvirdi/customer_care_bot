import sys
import json
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Allow imports from src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import run_pipeline  # assumes your run_pipeline returns (answer, context)

def calc_bleu(reference, hypothesis):
    # Tokenize by whitespace
    ref_tokens = reference.strip().split()
    hyp_tokens = hypothesis.strip().split()
    # BLEU with smoothing for short sentences
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)

def calc_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rougeL'].fmeasure  # Returns F1 measure of ROUGE-L

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

    bleu_scores, rouge_scores = [], []

    for example in gt_data:
        query = example["query"]
        reference = example["answer"]

        # Generate answer from pipeline
        answer, _ = run_pipeline(file_paths, query)
        # If answer is a dict or contains 'answer', extract text
        if isinstance(answer, dict) and 'answer' in answer:
            answer = answer['answer']

        # print(f"Query: {query}")
        # print(f"Ground Truth: {reference}")
        # print(f"Generated Answer: {answer}")

        bleu = calc_bleu(reference, answer)
        rouge = calc_rouge(reference, answer)
        bleu_scores.append(bleu)
        rouge_scores.append(rouge)

        # print(f"BLEU: {bleu:.2f}, ROUGE-L: {rouge:.2f}")
        # print("-" * 60)

    print("\n==== AVERAGE METRICS ACROSS ALL QUERIES ====")
    print(f"Avg BLEU: {sum(bleu_scores)/len(bleu_scores):.2f}")
    print(f"Avg ROUGE-L: {sum(rouge_scores)/len(rouge_scores):.2f}")

if __name__ == "__main__":
    main()
