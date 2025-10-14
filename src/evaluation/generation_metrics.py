# src/evaluation/generation_metrics.py
from collections import Counter
from rouge_score import rouge_scorer
from src.evaluation.utils import normalize_text

def exact_match(pred, gold):
    return float(normalize_text(pred) == normalize_text(gold))

def token_f1(pred, gold):
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p or not g:
        return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    prec = overlap / len(p)
    rec  = overlap / len(g)
    return (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

def rouge_l(pred, gold):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(normalize_text(gold), normalize_text(pred))["rougeL"].fmeasure


# from rouge_score import rouge_scorer
# from src.evaluation.utils import normalize_text

# # Exact Match — checks if model output == ground truth exactly (after normalization)
# def exact_match(pred, gold):
#     return float(normalize_text(pred) == normalize_text(gold))

# # Token-level F1 — measures partial overlap between predicted and true answers
# def token_f1(pred, gold):
#     p, g = normalize_text(pred), normalize_text(gold)
#     common = {w: min(p.count(w), g.count(w)) for w in set(p)}
#     num = sum(common.values())
#     if not p or not g:
#         return 0.0
#     prec, rec = num / len(p), num / len(g)
#     return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

# # ROUGE-L — measures longest common subsequence (text similarity)
# def rouge_l(pred, gold):
#     scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
#     return scorer.score(gold, pred)["rougeL"].fmeasure
