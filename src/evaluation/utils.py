import json, re, string


def load_ground_truth(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("ground_truth", data)
    out = []
    for r in rows:
        out.append({
            "query_id": r.get("query_id"),
            "query": r.get("query") or r.get("question"),
            "answer": r.get("answer") or "",   # <-- string
            "reference": r.get("reference"),
        })
    return out

def normalize_text(s):
    # accept list[str] or str
    if isinstance(s, list):
        s = " ".join(s)
    if s is None:
        s = ""
    # lowercase
    s = s.lower()
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s
