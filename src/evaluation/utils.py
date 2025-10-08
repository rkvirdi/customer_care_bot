import json, re

# Load test dataset (ground_truth.json)
def load_ground_truth(path="ground_truth.json"):
    with open(path, "r") as f:
        return json.load(f)

# Normalize text by:
# - Lowercasing
# - Removing punctuation
# - Splitting into words
def normalize_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\\s]", " ", s)
    return s.split()
