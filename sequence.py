from typing import Dict, List, Tuple
import json
import math
from collections import defaultdict

# Simple bigram transition model over endpoint paths.


def train_bigram_model(sequences: List[List[str]], k_smoothing: float = 1.0) -> Dict:
    counts = defaultdict(lambda: defaultdict(int))
    start_token = "<s>"
    for seq in sequences:
        prev = start_token
        for token in seq + ["</s>"]:
            counts[prev][token] += 1
            prev = token
    # Convert to probabilities with add-k smoothing per context
    model = {"start": start_token, "k": k_smoothing, "probs": {}}
    for prev, nexts in counts.items():
        total = sum(nexts.values()) + k_smoothing * (len(nexts) + 1)  # +1 for unseen bucket
        probs = {}
        for nxt, c in nexts.items():
            probs[nxt] = (c + k_smoothing) / total
        probs["<unseen>"] = k_smoothing / total
        model["probs"][prev] = probs
    return model


def save_bigram_model(model: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(model, f)


def load_bigram_model(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def sequence_negative_log_likelihood(model: Dict, sequence: List[str]) -> float:
    start = model.get("start", "<s>")
    probs = model.get("probs", {})
    prev = start
    nll = 0.0
    steps = 0
    for token in sequence + ["</s>"]:
        next_probs = probs.get(prev, {})
        p = next_probs.get(token, next_probs.get("<unseen>", 1e-6))
        nll += -math.log(max(p, 1e-9))
        prev = token
        steps += 1
    if steps == 0:
        return 0.0
    return nll / steps


def nll_to_score(nll: float, bias: float = 2.0, scale: float = 1.0) -> float:
    # Map NLL (>=0) to [0,1] where higher means more anomalous
    # score = 1 - sigmoid(-(nll - bias)/scale) = sigmoid((nll - bias)/scale)
    x = (nll - bias) / max(scale, 1e-6)
    return 1.0 / (1.0 + math.exp(-x))
