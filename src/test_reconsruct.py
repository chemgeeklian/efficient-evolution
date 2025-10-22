#!/usr/bin/env python3
"""
Test script for soft_reconstruct() using FBModel (ESM2 backbone).

This script:
  1. Loads facebook/esm2_t30_150M_UR50D via FBModel
  2. Runs model.predict_sequence_prob() through soft_reconstruct()
  3. Prints logits stats and some per-position example probs

Expected usage:
    python test_soft_reconstruct.py
"""

import numpy as np
import torch
import scipy.special
from fb_model import FBModel   # <-- your reference FBModel class
from pathlib import Path


def soft_reconstruct(seq, model, alpha=1.0, offset=1):
    """
    Evaluate sequence reconstruction probabilities for each position
    using a language model's per-token distribution.
    Returns full logits (L, V).
    """
    exclude = set(['B', 'J', 'O', 'U', 'X', 'Z', '-', '.'])
    logits = model.predict_sequence_prob(seq)
    probs = scipy.special.softmax(logits, axis=1)

    print(f"[INFO] Soft reconstruct: seq_len={probs.shape[0]}, vocab={probs.shape[1]}")
    print(f"[INFO] Example probs (first residue): top5 =",
          np.sort(probs[1])[-5:][::-1])  # skip BOS token

    # Optionally: identify where the model assigns high alt-AA probabilities
    mutations = []
    for i in range(1, probs.shape[0]):  # skip first token (BOS)
        pos = i - offset
        wt = seq[pos] if pos < len(seq) else None
        if wt is None:
            continue
        wt_idx = model.alphabet_.tok_to_idx.get(wt, model.unk_idx_)
        wt_prob = probs[i, wt_idx]
        alt_toks = []
        for j, p in enumerate(probs[i]):
            tok = model.alphabet_.all_toks[j]
            if tok in exclude or '<' in tok or j == wt_idx:
                continue
            if p > alpha * wt_prob:
                alt_toks.append((tok, p))
        if alt_toks:
            mutations.append((pos, wt, alt_toks[:5]))  # keep top-5 for printout
    print(f"[INFO] Found {len(mutations)} mutable positions (alpha={alpha})")
    if len(mutations) > 0:
        print("  Example:", mutations[:3])
    return logits


def main():
    model_name = "facebook/esm1_t6_43M_UR50S"
    seq = "AAAAAAA"

    print(f"[INFO] Loading model {model_name} ...")
    model = FBModel(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Running soft_reconstruct on sequence:\n{seq}\n")
    logits = soft_reconstruct(seq, model, alpha=1.2)

    print("[DONE] logits shape:", logits.shape)
    print("[DONE] mean =", np.mean(logits), "std =", np.std(logits))


if __name__ == "__main__":
    main()
