#!/usr/bin/env python3
"""
Simple test script to exercise `FBModel.encode`.

Usage:
    python src/test_fb_model_encode.py --model <model_name> --length 300

Default model is an ESM-1 family model. Downloading a pretrained ESM model may
be large and take some time.
"""
import argparse
import random
import sys

import numpy as np

from efficient_evolution.fb_model_new import FBModel

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def random_protein(length):
    return ''.join(random.choices(AMINO_ACIDS, k=length))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='facebook/esm2_t30_150M_UR50D',
                   help='Pretrained model name to load via esm.pretrained.load_model_and_alphabet')
    p.add_argument('--length', type=int, default=30, help='Random sequence length')
    p.add_argument('--use-cuda', action='store_true', help='Move model/tokens to CUDA if available')
    args = p.parse_args()

    seq = [random_protein(args.length)]
    model = FBModel(args.model)

    inputs = model.tokenizer(seq, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.model(**inputs, output_attentions=True, return_dict=True)
    print(outputs.logits.shape)
    
    hidden_states = outputs.hidden_states
    print(outputs.keys())
    print(hidden_states)


def _main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='esm1_t6_43M_UR50S',
                   help='Pretrained model name to load via esm.pretrained.load_model_and_alphabet')
    p.add_argument('--length', type=int, default=300, help='Random sequence length')
    p.add_argument('--use-cuda', action='store_true', help='Move model/tokens to CUDA if available')
    args = p.parse_args()

    seq = random_protein(args.length)
    print(f"Random sequence ({len(seq)}): {seq[:60]}{'...' if len(seq)>60 else ''}")

    print(f"Loading model {args.model} ... this may download weights if not cached")
    try:
        model = FBModel(args.model)
    except Exception as e:
        print("Failed to load model:", e, file=sys.stderr)
        raise

    # Optionally move model to cuda if requested
    if args.use_cuda and hasattr(model.model_, 'cuda'):
        if not torch.cuda.is_available():
            print('CUDA requested but not available; continuing on CPU')
        else:
            model.model_ = model.model_.cuda()

    print('Encoding sequence...')
    try:
        emb = model.encode(seq)
    except Exception as e:
        print('Error during encode():', e, file=sys.stderr)
        raise

    emb = np.asarray(emb)
    print('Embedding shape:', emb.shape)
    # print a small sample
    if emb.size:
        print('First residue embedding (first 10 dims):', emb[0][:10])

    # Basic sanity checks
    assert emb.shape[0] == len(seq), 'Embedding length should match sequence length'
    assert emb.ndim == 2, 'Embedding should be 2D (L x D)'

    print('OK — encode produced an L x D embedding for the sequence')

    # ------------------ decode test ------------------
    print('\nDecoding embedding back to logits using FBModel.decode()')
    try:
        logits = model.decode(emb)
    except Exception as e:
        print('Error during decode():', e, file=sys.stderr)
        raise

    logits = np.asarray(logits)
    print('Logits shape:', logits.shape)

    # vocab size sanity check using alphabet tok_to_idx mapping
    try:
        vocab_size = len(model.alphabet_.tok_to_idx)
    except Exception:
        # fallback: try unk index + 1
        vocab_size = getattr(model, 'vocab_size', None) or logits.shape[1]

    assert logits.ndim == 2, 'Decode must return 2D logits (L x V)'
    assert logits.shape[0] == emb.shape[0], 'Logits sequence length must match embedding length'
    assert logits.shape[1] == vocab_size, 'Logits vocabulary dimension mismatch'

    # Map argmax ids back to token strings using reverse mapping
    pred_ids = logits.argmax(axis=1).tolist()
    try:
        idx_to_tok = {v: k for k, v in model.alphabet_.tok_to_idx.items()}
        pred_tokens = [idx_to_tok.get(i, '?') for i in pred_ids]
    except Exception:
        # if alphabet not available, just show ids
        pred_tokens = [str(i) for i in pred_ids]

    # Build a simple amino-acid string from single-letter tokens
    decoded_aa = ''.join([t for t in pred_tokens if len(t) == 1 and t.isalpha()])
    print('Decoded token preview (first 60 chars):', decoded_aa[:60])


if __name__ == '__main__':
    import torch

    main()
