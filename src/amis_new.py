from pathlib import Path
import yaml
import sys
import numpy as np


def _config_path():
    # config/models.yaml at repo root
    p = Path(__file__).resolve().parent.parent / 'config' / 'models.yaml'
    return p


def _load_config():
    path = _config_path()
    if not path.exists():
        raise FileNotFoundError(f'models config not found at {path}')
    with open(path, 'r') as fh:
        cfg = yaml.safe_load(fh)
    return cfg or {}


def err_model(name):
    raise ValueError(f'Model {name} not supported')


def list_models():
    cfg = _load_config()
    models = cfg.get('models', {})
    return {k: v.get('description', '') for k, v in models.items()}


def get_model(args, device=None):
    """
    Instantiate a model from parsed args. Expects args.model_name to be set.
    """
    name = getattr(args, 'model_name', None)
    if name is None:
        raise ValueError('args must have model_name attribute')
    return get_model_name(name, device=device)


def get_model_name(name, device=None):

    cfg = _load_config()
    models = cfg.get('models', {})
    if name not in models:
        # support legacy prefixes like 'esm2_t33_650M' by searching keys
        # try exact match first, else fallback to startswith
        for k in models:
            if k == name or k.startswith(name) or name.startswith(k):
                name = k
                break
    if name not in models:
        err_model(name)

    entry = models[name]
    model_path = entry.get('path')
    model_type = entry.get('type', '').lower() if entry.get('type') else ''

    # Defer heavy imports until needed
    if name.startswith('esm') or name.startswith('esm2'):
        from fb_model_new import FBModel
        # FBModel signature: FBModel(name, repr_layer=[-1])
        # We pass the configured path (can be a HF path or local checkpoint)
        if device is not None:
            # FBModel in repo may not accept device; pass only name
            model = FBModel(model_path)
        else:
            model = FBModel(model_path)
        return model
    else:
        # fallback: try to instantiate FBModel with provided path
        try:
            from fb_model_new import FBModel
            return FBModel(model_path)
        except Exception:
            err_model(name)


def encode(seq, model):
    return model.encode(seq)

def decode(embedding, model, exclude=set()):
    """
    Decode embeddings/logits into an amino-acid string using the model's
    vocabulary (mirrors behavior of `amis.decode`).

    embedding: input passed to model.decode (embedding or logits)
    model: model object (expects model.decode and model.alphabet_.all_toks)
    exclude: set or 'unnatural' to exclude uncommon tokens
    """
    if exclude == 'unnatural':
        exclude = set([
            'B', 'J', 'O', 'U', 'X', 'Z', '-', '.',
        ])

    logits = model.decode(embedding)

    # handle MSA-model outputs
    if hasattr(model, 'name_') and 'esm_msa1_t' in model.name_:
        logits = logits[0]
        # embedding = embedding[0]  # not used further here

    logits = np.asarray(logits)

    # If logits has batch dim, take first
    if logits.ndim == 3:
        logits = logits[0]

    # Sanity checks
    # logits: (L, V)
    # embedding may be an array with length L; try to validate shape when possible
    try:
        L = logits.shape[0]
        V = logits.shape[1]
    except Exception:
        raise ValueError(f'unexpected logits shape from model.decode: {logits.shape}')

    # Model alphabet
    if not hasattr(model, 'alphabet_'):
        raise AttributeError('model.alphabet_ is required for amis_new.decode')

    all_toks = model.alphabet_.all_toks
    assert V == len(all_toks), f'vocab mismatch: logits vocab {V} vs alphabet {len(all_toks)}'

    valid_idx = [idx for idx, tok in enumerate(all_toks)]
    logits = logits[:, valid_idx]

    argmax_idxs = np.argmax(logits, axis=1)

    argmax = ''.join([
        all_toks[valid_idx[tok_idx]]
        if ('<' not in all_toks[valid_idx[tok_idx]] and all_toks[valid_idx[tok_idx]] not in exclude)
        else '.'
        for tok_idx in argmax_idxs
    ])

    return argmax

def reconstruct(seq, model, encode_kwargs={}, decode_kwargs={}):
    return decode(
        encode(seq, model, **encode_kwargs),
        model, **decode_kwargs
    )


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='amis_new: load models from config/models.yaml')
    p.add_argument('--list', action='store_true', help='List available models')
    p.add_argument('--model-name', help='Model key to instantiate')
    args = p.parse_args()

    if args.list:
        for k, v in list_models().items():
            print(f"{k}: {v}")
        sys.exit(0)

    if args.model_name:
        print(f'Instantiating model {args.model_name}...')
        m = get_model_name(args.model_name)
        print('Model instantiated:', type(m), getattr(m, 'name_', getattr(m, '__class__', None)))
    else:
        print('No action; use --list or --model-name')

# python amis_new.py --model-name esm2-dgoa-150M