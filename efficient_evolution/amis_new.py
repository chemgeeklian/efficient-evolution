from pathlib import Path
import yaml
import sys
import numpy as np
import scipy.special
try:
    from efficient_evolution.fb_model_new import FBModel
except ImportError:
    from fb_model_new import FBModel

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


def get_model_name(name):

    cfg = _load_config()
    models = cfg.get('models', {})
    if name not in models:
        # support legacy prefixes like 'esm2_t33_650M' by searching keys
        # try exact match first, else fallback to startswith
        for k in models:
            if k == name or k.startswith(name) or name.startswith(k):
                name = k
                break
    entry = models[name]
    model_path = entry.get('path')
    # model_type = entry.get('type', '').lower() if entry.get('type') else ''

    return FBModel(model_path)


def decode(embedding, model):
    """Decode embedding to logits."""
    logits = model.decode(embedding)
    argmax = np.argmax(logits, axis=-1)
    
    return argmax


def reconstruct(seq, model):
    emb = model.encode(seq)
    return decode(emb, model)


def soft_reconstruct(seq, model, alpha=1.0, offset=1):
    exclude = {'B', 'J', 'O', 'U', 'X', 'Z', '-', '.'}

    logits = model.predict_sequence_prob(seq)
    probs = scipy.special.softmax(logits, axis=1)

    tokenizer = model.tokenizer
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}

    mutations = []
    # iterate over 1..len(seq) (skip CLS=0, ignore EOS=len(seq)+1)
    for i in range(1, len(seq) + 1):
        pos = i - offset
        wt_token = seq[pos]
        wt_j = vocab.get(wt_token, vocab.get(tokenizer.unk_token))
        wt_prob = probs[i, wt_j]

        for j in range(probs.shape[1]):
            mt = id_to_token[j]
            if mt in exclude or '<' in mt or mt == wt_token:
                continue
            mt_prob = probs[i, j]
            if mt_prob > alpha * wt_prob:
                mutations.append((pos, wt_token, mt))

    return mutations


def diff(seq_old, seq_new, start=0, end=None):
    return [
        (i, a, b)
        for i, (a, b) in enumerate(zip(seq_old, seq_new))
        if start <= i and (end is None or i < end) and b != '.' and a != b
    ]


def reconstruct_multi_models(wt_seq, 
                             model_set="esm2_finetuned", 
                             alpha=1.0, 
                             return_names=False):
    """Reconstruct sequence using multiple models and aggregate mutations.

    Parameters
    ----------
    wt_seq : str
        Wildtype sequence to compare against.
    model_set : None | str | list
        If None, use all models defined under `models` in the config.
        If a string and matches a key under `model_sets` in `config/models.yaml`,
        that named set (list of model keys) will be used. If a string that
        does not match `model_sets`, it will be interpreted as a single model
        name. If a list, it is interpreted as an explicit list of model keys.
    alpha : float | None
        If None, use hard reconstruction (decode) and compute differences.
        If a float, use `soft_reconstruct` with that alpha threshold.
    return_names : bool
        If True, also return a mapping of which model names produced each
        mutation.

    Returns
    -------
    mutations_counts : dict
        Mapping (pos, wt, mt) -> count of models that suggested the mutation.
    (optional) mutations_model_names : dict
        Mapping (pos, wt, mt) -> list of model names that suggested it.

    Examples
    --------
    # use a named model set from config
    reconstruct_multi_models('MKT', model_set='esm2_finetuned')

    # use an explicit list
    reconstruct_multi_models('MKT', model_set=['esm2_t12_35M', 'esm2-dgoa-150M'])
    """

    cfg = _load_config()
    model_names = []

    # Resolve model_set into a list of model names
    if model_set is None:
        model_names = list(cfg.get('models', {}).keys())
    elif isinstance(model_set, str):
        model_sets = cfg.get('model_sets', {})
        if model_set in model_sets:
            model_names = list(model_sets[model_set])
        else:
            model_names = [model_set]
    elif isinstance(model_set, (list, tuple)):
        model_names = list(model_set)
    else:
        raise ValueError('model_set must be None, a string key, or a list of model names')

    mutations_counts = {}
    mutations_model_names = {}

    for model_name in model_names:
        model = get_model_name(model_name)

        if alpha is None:
            # hard decode reconstruct
            wt_new = reconstruct(wt_seq, model) #, decode_kwargs={'exclude': 'unnatural'})
            muts = diff(wt_seq, wt_new)
        else:
            muts = soft_reconstruct(wt_seq, model, alpha=alpha)

        for mutation in muts:
            if mutation not in mutations_counts:
                mutations_counts[mutation] = 0
                mutations_model_names[mutation] = []
            mutations_counts[mutation] += 1
            mutations_model_names[mutation].append(model_name)

        # free model if it holds large resources
        del model

    if return_names:
        return mutations_counts, mutations_model_names

    return mutations_counts