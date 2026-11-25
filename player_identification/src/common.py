"""Data layout, verified expert score axes and final fusion rules."""
import ast, json, math, os, sys
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
ARTIFACTS = HERE / 'artifacts'
TEST = HERE.parent / 'data/official/player_identification'
OA = ARTIFACTS / 'experts'
CACHE = HERE / 'outputs/layout'
OFFICIAL_EMBEDDINGS = ARTIFACTS / 'embeddings'
META = ARTIFACTS / 'pools/train_meta.parquet'
OUT = HERE / 'outputs/features'
CONFIG = json.loads((HERE / 'config/final.yaml').read_text())
KEYS = (['main_validation', 'main_holdout'] + [f'engine_{r}_{i}' for r in ('validation', 'holdout') for i in range(3)]
        + ['pub2', 'priv2'])
VAL = ('main_validation', 'engine_validation_0', 'engine_validation_1', 'engine_validation_2')
HOLD = ('main_holdout', 'engine_holdout_0', 'engine_holdout_1', 'engine_holdout_2')
FIXED_BALANCED = {"setmatch/e4": .55, "decisions": .20, "plda40": .35, "metadata": .10, "engine_only:strength": .20}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def zrow(s):
    s = np.asarray(s, dtype=np.float64)
    return (s - s.mean(1, keepdims=True)) / np.maximum(s.std(1, keepdims=True), 1e-9)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


@lru_cache(None)
def main_eps():
    return read(ARTIFACTS / 'pools/main.json')['episodes']


@lru_cache(None)
def engine_eps():
    return read(ARTIFACTS / 'pools/engine.json')['episodes']


def all_rows(eps):
    return np.array(sorted({r for ep in eps for part in ('gallery', 'questions') for x in ep[part] for r in x['row_ids']}), dtype=np.int64)


@lru_cache(None)
def dataset_rows(source):
    if source == 'main':
        return all_rows(main_eps().values())
    if source == 'engine':
        return all_rows([e for role in ('validation', 'holdout') for e in engine_eps()[role]])
    return np.arange(7100)


def source_of(key):
    return key.split('_')[0]


@lru_cache(None)
def colors_all():
    return pd.read_parquet(META, columns=['color']).color.to_numpy()


@lru_cache(None)
def episode(key):
    """gallery, queries, truth(or None), sizes, colours (indexed by local row)."""
    src = source_of(key)
    if src in ('pub2', 'priv2'):
        raw = read(CACHE / f'{src}_layout.json')
        return raw['gallery'], raw['queries'], None, np.array([q['num_games'] for q in raw['queries']]), np.array(raw['colors'])
    if src == 'main':
        raw = main_eps()['new_' + key.split('_')[1]]
    else:
        _, role, ind = key.split('_')
        raw = engine_eps()[role][int(ind)]
    rows = dataset_rows(src)
    pos = {int(r): i for i, r in enumerate(rows)}
    gallery = [{"player_id": g["player_id"], "row_ids": [pos[r] for r in g["row_ids"]]} for g in raw["gallery"]]
    queries = [{"player_id": q["player_id"], "num_games": q["num_games"], "question_id": q["question_id"],
                "row_ids": [pos[r] for r in q["row_ids"]]} for q in raw["questions"]]
    cand = {g["player_id"]: i for i, g in enumerate(gallery)}
    truth = np.array([cand[q["player_id"]] for q in queries])
    sizes = np.array([q["num_games"] for q in queries])
    return gallery, queries, truth, sizes, colors_all()[rows]


def axes(key):
    g, q, _, n, _ = episode(key)
    return dict(question_ids=np.array([x['question_id'] for x in q]), candidate_ids=np.array([x['player_id'] for x in g]), num_games=n)


def physical(b):
    return b.removeprefix("engine_only:").removeprefix("anchor_scaled:")


def active(b, key):
    return not (b.startswith("engine_only:") and key.startswith("main"))


def read_scores(branch, key):
    with np.load(OA / branch / f'{key}.npz', allow_pickle=False) as f:
        ax = axes(key)
        for name, x in ax.items():
            if not np.array_equal(f[name], x):
                raise ValueError((branch, key, name))
        v = f['scores_z']
    assert v.shape == (400, 600) and np.isfinite(v).all()
    return v


def matrix(branch, key):
    s = read_scores(physical(branch), key)
    if branch.startswith("anchor_scaled:"):
        ref = read_scores("baseline", key).copy()
        for name, w in FIXED_BALANCED.items():
            if active(name, key):
                ref += w * read_scores(physical(name), key)
        return s * ref.std(axis=1, keepdims=True)
    return s


def transform(scores, tau=.2, gamma=2.):
    s = zrow(scores)
    logits = s / tau
    kernel = np.exp(logits - logits.max(1, keepdims=True))
    v = np.ones(s.shape[1])
    for _ in range(60):
        u = 1 / (kernel @ v)
        v = (s.shape[0] / s.shape[1]) / (u @ kernel)
    return s + gamma * zrow(logits + np.log(v)[None, :])


def blend(key, weights):
    base = matrix("baseline", key).copy()
    sizes = episode(key)[3][:, None]
    for b, (lo, hi) in weights.items():
        if active(b, key):
            base += np.where(sizes <= 2, lo, hi) * matrix(b, key)
    return transform(base)


def positions(scores, truth):
    actual = scores[np.arange(len(truth)), truth][:, None]
    return 1 + (scores > actual).sum(1) + ((scores == actual) & (np.arange(scores.shape[1])[None, :] < truth[:, None])).sum(1)


def reward(ranks):
    ranks = np.asarray(ranks)
    return np.where(ranks <= 5, np.exp(1. - ranks), 0.)


def e4(source):
    root = ARTIFACTS / 'embeddings'
    if source in ('main', 'engine'):
        return np.load(root / f'emb_{source}_E4.npy', mmap_mode='r')
    root = OFFICIAL_EMBEDDINGS
    names = ('wide', 'seq64', 'seq64_s2', 'wide_s2')
    return np.concatenate([np.load(root / source / f'{name}.npy') for name in names], axis=1) / 2
