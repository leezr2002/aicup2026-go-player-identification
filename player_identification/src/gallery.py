import csv, math
import numpy as np
from common import TEST as TEST_DIR

def read_pool(prefix, suffix, split):
    """Official layout: candidate CSV rows first (gallery = first-appearance order), then queries."""
    csv.field_size_limit(1 << 26)
    items, gallery, queries = [], {}, []
    with (TEST_DIR / f"player_identification_candidates{suffix}_{split}.csv").open(encoding="utf-8-sig", newline="") as h:
        for row in csv.DictReader(h):
            gallery.setdefault(row["player_id"], []).append(len(items))
            items.append((row["sgf_content"], row["color"]))
    with (TEST_DIR / f"player_identification_test{suffix}_{split}.csv").open(encoding="utf-8-sig", newline="") as h:
        for row in csv.DictReader(h):
            rows = []
            for slot in range(1, int(row["num_games"]) + 1):
                rows.append(len(items))
                items.append((row[f"sgf_{slot}"], row[f"color_{slot}"]))
            queries.append({"question_id": row["question_id"], "num_games": int(row["num_games"]), "row_ids": rows})
    layout = {"gallery": [{"player_id": p, "row_ids": r} for p, r in gallery.items()],
              "queries": queries, "colors": [c for _, c in items]}
    return items, layout

def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

def zrow(s):
    s = np.asarray(s, dtype=np.float64)
    return (s - s.mean(1, keepdims=True)) / np.maximum(s.std(1, keepdims=True), 1e-9)

def components(E, colours, gallery, queries):
    """proto: query-mean vs candidate-mean cosine; pair: per query game, mean of its 2 best
    candidate games; colour: per query game, cosine to the candidate's same-colour prototype."""
    colours = np.asarray(colours)
    C, Q = len(gallery), len(queries)
    gp = unit(np.stack([E[g["row_ids"]].mean(0) for g in gallery]))
    qp = unit(np.stack([E[q["row_ids"]].mean(0) for q in queries]))
    proto = qp @ gp.T
    width = max(len(g["row_ids"]) for g in gallery)
    pad = np.zeros((C, width, E.shape[1]), dtype=np.float32)
    mask = np.zeros((C, width), dtype=bool)
    cp = {c: np.zeros((C, E.shape[1]), dtype=np.float32) for c in "BW"}
    for i, g in enumerate(gallery):
        r = g["row_ids"]
        pad[i, :len(r)], mask[i, :len(r)] = E[r], True
        for c in "BW":
            sel = [x for x in r if colours[x] == c]
            cp[c][i] = unit(E[sel].mean(0)) if sel else gp[i]
    flat = pad.reshape(-1, E.shape[1])
    pair, colour = np.zeros((Q, C)), np.zeros((Q, C))
    for qi, q in enumerate(queries):
        r = q["row_ids"]
        sims = (E[r] @ flat.T).reshape(len(r), C, width)
        sims = np.where(mask[None], sims, -2.0)
        top = np.sort(sims, axis=2)[:, :, -2:]
        pair[qi] = top.mean(2).mean(0)
        colour[qi] = np.mean([cp[colours[x]] @ E[x] for x in r], axis=0)
    return {"proto": proto, "pair": pair, "colour": colour}

def raw_score(comp, recipe):
    a, b = recipe
    return zrow(comp["proto"]) + a * zrow(comp["pair"]) + b * zrow(comp["colour"])

def sinkhorn(S, tau, iters=60):
    L = S / tau
    target = math.log(S.shape[0] / S.shape[1])
    for _ in range(iters):
        L = L - np.logaddexp.reduce(L, axis=1, keepdims=True)
        L = L - np.logaddexp.reduce(L, axis=0, keepdims=True) + target
    return L
