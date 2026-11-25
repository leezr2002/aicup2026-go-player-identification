import numpy as np
import torch
from common import CONFIG, OUT

def design(F, S, N):
    """Per-candidate design: blend score (list-standardised) + each pair feature raw, list z-score and gap to best other."""
    B, K, nf = F.shape
    X = [((S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6))[..., None], (S - S[:, :1])[..., None]]
    mu = F.mean(1, keepdims=True); sd = F.std(1, keepdims=True) + 1e-6
    X += [F, (F - mu) / sd]
    other_max = np.stack([np.delete(F, j, axis=1).max(1) for j in range(K)], 1)
    X.append(F - other_max)
    X.append(np.broadcast_to(np.linspace(0, 1, K)[None, :, None], (B, K, 1)))  # blend rank position
    X = np.concatenate(X, -1).astype(np.float32)
    short = (N <= 2).astype(np.float32)[:, None, None]
    return np.concatenate([X * short, X * (1 - short)], -1)

class Scorer(torch.nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.lin = torch.nn.Linear(d, 1)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Dropout(.2), torch.nn.Linear(hidden, 1)) if hidden else None

    def forward(self, x):
        out = self.lin(x).squeeze(-1)
        if self.mlp is not None:
            out = out + self.mlp(x).squeeze(-1)
        return out

def fit(X, P, hidden=0, l2=1e-2, epochs=300, seed=0):
    torch.manual_seed(seed)
    m = P >= 0
    Xt = torch.tensor(X[m]); y = torch.tensor(P[m])
    mean = Xt.reshape(-1, Xt.shape[-1]).mean(0); std = Xt.reshape(-1, Xt.shape[-1]).std(0) + 1e-6
    net = Scorer(X.shape[-1], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3 if hidden else 1e-2, weight_decay=0)
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        logits = net((Xt - mean) / std)
        loss = torch.nn.functional.cross_entropy(logits, y) + l2 * sum((p ** 2).sum() for n, p in net.named_parameters() if 'weight' in n)
        loss.backward(); opt.step()
    net.eval()
    return {'model': net.state_dict(), 'mean': mean, 'std': std}

def dataset(keys):
    pairs, scores, sizes, positions = [], [], [], []
    for key in keys:
        with np.load(OUT / f"feat_{CONFIG['base']}_{key}.npz") as z:
            pairs.append(z['F'])
            scores.append(z['S'])
            sizes.append(z['sizes'])
            positions.append(z['pos'])
    expert = np.concatenate([np.load(OUT / f"experts_{CONFIG['base']}_{key}.npz")['X'] for key in keys])
    expert = (expert - expert.mean(1, keepdims=True)) / (expert.std(1, keepdims=True) + 1e-6)
    features = np.concatenate([np.concatenate(pairs), expert], axis=-1)
    return design(features, np.concatenate(scores), np.concatenate(sizes)), np.concatenate(positions)
