from common import *
VARIANT = CONFIG['base']
K = CONFIG['shortlist']
BR = ['baseline', *CONFIG['weights']]

def build(key):
    z = np.load(OUT / f'feat_{VARIANT}_{key}.npz')
    idx = z['idx']
    rows = np.arange(400)[:, None]
    X = []
    for b in BR:
        if active(b, key):
            m = matrix(b, key)
            v = m[rows, idx]
        else:
            v = np.zeros((400, K))
        X.append(v)
    X = np.stack(X, -1).astype(np.float32)                    # (400, K, nb)
    act = np.array([active(b, key) for b in BR], np.float32)
    np.savez(OUT / f'experts_{VARIANT}_{key}.npz', X=X, active=act)
    return X.shape
