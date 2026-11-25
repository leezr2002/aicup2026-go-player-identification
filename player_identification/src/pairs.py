from common import *
VARIANT = CONFIG['base']
K = CONFIG['shortlist']
HT = ARTIFACTS / 'human_trunk'

def d18_table(source):
    if source in ('pub2', 'priv2'):
        x = np.load(HT / f'{source}_d18_x.npy').astype(np.float32)
        rows = np.load(HT / f'{source}_d18_rows.npy')
        assert np.array_equal(rows, np.arange(7100))
        has = np.ones(7100, bool)
    else:
        tr = np.load(HT / 'train_d18_rows.npy')
        xx = np.load(HT / 'train_d18_x.npy', mmap_mode='r')
        rows = dataset_rows(source)
        pos = np.searchsorted(tr, rows)
        pos = np.clip(pos, 0, len(tr) - 1)
        has = tr[pos] == rows
        x = np.zeros((len(rows), 12, 96), np.float32)
        x[has] = np.asarray(xx[pos[has]], dtype=np.float32)
    x = x - x.mean(axis=(0, 1), keepdims=True) * 0  # keep raw; centred below per pool
    return x, has

def build(key):
    src = source_of(key)
    z = np.load(OUT / f'blend_{VARIANT}_{key}.npz')
    S, truth, sizes = z['S'].astype(np.float64), z['truth'], z['sizes']
    g, q, _, _, col = episode(key)
    E = np.asarray(e4(src), dtype=np.float32)
    arms = [unit(E[:, 256 * a:256 * (a + 1)]) for a in range(4)]
    D, has = d18_table(src)
    # centre HumanSL descriptors with the pool's gallery mean so that cosine measures deviation from the pool
    grow = np.array([r for x in g for r in x['row_ids']])
    gm = D[grow][has[grow]].reshape(-1, 96).mean(0)
    Du = unit(D - gm)
    cent = [unit(np.stack([a[x['row_ids']].mean(0) for x in g])) for a in arms]
    idx = np.argsort(-S, axis=1, kind='stable')[:, :K]
    feats = []
    for i, qq in enumerate(q):
        qr = np.array(qq['row_ids']); qc = col[qr]
        qcent = [unit(a[qr].mean(0)) for a in arms]
        rowf = []
        for c in idx[i]:
            gr = np.array(g[c]['row_ids']); gc = col[gr]
            f = []
            M = np.mean([a[qr] @ a[gr].T for a in arms], axis=0)          # (nq, 10) mean over arms
            f.append(np.mean([qcent[k] @ cent[k][c] for k in range(4)]))   # centroid cosine
            srt = np.sort(M, axis=1)[:, ::-1]
            f.append(srt[:, 0].mean())                                      # MaxSim
            f.append(srt[:, :2].mean())                                     # top-2
            f.append(srt[:, :3].mean())
            f.append(M.max())                                               # best single pair
            f.append(np.log(np.exp(M / .05).mean(axis=1)).mean() * .05)     # soft max
            same = qc[:, None] == gc[None, :]
            ms = np.where(same, M, -9).max(1); mc = np.where(~same, M, -9).max(1)
            f.append(np.where(ms > -9, ms, mc).mean())                      # same-colour MaxSim
            f.append(np.where(mc > -9, mc, ms).mean())                      # cross-colour MaxSim
            f.append(same.sum(1).mean() / 10.)                              # same-colour gallery share
            # HumanSL per-turn MaxSim (late interaction on 12 sampled turns per game)
            qh = qr[has[qr]]; gh = gr[has[gr]]
            if len(qh) and len(gh):
                T = Du[qh].reshape(-1, 96) @ Du[gh].reshape(-1, 96).T
                f += [T.max(1).mean(), np.sort(T, 1)[:, -3:].mean(), unit(Du[qh].reshape(-1, 96).mean(0)) @ unit(Du[gh].reshape(-1, 96).mean(0)), 1.]
            else:
                f += [0., 0., 0., 0.]
            rowf.append(f)
        feats.append(rowf)
    F = np.array(feats, np.float32)                                         # (400, K, nf)
    pos = np.full(400, -1)
    if truth[0] >= 0:
        for i in range(400):
            w = np.where(idx[i] == truth[i])[0]
            pos[i] = w[0] if len(w) else -1
    np.savez(OUT / f'feat_{VARIANT}_{key}.npz', F=F, idx=idx, S=S[np.arange(400)[:, None], idx].astype(np.float32),
             pos=pos, sizes=sizes, truth=truth)
    return F.shape
