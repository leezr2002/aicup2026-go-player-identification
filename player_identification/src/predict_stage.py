"""Score four official pools and emit their predictions in the template order."""
import argparse
import csv
from threadpoolctl import threadpool_limits
import gallery
import common
from common import ARTIFACTS, CACHE, CONFIG, HERE, OUT, TEST, axes, np, read
from prepare_stage import prepare, layouts
from reranking import Scorer, dataset, torch


def old_pool(key):
    layout = read(CACHE / f'{key}_layout.json')
    root = common.OFFICIAL_EMBEDDINGS / key
    arms = [np.load(root / f'{name}.npy') for name in ('wide', 'seq64', 'seq64_s2', 'wide_s2')]
    embedding = np.concatenate(arms, axis=1) / np.sqrt(len(arms))
    scores = gallery.components(embedding, layout['colors'], layout['gallery'], layout['queries'])
    fused = gallery.zrow(gallery.zrow(gallery.raw_score(scores, CONFIG['old_recipe'])))
    order = np.argsort(-gallery.sinkhorn(fused, CONFIG['old_tau']), axis=1, kind='stable')[:, :5]
    candidates = np.array([g['player_id'] for g in layout['gallery']])
    return dict(zip([q['question_id'] for q in layout['queries']], candidates[order]))


def new_pool(key, folder):
    x, _ = dataset([key])
    logits = []
    for seed in CONFIG['seeds']:
        state = torch.load(folder / f'rerank_{seed}.pt', map_location='cpu', weights_only=True)
        model = Scorer(x.shape[-1], 0)
        model.load_state_dict(state['model'])
        with torch.inference_mode():
            logits.append(model((torch.from_numpy(x) - state['mean']) / state['std']).numpy())
    order = np.argsort(-np.mean(logits, axis=0), axis=1, kind='stable')[:, :5]
    with np.load(OUT / f"feat_{CONFIG['base']}_{key}.npz") as z:
        selected = np.take_along_axis(z['idx'], order, axis=1)
    ax = axes(key)
    return dict(zip(ax['question_ids'], ax['candidate_ids'][selected]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained', action='store_true', help='Use weights produced by train.py')
    parser.add_argument('--rebuilt-embeddings', action='store_true', help='Read E4 outputs rebuilt from SGF')
    args = parser.parse_args()
    if args.rebuilt_embeddings:
        common.OFFICIAL_EMBEDDINGS = HERE / 'outputs/embeddings'
    folder = HERE / ('outputs/weights' if args.trained else 'artifacts/weights')
    torch.set_num_threads(4)
    prepare(('pub2', 'priv2'))
    with threadpool_limits(4):
        answers = {**old_pool('pub'), **old_pool('priv'),
                   **new_pool('pub2', folder), **new_pool('priv2', folder)}
    with (TEST / 'sample_submission_player_v2.csv').open(encoding='utf-8-sig', newline='') as sample:
        ids = [row['question_id'] for row in csv.DictReader(sample)]
    if len(set(ids)) != 1600 or set(answers) != set(ids):
        raise ValueError('Question IDs differ from the official template')
    dest = HERE / 'outputs/submission_player.csv'
    temp = dest.with_suffix('.tmp')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['question_id', *(f'top_{i}' for i in range(1, 6))])
        writer.writerows([qid, *answers[qid]] for qid in ids)
    temp.replace(dest)
    print(f'Wrote {len(ids)} predictions to outputs/{dest.name}')
