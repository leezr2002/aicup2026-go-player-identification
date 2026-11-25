"""Fit three shortlist rankers on the eight labeled local pools."""
from common import CONFIG, HERE, OUT
from prepare_stage import prepare
from reranking import dataset, fit, torch


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoders', action='store_true', help='Train the four full-data CUDA E4 backbones')
    parser.add_argument('--workers', type=int, default=4, help='SGF preparation workers for E4')
    args = parser.parse_args()
    if args.encoders:
        from encoder_workflow import fit_backbones
        fit_backbones(args.workers)
        return
    torch.set_num_threads(4)
    keys = CONFIG['train_pools']
    if any(not (OUT / f"experts_{CONFIG['base']}_{k}.npz").is_file() for k in keys):
        prepare(keys)
    design, labels = dataset(keys)
    dest = HERE / 'outputs/weights'
    dest.mkdir(parents=True, exist_ok=True)
    for seed in CONFIG['seeds']:
        state = fit(design, labels, l2=CONFIG['l2'], epochs=CONFIG['epochs'], seed=seed)
        torch.save(state, dest / f'rerank_{seed}.pt')
        print(f'Trained seed {seed}', flush=True)
