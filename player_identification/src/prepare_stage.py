"""Rebuild the 05 shortlist, paired features and expert evidence for B."""
import json
from threadpoolctl import threadpool_limits
from common import ARTIFACTS, CACHE, CONFIG, KEYS, OUT, blend, episode, np
import gallery
import experts
import pairs


def check_official_files(directory, config):
    from hashlib import file_digest
    for name, expected in config['official_inputs'].items():
        with (directory / name).open('rb') as handle:
            if file_digest(handle, 'sha256').hexdigest() != expected:
                raise ValueError(f'Official input differs from the frozen branch inputs: {name}')


def layouts():
    from common import TEST
    check_official_files(TEST, CONFIG)
    CACHE.mkdir(parents=True, exist_ok=True)
    for key, suffix, split in [('pub', '', 'public'), ('priv', '', 'private'),
                               ('pub2', '2', 'public'), ('priv2', '2', 'private')]:
        _, layout = gallery.read_pool(key, suffix, split)
        (CACHE / f'{key}_layout.json').write_text(json.dumps(layout), encoding='utf-8')


def prepare(keys):
    OUT.mkdir(parents=True, exist_ok=True)
    layouts()
    with threadpool_limits(1):
        for key in keys:
            scores = blend(key, CONFIG['weights']).astype(np.float32)
            _, _, truth, sizes, _ = episode(key)
            if truth is None:
                truth = np.full(len(sizes), -1)
            np.savez(OUT / f"blend_{CONFIG['base']}_{key}.npz", S=scores, truth=truth, sizes=sizes)
            pairs.build(key)
            experts.build(key)
            print(f'Prepared {key}', flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encode-sgf', action='store_true', help='Rebuild official E4 embeddings using CUDA')
    parser.add_argument('--trained-encoders', action='store_true', help='Use locally retrained E4 weights')
    args = parser.parse_args()
    if args.trained_encoders and not args.encode_sgf:
        parser.error('--trained-encoders requires --encode-sgf')
    if args.encode_sgf:
        from encoder_workflow import encode_official
        encode_official(args.trained_encoders)
    else:
        prepare(KEYS)
