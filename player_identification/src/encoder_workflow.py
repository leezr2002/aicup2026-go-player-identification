"""Optional CUDA rebuild of the four game encoders and the official E4 game embeddings."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from common import ARTIFACTS, CONFIG, HERE
import encoders as enc

MODELS = {'seq64': (128, 20261113), 'wide': (192, 20261115),
          'seq64_s2': (128, 20261116), 'wide_s2': (192, 20261117)}
POOLS = [('pub', '', 'public'), ('priv', '', 'private'),
         ('pub2', '2', 'public'), ('priv2', '2', 'private')]


def check_device():
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Rebuilding E4 requires CUDA with bfloat16 support')


def fit_backbones(workers):
    check_device()
    cache = enc.WORK / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / 'train_seq64.npy'
    metadata = pd.read_parquet(ARTIFACTS / 'pools/train_meta.parquet')
    if not path.is_file():
        temp = path.with_suffix('.partial.npy')
        np.lib.format.open_memmap(temp, mode='w+', dtype='uint8', shape=(len(metadata), enc.K, enc.PACKED)).flush()
        tasks = []
        for i, grade in enumerate(('D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D')):
            for j in range(0, 100000, 10000):
                tasks.append((f'train_{grade}.csv', i * 100000 + j, j, 10000, str(temp)))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for offset, records in pool.map(enc._train_task, tasks):
                expected = metadata.iloc[offset:offset + len(records)]
                if len(records) != 10000 or [row[1] for row in records] != expected.game_id.tolist():
                    raise ValueError('Training game order differs from the model roster')
        temp.replace(path)
    for name, (width, seed) in MODELS.items():
        options = argparse.Namespace(width=width, blocks=10, emb=256, seed=seed, batch=512,
                                     lr=.002, wd=.05, scale=30., margin=.2)
        enc.train(enc.WORK / name, metadata.row_id.to_numpy(), metadata.player_id.to_numpy(),
                  metadata.color.to_numpy(), 23438, options, eval_at=(23438,))


def encode_official(use_trained=False):
    check_device()
    for key, suffix, split in POOLS:
        items, layout = enc.read_pool(key, suffix, split)
        folder = HERE / 'outputs/embeddings' / key
        folder.mkdir(parents=True, exist_ok=True)
        packed = np.empty((len(items), enc.K, enc.PACKED), dtype='uint8')
        for i, (sgf, color) in enumerate(items):
            packed[i], count = enc.replay(sgf, color)
            if not count:
                raise ValueError(f'Empty game in {key} at row {i}')
        for name, (width, _) in MODELS.items():
            path = enc.WORK / name / 'final.pt' if use_trained else ARTIFACTS / 'encoders' / f'{name}.pt'
            model = enc.load_model(path, argparse.Namespace(width=width, blocks=10, emb=256))
            values = enc.encode(model, packed, np.arange(len(items)), np.array(layout['colors']))
            np.save(folder / f'{name}.npy', values)
        print(f'Encoded {key} from SGF', flush=True)
