"""Download and verify the large player-identification artifacts published as GitHub release assets.

Bundles (see player_identification/artifacts/manifest.json):
  inference  required to reproduce the submitted CSV with acceptance.py
  training   needed to retrain the shortlist reranker (train.py, acceptance.py --trained)
  encoders   needed to rebuild the E4 game embeddings from official SGF on CUDA

Examples:
  python download_artifacts.py                                   # inference bundle only
  python download_artifacts.py --bundle inference training       # also retraining inputs
  python download_artifacts.py --bundle all --from-dir D:/assets  # use zips downloaded by hand
  python download_artifacts.py --check                           # verify files already in place

Uses only the Python standard library. Every zip and every extracted file is checked against SHA-256.
"""
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'player_identification' / 'artifacts' / 'manifest.json'
DOWNLOADS = ROOT / '.downloads'


def sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def problems(bundle):
    """Return the files of a bundle that are missing or differ from the manifest."""
    bad = []
    for name, meta in bundle['files'].items():
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != meta['bytes'] or sha256(path) != meta['sha256']:
            bad.append(name)
    return bad


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'artifact-downloader'})
    with urllib.request.urlopen(request) as response, open(partial, 'wb') as out:
        total = int(response.headers.get('Content-Length') or 0)
        done, step = 0, 0
        while block := response.read(1 << 20):
            out.write(block)
            done += len(block)
            if total and done * 20 // total > step:
                step = done * 20 // total
                print(f'  {done / 2**20:,.0f} / {total / 2**20:,.0f} MiB', flush=True)
    partial.replace(dest)


def extract(archive, bundle):
    expected = set(bundle['files'])
    with zipfile.ZipFile(archive) as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        if set(names) != expected:
            raise ValueError(f'{archive.name}: archive members differ from the manifest')
        for info in zf.infolist():
            target = (ROOT / info.filename).resolve()
            if not target.is_relative_to(ROOT):
                raise ValueError(f'Unsafe path in archive: {info.filename}')
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst, 1 << 22)


def main():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    choices = list(manifest['bundles'])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', nargs='+', choices=choices + ['all'], default=['inference'])
    parser.add_argument('--from-dir', type=Path, help='Folder that already holds the release zip files')
    parser.add_argument('--keep-zips', action='store_true', help='Keep downloaded zips in .downloads/')
    parser.add_argument('--check', action='store_true', help='Only verify files that are already in place')
    args = parser.parse_args()
    selected = choices if 'all' in args.bundle else list(dict.fromkeys(args.bundle))

    failed = False
    for name in selected:
        bundle = manifest['bundles'][name]
        bad = problems(bundle)
        if not bad:
            print(f'[{name}] {len(bundle["files"])} files verified')
            continue
        if args.check:
            print(f'[{name}] {len(bad)} of {len(bundle["files"])} files missing or different, e.g. {bad[0]}')
            failed = True
            continue
        local = args.from_dir / bundle['asset'] if args.from_dir else None
        if local is not None and local.is_file():
            archive = local
        else:
            archive = DOWNLOADS / bundle['asset']
            if not archive.is_file() or sha256(archive) != bundle['sha256']:
                url = manifest['release']['url'] + bundle['asset']
                print(f'[{name}] downloading {bundle["asset"]} ({bundle["bytes"] / 2**20:,.0f} MiB) from {url}')
                download(url, archive)
        if sha256(archive) != bundle['sha256']:
            sys.exit(f'[{name}] checksum mismatch for {archive}')
        print(f'[{name}] extracting {len(bundle["files"])} files')
        extract(archive, bundle)
        bad = problems(bundle)
        if bad:
            sys.exit(f'[{name}] {len(bad)} extracted files failed verification, e.g. {bad[0]}')
        print(f'[{name}] {len(bundle["files"])} files verified')
        if archive.parent == DOWNLOADS and not args.keep_zips:
            archive.unlink()
    if DOWNLOADS.is_dir() and not any(DOWNLOADS.iterdir()):
        DOWNLOADS.rmdir()
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
