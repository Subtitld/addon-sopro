"""Write sopro_addon/pins.json from the ONNX export's manifest at a revision.

    python tools/pin_model.py [REVISION]

Pins each file the CPU profile reads (graphs, weight shards, tokenizer,
manifest) by size and sha256. The graphs and shards come with sizes and hashes
in upstream's manifest; this checks them against Hugging Face's own LFS
metadata, and hashes the small files it downloads.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = 'samuel-vitorino/sopro-v2-turbo-onnx'
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from sopro_addon.engine import model_files  # noqa: E402


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main() -> int:
    revision = sys.argv[1] if len(sys.argv) > 1 else json.loads(get(f'https://huggingface.co/api/models/{REPO}'))['sha']
    tree = {item['path']: item for item in json.loads(get(
        f'https://huggingface.co/api/models/{REPO}/tree/{revision}?recursive=1'))}
    raw = get(f'https://huggingface.co/{REPO}/resolve/{revision}/manifest.json')
    manifest = json.loads(raw)
    files = []
    for name in ('manifest.json', manifest['tokenizer']):
        data = raw if name == 'manifest.json' else get(f'https://huggingface.co/{REPO}/resolve/{revision}/{name}')
        files.append({'path': name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    for item in model_files(manifest):
        meta = tree[item['path']]
        lfs = meta.get('lfs') or {}
        sha = lfs.get('oid') or lfs.get('sha256')
        if meta['size'] != item['bytes'] or (sha and sha != item['sha256']):
            raise SystemExit(f"{item['path']}: Hugging Face has {meta['size']} bytes / {sha}, the manifest {item['bytes']} / {item['sha256']}")
        files.append({'path': item['path'], 'size': item['bytes'], 'sha256': item['sha256']})
    out = {'repo': REPO, 'revision': revision, 'export_revision': manifest.get('revision'), 'files': files}
    (HERE / 'sopro_addon' / 'pins.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    print(f"{len(files)} files, {sum(f['size'] for f in files) / 1e6:.0f} MB at {revision}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
