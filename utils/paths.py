"""Project locations resolved independently of the current working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / 'configs'


def training_cache():
    """Rebuild source descriptors and embeddings only when training needs them."""
    import subprocess
    import sys

    supplied = ROOT / 'data/prepared'
    required = ['source_numeric.npy', 'source_observations.pkl'] + [
        f'source_embedding_{seed}.npy' for seed in (20260903, 20260904, 20260905)
    ]
    if all((supplied / name).is_file() for name in required):
        return supplied
    output = ROOT / 'results/generated/feature_cache'
    cache = output / 'prepared'
    if not all((cache / name).is_file() for name in required) or not (cache / 'completion.json').is_file():
        subprocess.run([
            sys.executable, '-B', str(ROOT / 'train.py'), 'prepare',
            '--output', str(output), '--isolation-check',
        ], check=True)
    return cache
