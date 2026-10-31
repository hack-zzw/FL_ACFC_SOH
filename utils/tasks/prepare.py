"""Rebuild the223 observable descriptors and36 frozen embeddings from supplied features."""
import argparse, json, os
from utils.cli import *
parser = argparse.ArgumentParser()
parser.add_argument('--output')
parser.add_argument('--isolation-check', action='store_true')
args = parser.parse_args()
out = output_path(args.output or str(ROOT / 'results/generated/data_rebuild'))
if args.isolation_check:
    isolation_guard()
from utils.replays import module
import numpy as np
os.environ['SOH_FEATURE_CACHE'] = str(out / 'prepared')
c = module(ROOT / 'models/readouts/common.py', 'common')
c.OUT = out
c.PREP = out / 'prepared'
m = module(ROOT / 'models/readouts/prepare.py', 'registered_preparation')
m.main()
import pandas as pd
for part in ['source', 'target']:
    observations = pd.read_pickle(c.PREP / f'{part}_observations.pkl')
    for seed in [20260903, 20260904, 20260905]:
        c.embedding(observations, seed, part)
checks = []
for p in c.PREP.glob('*.npy'):
    expected = ROOT / 'data/prepared' / p.name
    if expected.exists():
        a = np.load(p, mmap_mode='r')
        b = np.load(expected, mmap_mode='r')
        assert a.shape == b.shape
        equal = bool(np.array_equal(a, b, equal_nan=True))
        assert equal, p.name
        checks.append(dict(file=p.name, shape=list(a.shape), bit_exact=True))
(out / 'rebuild_verification.json').write_text(json.dumps(dict(passed=True, scope='processed features to observable descriptors and embeddings', checks=checks), indent=2), encoding='utf-8')
print('Feature reconstruction passed', flush=True)
