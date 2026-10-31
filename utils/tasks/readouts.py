"""Refit source-only ridge controls; optionally repeat five-fold selection."""
import argparse, json, pickle
from utils.cli import *
parser = argparse.ArgumentParser()
parser.add_argument('--scope', choices=['global', 'material', 'domain'], default='domain')
parser.add_argument('--seed', type=int, choices=[20260903, 20260904, 20260905], default=20260903)
parser.add_argument('--output')
parser.add_argument('--select-from-source', action='store_true')
parser.add_argument('--isolation-check', action='store_true')
args = parser.parse_args()
out = output_path(args.output or str(ROOT / 'results/generated/training_readouts'))
if args.isolation_check:
    isolation_guard()
import numpy as np, pandas as pd, torch
from utils.replays import module
torch.set_num_threads(4)
from utils.paths import training_cache
c = module(ROOT / 'models/readouts/common.py', 'common')
c.PREP = training_cache()
fit = module(ROOT / 'models/readouts/fit_readout.py', 'ridge_fit')
material = module(ROOT / 'models/readouts/material.py', 'material_fit')
c.OUT = out
fit.OUT = out
(out / 'source_statistics').mkdir(exist_ok=True)
with (c.PREP / 'source_tasks.pkl').open('rb') as f:
    tasks = pickle.load(f)
x = np.column_stack([np.load(c.PREP / 'source_numeric.npy', mmap_mode='r'), np.load(c.PREP / f'source_embedding_{args.seed}.npy', mmap_mode='r')])
arm = {'global': 'M1_shared_mapping', 'material': 'Mmid_material_mapping', 'domain': 'M2_domain_mapping'}[args.scope]
ref = np.load(ROOT / f'checkpoints/readouts/{arm}/model_{args.seed}.npz')
selected = float(ref['lambda_value'])
scores = []
if args.select_from_source:
    for fold in range(5):
        stats = fit.stats_for_fold(x, tasks, args.seed, fold)
        candidates = material.candidates(stats) if args.scope == 'material' else fit.candidate_models(stats, arm)
        for t in tasks:
            if t['fold'] != fold:
                continue
            for model in candidates:
                if args.scope == 'material':
                    raw = material.predict_raw(model, x[t['start']:t['end']], t)
                    pred = material.post(raw, t['support_last'], 'projected')
                else:
                    _, pred = c.predict_task(model, x[t['start']:t['end']], t, 'projected')
                scores.append(dict(fold=fold, battery_id=t['battery_id'], domain=t['domain'], lambda_value=model['lambda_value'], mae=float(np.mean(abs(t['query_y'] - pred)))))
    cv = pd.DataFrame(scores)
    means = cv.groupby(['lambda_value', 'domain']).mae.mean().groupby('lambda_value').mean()
    selected = float(max((k for k, v in means.items() if v <= means.min() + 1e-12)))
    assert selected == float(ref['lambda_value'])
    cv.to_csv(out / f'{arm}_source_cv_{args.seed}.csv', index=False)
stats = fit.stats_for_fold(x, tasks, args.seed, None)
models = material.candidates(stats) if args.scope == 'material' else fit.candidate_models(stats, arm)
model = next((v for v in models if v['lambda_value'] == selected))
beta = model['beta']
intercept = model['intercept']
delta = max(float(np.max(abs(beta - ref['beta']))), float(np.max(abs(intercept - ref['intercept']))))
assert delta < 2e-09, delta
dest = out / arm
dest.mkdir(exist_ok=True)
np.savez(dest / f'model_{args.seed}.npz', beta=beta, intercept=intercept, lambda_value=selected, mean=stats['mean'], sd=stats['sd'], arm=arm)
report = dict(passed=True, scope=args.scope, seed=args.seed, lambda_value=selected, source_selected_again=args.select_from_source, maximum_parameter_difference=delta, source_batteries=len(tasks), target_labels_used=False)
(dest / f'verification_{args.seed}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
