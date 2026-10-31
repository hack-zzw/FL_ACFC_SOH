"""Independent boundary controls. No writes to the main project or prior evidence."""
from __future__ import annotations
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '4')
import numpy as np
import pandas as pd
import torch
from torch import nn
from utils.paths import ROOT as PACKAGE
ROOT = Path(os.environ.get('SOH_RUN_OUTPUT', str(PACKAGE / 'results/generated'))) / 'controls'
ROOT.mkdir(parents=True, exist_ok=True)
BASE = PACKAGE / 'data'
PREP = BASE / 'prepared'
PROJECT = PACKAGE
RELEASE = PROJECT
from models.encoder import 创建模型, CfC编码器
from configs.schema import 特征列
from utils.state import restore_scaler
SEEDS = (20260903, 20260904, 20260905)
DOMAINS = ('LFP|HUST', 'LFP|MIT', 'NCA|TJU', 'NCM|TJU', 'NCM|XJTU')
CHEM = ((0, 1), (2,), (3, 4))
CHECKPOINTS = (5, 10, 20)
VARIANTS = ('projected', 'clip_only')
DEVICE = torch.device('cuda')
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)

def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def stamp():
    return pd.Timestamp.now(tz='Asia/Shanghai').isoformat()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def atomic_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    torch.save(obj, temp)
    os.replace(temp, path)

def tasks(part):
    with (PREP / f'{part}_tasks.pkl').open('rb') as stream:
        result = pickle.load(stream)
    assert len(result) == (303 if part == 'source' else 75)
    if part == 'target':
        assert all(('query_y' not in t for t in result))
    return result

def metric(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    e = p - y
    den = float(np.sum((y - y.mean()) ** 2))
    return dict(mae=float(np.mean(abs(e))), rmse=float(np.sqrt(np.mean(e * e))), r2=float(1 - np.sum(e * e) / den) if den > 1e-15 else float('nan'), mape=float(np.mean(abs(e) / np.maximum(abs(y), 1e-08)) * 100))

def post(raw, anchor, variant):
    if variant == 'projected':
        raw = np.minimum.accumulate(np.r_[anchor, raw])[1:]
    return np.clip(raw, 0.0, 1.1)

def random_embedding(part, seed):
    dest = ROOT / 'prepared' / f'{part}_random_embedding_{seed}.npy'
    if dest.exists():
        return np.load(dest, mmap_mode='r')
    observations = pd.read_pickle(PREP / f'{part}_observations.pkl')
    assert not {'soh', 'capacity', 'true_soh'} & set(observations)
    bundle = torch.load(RELEASE / 'checkpoints/main' / f'seed_{seed}.pt', map_location='cpu', weights_only=True)
    scaler = restore_scaler(bundle['scaler'])
    seed_all(seed)
    model = 创建模型('acfc', cfc_dt=1.0, tau_init='original_uniform').to(DEVICE).eval()
    x = scaler.变换(observations)
    out = []
    with torch.no_grad():
        for a in range(0, len(x), 16384):
            out.append(model.solution.encoder(torch.as_tensor(x[a:a + 16384], dtype=torch.float32, device=DEVICE)).cpu().numpy())
    emb = np.concatenate(out)
    assert emb.shape == (len(x), 36) and np.isfinite(emb).all()
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, emb)
    dump(dest.with_suffix('.json'), dict(seed=seed, part=part, sha256=sha(dest), encoder='random initialisation; frozen', inherited_source_scaler=True, query_labels_accessed=False))
    return emb

def raw_features(part, seed, kind):
    numeric = np.load(PREP / f'{part}_numeric.npy', mmap_mode='r')
    if kind == 'raw':
        return np.asarray(numeric)
    emb = np.load(PREP / f'{part}_embedding_{seed}.npy', mmap_mode='r') if kind == 'learned' else random_embedding(part, seed)
    return np.column_stack((numeric, emb))

def row_metadata(ts):
    n = max((t['end'] for t in ts))
    d = np.empty(n, np.int64)
    y = np.zeros(n, np.float32)
    q = []
    for t in ts:
        a, b = (t['start'], t['end'])
        d[a:b] = DOMAINS.index(t['domain'])
        if 'query_y' in t:
            y[a + 20:b] = t['query_y'] - t['support_median']
        q.append(np.arange(a + 20, b))
    return (d, y, np.concatenate(q))

def weights(ts, fold):
    n = max((t['end'] for t in ts))
    w = np.zeros(n, np.float64)
    active = [t for t in ts if fold is None or int(t['fold']) != fold]
    counts = {d: sum((t['domain'] == d for t in active)) for d in DOMAINS}
    for t in active:
        a, b = (t['start'] + 20, t['end'])
        w[a:b] = 1 / (5 * counts[t['domain']] * (b - a))
    assert abs(w.sum() - 1) < 1e-12
    return w

def normalization(x, w):
    n = x.shape[1]
    m = np.zeros(n)
    s = np.zeros(n)
    mass = np.zeros(n)
    for a in range(0, len(x), 8192):
        v = np.asarray(x[a:a + 8192], float)
        ww = w[a:a + 8192]
        good = np.isfinite(v)
        safe = np.where(good, v, 0)
        mass += ww @ good
        m += ww @ safe
        s += ww @ (safe * safe)
    mean = np.divide(m, mass, out=np.zeros_like(m), where=mass > 0)
    var = np.divide(s, mass, out=np.zeros_like(s), where=mass > 0) - mean * mean
    sd = np.sqrt(np.maximum(var, 0))
    sd[sd < 1e-08] = 1
    return {'mean': mean, 'sd': sd}

def normalize(x, norm):
    x = np.asarray(x, dtype=np.float32)
    good = np.isfinite(x)
    z = np.where(good, (x - norm['mean']) / norm['sd'], 0.0)
    return np.column_stack((np.clip(z, -10, 10), (~good).astype(np.float32))).astype(np.float32)

def norm_for(x, w, kind, seed, fold):
    dest = ROOT / 'prepared' / f"norm_{kind}_{seed}_{(fold if fold is not None else 'all')}.npz"
    if dest.exists():
        a = np.load(dest)
        return {'mean': a['mean'], 'sd': a['sd']}
    norm = normalization(x, w)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dest, **norm)
    return norm

def pseudo_assignments(repetition, part):
    """Reassign batteries within chemistry; exact real-domain source group sizes."""
    ts = tasks(part)
    result = {}
    for chem in CHEM:
        rows = [t for t in ts if DOMAINS.index(t['domain']) in chem]
        rows.sort(key=lambda t: hashlib.sha256(f"boundary-pseudo-{repetition}|{part}|{t['battery_id']}".encode()).hexdigest())
        cursor = 0
        for d in chem:
            n = sum((DOMAINS.index(t['domain']) == d for t in rows))
            for t in rows[cursor:cursor + n]:
                result[t['battery_id']] = d
            cursor += n
        assert cursor == len(rows)
    assert len(result) == len(ts)
    return result

class HeadBank(nn.Module):
    """Dense implementation: equal head count and dense arithmetic across scopes."""

    def __init__(self, dim, scope):
        super().__init__()
        self.linear = nn.Linear(dim, 160)
        self.output = nn.Parameter(torch.zeros(160))
        self.intercepts = nn.Parameter(torch.zeros(5))
        mask = torch.zeros(5, 160)
        if scope == 'shared':
            mask[:] = 1
        elif scope == 'material':
            mask[:2, :54] = 1
            mask[2, 54:107] = 1
            mask[3:, 107:] = 1
        else:
            for d in range(5):
                mask[d, d * 32:(d + 1) * 32] = 1
        self.register_buffer('mask', mask)

    def forward(self, x, domain, route=None):
        route = domain if route is None else route
        return (torch.nn.functional.silu(self.linear(x)) * self.mask[route] * self.output).sum(1) + self.intercepts[domain]

def count(model):
    return sum((p.numel() for p in model.parameters() if p.requires_grad))

def routes_for(ts, pseudo=None):
    d, _, _ = row_metadata(ts)
    if pseudo is None:
        return d
    r = d.copy()
    for t in ts:
        r[t['start']:t['end']] = pseudo[t['battery_id']]
    return r

def forward_all(model, x, d, route):
    parts = []
    model.eval()
    with torch.no_grad():
        for a in range(0, len(x), 16384):
            parts.append(model(x[a:a + 16384], d[a:a + 16384], route[a:a + 16384]).cpu().numpy())
    result = np.concatenate(parts).astype(float)
    assert np.isfinite(result).all()
    return result

def task_raw(residual, t):
    v = t['support_median'] + residual[t['start']:t['end']]
    delta = float(np.sum(t['support_y'] - v[:20]) / 40.0)
    return v[20:] + delta

def source_score(residual, ts, fold):
    rows = []
    for t in ts:
        if int(t['fold']) != fold:
            continue
        pred = post(task_raw(residual, t), t['support_last'], 'projected')
        rows.append({'battery_id': t['battery_id'], 'domain': t['domain'], **metric(t['query_y'], pred)})
    frame = pd.DataFrame(rows)
    return (float(frame.groupby('domain').mae.mean().mean()), frame)

def assert_protocol():
    assert (PACKAGE / 'configs/controls/protocol.json').is_file()
    assert (PREP / 'source_tasks.pkl').is_file() and (PREP / 'target_truth.csv').is_file()

def train_head_run(config, seed, fold, x, norm, ts, selected_epoch=None):
    assert_protocol()
    tag = 'all' if fold is None else f'fold{fold}'
    dest = ROOT / config['stage'] / config['name'] / f'seed_{seed}' / tag
    done = dest / 'completion.json'
    if done.exists():
        result = js(done)
        assert result['config'] == config
        for item in result['checkpoints']:
            assert sha(dest / item['file']) == item['sha256']
        return result
    assert not (ROOT / config['stage'] / 'target_scoring_started.json').exists()
    dest.mkdir(parents=True, exist_ok=True)
    w = weights(ts, fold)
    d, y, _ = row_metadata(ts)
    routes = routes_for(ts, pseudo_assignments(config['pseudo'], 'source') if config.get('pseudo') else None)
    z = torch.as_tensor(normalize(x, norm), device=DEVICE)
    td = torch.as_tensor(d, device=DEVICE)
    tr = torch.as_tensor(routes, device=DEVICE)
    ty = torch.as_tensor(y, device=DEVICE)
    active = np.flatnonzero(w > 0)
    tw = torch.as_tensor(w * len(active), dtype=torch.float32, device=DEVICE)
    ti = torch.as_tensor(active, device=DEVICE)
    start_seed = seed + (1000 * (fold + 1) if fold is not None else 0)
    seed_all(start_seed)
    model = HeadBank(z.shape[1], config['scope']).to(DEVICE)
    init_hash = state_hash(model.state_dict())
    params = count(model)
    assert params == (z.shape[1] + 2) * 160 + 5
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    wanted = (selected_epoch,) if selected_epoch else CHECKPOINTS
    records = []
    checkpoints = []
    start = time.perf_counter()
    last_loss = 0.0
    for epoch in range(1, max(wanted) + 1):
        model.train()
        order = ti[torch.randperm(len(ti), device=DEVICE)]
        for offset in range(0, len(order), 4096):
            ix = order[offset:offset + 4096]
            opt.zero_grad(set_to_none=True)
            pred = model(z[ix], td[ix], tr[ix])
            loss = (torch.nn.functional.huber_loss(pred, ty[ix], delta=0.05, reduction='none') * tw[ix]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            last_loss = float(loss.detach())
        assert np.isfinite(last_loss)
        if epoch in wanted:
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            path = dest / f'epoch_{epoch}.pt'
            atomic_torch(path, dict(state=state, normalization={k: torch.from_numpy(v) for k, v in norm.items()}, config=config, seed=seed, fold=fold, epoch=epoch, input_dim=z.shape[1], initial_state_sha256=init_hash, parameters=params, source_only=True))
            checkpoints.append(dict(epoch=epoch, file=path.name, sha256=sha(path)))
            if fold is not None:
                residual = forward_all(model, z, td, tr)
                score, frame = source_score(residual, ts, fold)
                frame.assign(epoch=epoch, fold=fold, seed=seed).to_csv(dest / f'epoch_{epoch}_source_validation.csv', index=False)
                records.append(dict(epoch=epoch, score=score, fold=fold, seed=seed))
    result = dict(config=config, seed=seed, fold=fold, parameters=params, initial_sha256=init_hash, checkpoints=checkpoints, scores=records, epochs=max(wanted), optimizer_steps=int(np.ceil(len(active) / 4096)) * max(wanted), source_query_records=len(active), elapsed_seconds=time.perf_counter() - start, last_loss=last_loss)
    dump(done, result)
    print(json.dumps(dict(stage=config['stage'], arm=config['name'], seed=seed, fold=fold, seconds=round(result['elapsed_seconds'], 1), scores=records), ensure_ascii=False), flush=True)
    del z, model, opt
    torch.cuda.empty_cache()
    return result

def score_frame(frame, method, variant):
    b = []
    dom = []
    for bid, g in frame.groupby('battery_id', sort=True):
        b.append(dict(method=method, variant=variant, battery_id=bid, domain=g.domain.iloc[0], query_records=len(g), **metric(g.true_soh, g.pred_soh)))
    b = pd.DataFrame(b)
    for d, g in frame.groupby('domain', sort=True):
        bm = b[b.domain.eq(d)]
        dom.append(dict(method=method, variant=variant, domain=d, batteries=len(bm), query_records=len(g), **{'battery_macro_' + k: float(bm[k].mean()) for k in ['mae', 'rmse', 'r2', 'mape']}, **{'pooled_' + k: v for k, v in metric(g.true_soh, g.pred_soh).items()}))
    dm = pd.DataFrame(dom)
    overall = dict(method=method, variant=variant, batteries=len(b), query_records=len(frame), **{'pooled_' + k: v for k, v in metric(frame.true_soh, frame.pred_soh).items()}, **{'battery_macro_' + k: float(b[k].mean()) for k in ['mae', 'rmse', 'r2', 'mape']}, domain_equal_macro_mae=float(dm.battery_macro_mae.mean()), minimum_domain_r2=float(dm.battery_macro_r2.min()), pct_error_le_1pp=float(np.mean(abs(frame.true_soh - frame.pred_soh) <= 0.01) * 100))
    return (overall, b, dm)

def paired_summary(battery, arms, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for v in VARIANTS:
        for first, second in zip(arms[:-1], arms[1:]):
            a = battery[(battery.method == first) & (battery.variant == v)].set_index('battery_id')
            b = battery[(battery.method == second) & (battery.variant == v)].set_index('battery_id').reindex(a.index)
            if len(a) != 75 or len(b) != 75:
                continue
            ds = a.domain.to_numpy()
            for col in ['mae', 'r2']:
                delta = (b[col] - a[col]).to_numpy()
                boot = np.zeros(5000)
                for domain in DOMAINS:
                    ix = np.flatnonzero(ds == domain)
                    draw = delta[ix][rng.integers(len(ix), size=(5000, len(ix)))]
                    boot += draw.sum(1) / 75
                lo, hi = np.quantile(boot, [0.025, 0.975])
                rows.append(dict(variant=v, reference=first, candidate=second, metric=col, mean_paired_difference=float(delta.mean()), conditional_95_low=float(lo), conditional_95_high=float(hi), bootstrap_draws=5000, conditioning='fixed fitted 3-component ensemble; stratified paired battery resampling'))
    return pd.DataFrame(rows)

def evaluate_heads(stage, configs, selections):
    dest = ROOT / stage
    start = dest / 'target_scoring_started.json'
    if (dest / 'evaluation_completion.json').exists():
        return
    for c in configs:
        for s in SEEDS:
            assert (dest / c['name'] / f'seed_{s}' / 'all/completion.json').exists()
    dump(start, dict(started_at=stamp(), source_selection_sha256=sha(dest / 'source_selections.json'), all_final_heads_fixed=True))
    ts = tasks('target')
    truth = pd.read_csv(PREP / 'target_truth.csv').sort_values(['battery_id', 'cycle_index']).reset_index(drop=True)
    assert len(truth) == 58214 and truth.battery_id.nunique() == 75
    d, _, _ = row_metadata(ts)
    td = torch.as_tensor(d, device=DEVICE)
    all_metrics = []
    all_b = []
    all_d = []
    component = []
    guards = []
    pred_dir = dest / 'predictions'
    pred_dir.mkdir(exist_ok=True)
    for c in configs:
        parts = {v: [] for v in VARIANTS}
        for s in SEEDS:
            epoch = int(selections[c['name']][str(s)])
            model_file = dest / c['name'] / f'seed_{s}' / 'all' / f'epoch_{epoch}.pt'
            saved = torch.load(model_file, map_location='cpu', weights_only=True)
            norm = {k: v.numpy() for k, v in saved['normalization'].items()}
            z = torch.as_tensor(normalize(raw_features('target', s, c['kind']), norm), device=DEVICE)
            route = routes_for(ts, pseudo_assignments(c['pseudo'], 'target') if c.get('pseudo') else None)
            tr = torch.as_tensor(route, device=DEVICE)
            model = HeadBank(saved['input_dim'], c['scope']).to(DEVICE)
            model.load_state_dict(saved['state'])
            residual = forward_all(model, z, td, tr)
            replay = HeadBank(saved['input_dim'], c['scope']).to(DEVICE)
            replay.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=True)['state'])
            replay_diff = float(np.max(abs(forward_all(replay, z, td, tr) - residual)))
            assert replay_diff <= 1e-07
            local = {v: [] for v in VARIANTS}
            for t in ts:
                raw = task_raw(residual, t)
                cut = min(t['end'], t['start'] + 21)
                with torch.no_grad():
                    r = model(z[t['start']:cut], td[t['start']:cut], tr[t['start']:cut]).cpu().numpy().astype(float)
                prefix = t['support_median'] + r
                prefix = prefix[20:] + np.sum(t['support_y'] - prefix[:20]) / 40.0
                diff = float(np.max(abs(prefix - raw[:len(prefix)]))) if len(prefix) else 0.0
                assert diff < 2e-05
                guards.append(dict(method=c['name'], seed=s, battery_id=t['battery_id'], query_labels_in_inputs=False, replay_max_abs=replay_diff, prefix_max_abs=diff))
                for v in VARIANTS:
                    local[v].append(pd.DataFrame(dict(battery_id=t['battery_id'], cycle_index=t['cycle_index'][20:], domain=t['domain'], raw_pred_soh=raw, pred_soh=post(raw, t['support_last'], v))))
            for v in VARIANTS:
                f = pd.concat(local[v], ignore_index=True).sort_values(['battery_id', 'cycle_index']).reset_index(drop=True)
                pd.testing.assert_frame_equal(f[['battery_id', 'cycle_index']], truth[['battery_id', 'cycle_index']], check_dtype=False)
                f['true_soh'] = truth.true_soh.to_numpy()
                parts[v].append(f)
                f.assign(seed=s).to_csv(pred_dir / f"{c['name']}__{v}__{s}.csv.gz", index=False)
                a, _, _ = score_frame(f, c['name'], v)
                component.append(dict(seed=s, **a))
            del model, replay, z
            torch.cuda.empty_cache()
        for v in VARIANTS:
            f = parts[v][0].copy()
            f.pred_soh = np.mean([q.pred_soh.to_numpy() for q in parts[v]], axis=0)
            f.raw_pred_soh = np.mean([q.raw_pred_soh.to_numpy() for q in parts[v]], axis=0)
            f.to_csv(pred_dir / f"{c['name']}__{v}__ensemble.csv.gz", index=False)
            a, b, dm = score_frame(f, c['name'], v)
            all_metrics.append(a)
            all_b.append(b)
            all_d.append(dm)
    overall = pd.DataFrame(all_metrics)
    batteries = pd.concat(all_b, ignore_index=True)
    overall.to_csv(dest / 'overall_metrics.csv', index=False)
    batteries.to_csv(dest / 'battery_metrics.csv', index=False)
    pd.concat(all_d, ignore_index=True).to_csv(dest / 'domain_metrics.csv', index=False)
    pd.DataFrame(component).to_csv(dest / 'component_metrics.csv', index=False)
    pd.DataFrame(guards).to_csv(dest / 'prediction_guard_checks.csv', index=False)
    paired_summary(batteries, [c['name'] for c in configs], 20261028).to_csv(dest / 'paired_battery_effects.csv', index=False)
    dump(dest / 'evaluation_completion.json', dict(passed=True, arms=len(configs), component_models=3 * len(configs), target_batteries=75, target_queries=58214, source_selection_before_target_scoring=True, all_queries_retained=True, queries_are_development_evaluation=True, max_replay_error=max((g['replay_max_abs'] for g in guards)), max_prefix_error=max((g['prefix_max_abs'] for g in guards)), completed_at=stamp()))
    print(overall.to_string(index=False), flush=True)
