"""Fresh common-information federated four-cell experiment; no main-project edits."""
import gc
from common import *
STAGE = '04_federated'
ROUNDS = 45
ARMS = {'F00': ('global', 'flat'), 'F01': ('domain', 'flat'), 'F10': ('global', 'hierarchical'), 'F11': ('domain', 'hierarchical')}

class FLModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = CfC编码器(17, hidden_dim=32, embedding_dim=36, layers=2, dropout=0.05, cfc_dt=1.0, tau_init='original_uniform')
        self.head = nn.Sequential(nn.Linear(487, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def features(self, numeric, core, domain):
        return torch.cat((numeric, self.encoder(core), torch.nn.functional.one_hot(domain, 5).float()), dim=1)

    def forward(self, numeric, core, domain):
        return self.head(self.features(numeric, core, domain)).squeeze(-1)

def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

def aggregate(states, weights, reference):
    total = sum(weights)
    clipped = []
    norms = []
    for state in states:
        delta = {k: state[k] - reference[k] for k in reference}
        norm = torch.sqrt(sum((torch.sum(v.double() ** 2) for v in delta.values()))).item()
        factor = min(1.0, 1.0 / max(norm, 1e-30))
        norms.append(norm)
        clipped.append({k: v * factor for k, v in delta.items()})
    result = {k: reference[k] + sum((delta[k] * (w / total) for delta, w in zip(clipped, weights))) for k in reference}
    return (result, norms)

def aggregate_encoder(records, mode, reference):
    if mode == 'flat':
        return aggregate([r['encoder'] for r in records], [r['weight'] for r in records], reference)[0]
    domain_states = {}
    for d in range(5):
        group = [r for r in records if r['domain'] == d]
        domain_states[d] = aggregate([r['encoder'] for r in group], [r['weight'] for r in group], reference)[0]
    chem_states = []
    for chem in CHEM:
        chem_states.append(aggregate([domain_states[d] for d in chem], [1] * len(chem), reference)[0])
    return aggregate(chem_states, [1] * 3, reference)[0]

def prepare_numeric(seed, source_tasks):
    raw = raw_features('source', seed, 'raw')
    w = weights(source_tasks, None)
    norm = norm_for(raw, w, 'raw', seed, None)
    return norm

def core_norm(ts):
    source = pd.read_pickle(PREP / 'source_observations.pkl')
    cols = list(特征列) + ['cycle_index']
    x = source[cols].to_numpy(float)
    d, _, _ = row_metadata(ts)
    w = weights(ts, None)
    norms = []
    for chem in CHEM:
        wc = w * np.isin(d, chem)
        wc /= wc.sum()
        norms.append(normalization(x, wc))
    return norms

def transform_core(part, ts, norms):
    frame = pd.read_pickle(PREP / f'{part}_observations.pkl')
    x = frame[list(特征列) + ['cycle_index']].to_numpy(float)
    d, _, _ = row_metadata(ts)
    out = np.empty_like(x, dtype=np.float32)
    for chem, norm in zip(CHEM, norms):
        sel = np.isin(d, chem)
        v = x[sel]
        good = np.isfinite(v)
        out[sel] = np.clip(np.where(good, (v - norm['mean']) / norm['sd'], 0), -10, 10)
    return out

def training_data(ts, norm, cnorm):
    numeric = torch.as_tensor(normalize(raw_features('source', SEEDS[0], 'raw'), norm), device=DEVICE)
    core = torch.as_tensor(transform_core('source', ts, cnorm), device=DEVICE)
    d, y, _ = row_metadata(ts)
    return (numeric, core, torch.as_tensor(d, device=DEVICE), torch.as_tensor(y, device=DEVICE))

def registered_stage():
    assert_protocol()
    dest = ROOT / STAGE
    dest.mkdir(exist_ok=True)
    path = dest / 'stage_registration.json'
    if path.exists():
        assert js(path)['script_sha256'] == sha(__file__)
        return
    seed_all(SEEDS[0])
    model = FLModel()
    parameters = dict(encoder=count(model.encoder), head=count(model.head))
    assert parameters == {'encoder': 6116, 'head': 15649}
    dump(path, dict(registered_at=stamp(), script_sha256=sha(__file__), protocol_sha256=sha(PROJECT / 'configs/controls/protocol.json'), arms={k: dict(head=v[0], encoder_aggregation=v[1]) for k, v in ARMS.items()}, parameters=parameters, global_head_logical_total=21765, domain_heads_logical_total=84361, input_information_identical=True, global_head_aggregation_constant_between_F00_F10=True, no_saved_paper_residuals_attached=True, fixed_rounds=45, local_epochs=2, per_battery_samples_per_epoch=128, batch=1024, head_target_epochs=3, source_cv_model_selection=False, hyperparameters_fixed_before_scoring=True, security='single-process simulation; no added secure protocol'))

def train_one(arm, seed, ts, data, norm, cnorm):
    head_level, mode = ARMS[arm]
    dest = ROOT / STAGE / arm / f'seed_{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    done = dest / 'completion.json'
    if done.exists():
        result = js(done)
        assert sha(dest / 'model.pt') == result['model_sha256']
        return
    assert not (ROOT / STAGE / 'target_scoring_started.json').exists()
    numeric, core, domain, y = data
    seed_all(seed)
    model = FLModel().to(DEVICE)
    encoder = cpu_state(model.encoder)
    initial_head = cpu_state(model.head)
    initial_sha = state_hash(model.state_dict())
    heads = {k: {n: v.clone() for n, v in initial_head.items()} for k in (['global'] if head_level == 'global' else list(range(5)))}
    clients = sorted({t['client_id'] for t in ts})
    logs = []
    first_round = 1
    existing = sorted(dest.glob('round_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
    if existing:
        saved = torch.load(existing[-1], map_location='cpu', weights_only=True)
        encoder = saved['encoder']
        heads = saved['heads']
        logs = saved['logs']
        first_round = saved['round'] + 1
    start = time.perf_counter()
    for round_id in range(first_round, ROUNDS + 1):
        records = []
        for ci, client in enumerate(clients):
            local = [t for t in ts if t['client_id'] == client]
            d = DOMAINS.index(local[0]['domain'])
            assert all((t['domain'] == DOMAINS[d] for t in local))
            key = 'global' if head_level == 'global' else d
            model.encoder.load_state_dict(encoder)
            model.head.load_state_dict(heads[key])
            model.train()
            local_seed = seed + round_id * 10000 + ci * 100
            seed_all(local_seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            rng = np.random.default_rng(local_seed)
            losses = []
            sample_hash = hashlib.sha256()
            steps = 0
            for epoch in range(2):
                ix = np.concatenate([rng.integers(t['start'] + 20, t['end'], size=128) for t in local])
                rng.shuffle(ix)
                sample_hash.update(ix.tobytes())
                for a in range(0, len(ix), 1024):
                    selected = torch.as_tensor(ix[a:a + 1024], device=DEVICE)
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(numeric[selected], core[selected], domain[selected])
                    loss = torch.nn.functional.huber_loss(pred, y[selected], delta=0.05)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    steps += 1
            assert np.isfinite(losses).all()
            records.append(dict(encoder=cpu_state(model.encoder), head=cpu_state(model.head), weight=len(local), domain=d))
            logs.append(dict(round=round_id, client_id=client, domain=DOMAINS[d], source_batteries=len(local), examples=2 * 128 * len(local), optimizer_steps=steps, loss=float(np.mean(losses)), sample_sha256=sample_hash.hexdigest()))
        encoder = aggregate_encoder(records, mode, encoder)
        if head_level == 'global':
            heads['global'] = aggregate([r['head'] for r in records], [r['weight'] for r in records], heads['global'])[0]
        else:
            for d in range(5):
                group = [r for r in records if r['domain'] == d]
                heads[d] = aggregate([r['head'] for r in group], [r['weight'] for r in group], heads[d])[0]
        if round_id % 15 == 0:
            atomic_torch(dest / f'round_{round_id}.pt', dict(round=round_id, encoder=encoder, heads=heads, logs=logs, initial_sha256=initial_sha, arm=arm, seed=seed))
            print(json.dumps(dict(stage=STAGE, arm=arm, seed=seed, round=round_id, elapsed_seconds=round(time.perf_counter() - start, 1), mean_last_round_loss=float(np.mean([r['loss'] for r in logs[-12:]])))), flush=True)
    atomic_torch(dest / 'model.pt', dict(encoder=encoder, heads=heads, arm=arm, seed=seed, normalization={k: torch.from_numpy(v) for k, v in norm.items()}, core_normalization=[{k: torch.from_numpy(v) for k, v in n.items()} for n in cnorm], initial_sha256=initial_sha, rounds=45, source_script_sha256=sha(__file__)))
    pd.DataFrame(logs).to_csv(dest / 'training_log.csv', index=False)
    dump(done, dict(passed=True, arm=arm, seed=seed, initial_sha256=initial_sha, model_sha256=sha(dest / 'model.pt'), rounds=45, client_rounds=len(logs), training_examples=sum((r['examples'] for r in logs)), optimizer_steps=sum((r['optimizer_steps'] for r in logs)), elapsed_seconds=time.perf_counter() - start))
    del model
    torch.cuda.empty_cache()

def adapted_predictions(model, head, features, t, seed):
    seed_all(seed)
    local = nn.Sequential(nn.Linear(487, 32), nn.SiLU(), nn.Linear(32, 1)).to(DEVICE)
    local.load_state_dict(head)
    local.train()
    y = torch.as_tensor(t['support_y'] - t['support_median'], dtype=torch.float32, device=DEVICE)
    optimizer = torch.optim.AdamW(local.parameters(), lr=0.0005, weight_decay=0.0001)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.huber_loss(local(features[:20]).squeeze(1), y, delta=0.05)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local.parameters(), 5.0)
        optimizer.step()
    local.eval()
    with torch.no_grad():
        residual = local(features).squeeze(1).cpu().numpy().astype(float)
    raw = t['support_median'] + residual
    return raw[20:] + np.sum(t['support_y'] - raw[:20]) / 40.0

def evaluate(ts):
    dest = ROOT / STAGE
    if (dest / 'evaluation_completion.json').exists():
        return
    for arm in ARMS:
        for seed in SEEDS:
            assert (dest / arm / f'seed_{seed}/completion.json').exists()
    dump(dest / 'target_scoring_started.json', dict(started_at=stamp(), all12_source_models_fixed=True, checkpoint_selection='fixed round45 before scoring', script_sha256=sha(__file__)))
    truth = pd.read_csv(PREP / 'target_truth.csv').sort_values(['battery_id', 'cycle_index']).reset_index(drop=True)
    target = tasks('target')
    d, _, _ = row_metadata(target)
    td = torch.as_tensor(d, device=DEVICE)
    pred_dir = dest / 'predictions'
    pred_dir.mkdir(exist_ok=True)
    metrics = []
    batteries = []
    domains = []
    components = []
    guards = []
    for arm, (head_level, mode) in ARMS.items():
        ensemble = {v: [] for v in VARIANTS}
        for seed in SEEDS:
            saved = torch.load(dest / arm / f'seed_{seed}/model.pt', map_location='cpu', weights_only=True)
            norm = {k: v.numpy() for k, v in saved['normalization'].items()}
            cnorm = [{k: v.numpy() for k, v in n.items()} for n in saved['core_normalization']]
            numeric = torch.as_tensor(normalize(raw_features('target', seed, 'raw'), norm), device=DEVICE)
            core = torch.as_tensor(transform_core('target', target, cnorm), device=DEVICE)
            model = FLModel().to(DEVICE)
            model.encoder.load_state_dict(saved['encoder'])
            model.eval()
            for parameter in model.encoder.parameters():
                parameter.requires_grad_(False)
            frozen_before = state_hash(model.encoder.state_dict())
            local = {v: [] for v in VARIANTS}
            for bi, t in enumerate(target):
                a, b = (t['start'], t['end'])
                di = DOMAINS.index(t['domain'])
                key = 'global' if head_level == 'global' else di
                with torch.no_grad():
                    features = model.features(numeric[a:b], core[a:b], td[a:b]).detach()
                adapt_seed = seed + 100000 + bi
                raw = adapted_predictions(model, saved['heads'][key], features, t, adapt_seed)
                replay = adapted_predictions(model, saved['heads'][key], features, t, adapt_seed)
                prefix = adapted_predictions(model, saved['heads'][key], features[:21], t, adapt_seed)
                replay_error = float(np.max(abs(replay - raw)))
                prefix_error = float(np.max(abs(prefix - raw[:len(prefix)])))
                assert replay_error < 1e-07 and prefix_error < 2e-05
                guards.append(dict(method=arm, seed=seed, battery_id=t['battery_id'], support_labels=20, target_encoder_frozen=True, query_labels_in_inputs=False, replay_max_abs=replay_error, prefix_max_abs=prefix_error))
                for v in VARIANTS:
                    local[v].append(pd.DataFrame(dict(battery_id=t['battery_id'], cycle_index=t['cycle_index'][20:], domain=t['domain'], pred_soh=post(raw, t['support_last'], v), raw_pred_soh=raw)))
            assert state_hash(model.encoder.state_dict()) == frozen_before
            for v in VARIANTS:
                f = pd.concat(local[v], ignore_index=True).sort_values(['battery_id', 'cycle_index']).reset_index(drop=True)
                pd.testing.assert_frame_equal(f[['battery_id', 'cycle_index']], truth[['battery_id', 'cycle_index']], check_dtype=False)
                f['true_soh'] = truth.true_soh.to_numpy()
                ensemble[v].append(f)
                f.assign(seed=seed).to_csv(pred_dir / f'{arm}__{v}__{seed}.csv.gz', index=False)
                a, _, _ = score_frame(f, arm, v)
                components.append(dict(seed=seed, **a))
            del model, numeric, core
            torch.cuda.empty_cache()
        for v in VARIANTS:
            f = ensemble[v][0].copy()
            f.pred_soh = np.mean([q.pred_soh.to_numpy() for q in ensemble[v]], axis=0)
            f.raw_pred_soh = np.mean([q.raw_pred_soh.to_numpy() for q in ensemble[v]], axis=0)
            f.to_csv(pred_dir / f'{arm}__{v}__ensemble.csv.gz', index=False)
            a, b, d = score_frame(f, arm, v)
            metrics.append(a)
            batteries.append(b)
            domains.append(d)
    battery = pd.concat(batteries, ignore_index=True)
    pd.DataFrame(metrics).to_csv(dest / 'overall_metrics.csv', index=False)
    battery.to_csv(dest / 'battery_metrics.csv', index=False)
    pd.concat(domains, ignore_index=True).to_csv(dest / 'domain_metrics.csv', index=False)
    pd.DataFrame(components).to_csv(dest / 'component_metrics.csv', index=False)
    pd.DataFrame(guards).to_csv(dest / 'prediction_guard_checks.csv', index=False)
    effects = []
    for variant in VARIANTS:
        table = battery[battery.variant.eq(variant)].pivot(index='battery_id', columns='method', values='mae')
        domain = battery[battery.variant.eq(variant) & battery.method.eq('F00')].set_index('battery_id').domain.reindex(table.index)
        rng = np.random.default_rng(20261028)
        contrasts = {'domain_head_at_flat': table.F01 - table.F00, 'domain_head_at_hierarchy': table.F11 - table.F10, 'encoder_hierarchy_at_global_head': table.F10 - table.F00, 'encoder_hierarchy_at_domain_head': table.F11 - table.F01, 'interaction_difference_of_differences': table.F11 - table.F10 - (table.F01 - table.F00)}
        for name, delta in contrasts.items():
            boot = np.zeros(5000)
            values = delta.to_numpy()
            for dd in DOMAINS:
                ix = np.flatnonzero(domain.to_numpy() == dd)
                boot += values[ix][rng.integers(len(ix), size=(5000, len(ix)))].sum(1) / 75
            lo, hi = np.quantile(boot, [0.025, 0.975])
            effects.append(dict(variant=variant, contrast=name, battery_macro_mae_difference=float(delta.mean()), conditional95_low=float(lo), conditional95_high=float(hi), bootstrap_draws=5000))
    pd.DataFrame(effects).to_csv(dest / 'factorial_paired_effects.csv', index=False)
    dump(dest / 'evaluation_completion.json', dict(passed=True, component_models=12, target_batteries=75, target_queries=58214, max_replay_error=max((g['replay_max_abs'] for g in guards)), max_prefix_error=max((g['prefix_max_abs'] for g in guards)), target_encoder_frozen=True, all_queries_retained=True, completed_at=stamp()))
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)

def main():
    registered_stage()
    ts = tasks('source')
    cnorm = core_norm(ts)
    dest = ROOT / STAGE
    if (dest / 'completion.json').exists():
        print(STAGE + ' already completed', flush=True)
        return
    for seed in SEEDS:
        norm = prepare_numeric(seed, ts)
        data = training_data(ts, norm, cnorm)
        for arm in ARMS:
            train_one(arm, seed, ts, data, norm, cnorm)
        del data
        gc.collect()
        torch.cuda.empty_cache()
    for seed in SEEDS:
        records = [js(dest / a / f'seed_{seed}/completion.json') for a in ARMS]
        assert len({r['initial_sha256'] for r in records}) == 1
        frames = [pd.read_csv(dest / a / f'seed_{seed}/training_log.csv') for a in ARMS]
        for f in frames[1:]:
            pd.testing.assert_frame_equal(frames[0][['round', 'client_id', 'sample_sha256', 'examples', 'optimizer_steps']], f[['round', 'client_id', 'sample_sha256', 'examples', 'optimizer_steps']])
    evaluate(ts)
    dump(dest / 'completion.json', dict(passed=True, completed_at=stamp(), source_models=12, identical_initial_parameters_and_sampling_verified=True, evaluation=js(dest / 'evaluation_completion.json')))
if __name__ == '__main__':
    main()
