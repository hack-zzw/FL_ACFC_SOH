"""All nine NASA protocols: early-prefix federation and later-cycle evaluation."""
from __future__ import annotations
import itertools
import torch
from dataloader.external import 读取外部数据清单
from models.external.source import digest
from models.external.nasa_training import initialize, train, payload, restore, validate_support
SEEDS = (20260903, 20260904, 20260905)
SHOTS = (5, 10, 20)
METHODS = ('source_federated', 'source_local', 'source_frozen_local', 'random_federated')
HEADS = [dict(strength=s, alpha=a) for s, a in itertools.product((1.0, 10.0), (0.0, 0.5, 1.0))]
IMPLEMENTATION = ('models/external/nasa_training.py', 'models/external/nasa_pipeline.py', 'validation/test_nasa_all_protocol.py')

def key(candidate):
    return '_'.join((str(candidate[k]) for k in ('variant', 'prior_strength', 'scope', 'rounds')))

def clients_for(args):
    return {f'NASA_{i:02d}': 读取外部数据清单(args.manifests / f'NASA_{i:02d}.json') for i in range(1, 10)}

def get_models(args, phase, method, shot, seed, candidate, training, limit):
    path = args.output / 'models' / phase / method / f'{shot}_{seed}_{key(candidate)}.pt'
    reg_sha = digest(args.output / 'registration.json')
    source_path = args.source / f'source_seed_{seed}.pt'
    supports = {c: g[['battery_id', 'record_index', 'cycle_index', 'soh']].to_dict('list') for c, g in training.items()}
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['registration_sha256'] != reg_sha or saved['support_records'] != supports:
            raise ValueError('Cached provenance mismatch')
        return {c: restore(s, args.cfg, args.device, seed) for c, s in saved['models'].items()}
    validate_support(training, limit)
    src = torch.load(source_path, map_location='cpu', weights_only=True)
    if src['external_training']:
        raise ValueError('Must restart from source-only model')
    family = 'random' if method == 'random_federated' else 'source'
    groups = {'shared': training} if method.endswith('federated') else {c: {c: g} for c, g in training.items()}
    models, saved_models, saved_initial, logs, frozen = ({}, {}, {}, [], 0)
    for scope, local in groups.items():
        model = initialize(args.cfg, src['global_state'], seed, family, candidate['variant'], local, args.scaler, args.device, limit, candidate['prior_strength'])
        saved_initial[scope] = payload(model)
        ll, n = train(model, local, candidate, args.device, limit)
        frozen += n
        logs.extend([dict(scope=scope, **r) for r in ll])
        models[scope], saved_models[scope] = (model, payload(model))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(models=saved_models, initial_models=saved_initial, method=method, shots=shot, seed=seed, candidate=candidate, phase=phase, support_records=supports, training_prefix_limit=limit, participants=sorted(training), source_sha256=digest(source_path), registration_sha256=reg_sha, logs=logs, frozen_parameter_checks=frozen), path)
    return models
