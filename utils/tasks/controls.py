"""Train the published fixed settings, or repeat source-fold model selection."""
import argparse, json, os
from utils.cli import *
parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['01_capacity', '02_representation', '03_semantic_groups', '04_federated'], default='01_capacity')
parser.add_argument('--seed', type=int, choices=[20260903, 20260904, 20260905], default=20260903)
parser.add_argument('--method', default='domain')
parser.add_argument('--output')
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--all', action='store_true')
parser.add_argument('--select-from-source', action='store_true')
parser.add_argument('--isolation-check', action='store_true')
args = parser.parse_args()
out = output_path(args.output or str(ROOT / 'results/generated/training'))
if args.all:
    if args.smoke or args.select_from_source:
        parser.error('--all uses the48 fixed full-training settings; select a single stage for smoke or source selection.')
    import subprocess, csv
    with (ROOT / 'configs/control_models.csv').open(encoding='utf-8', newline='') as f:
        jobs = list(csv.DictReader(f))
    logs = out / 'logs'
    logs.mkdir(exist_ok=True)
    for i, j in enumerate(jobs, 1):
        name = f"{j['stage']}_{j['method']}_{j['seed']}"
        command = [sys.executable, '-B', str(ROOT / 'train.py'), 'controls', '--stage', j['stage'], '--method', j['method'], '--seed', j['seed'], '--output', str(out)]
        if args.isolation_check:
            command.append('--isolation-check')
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')
        (logs / f'{name}.log').write_text(result.stdout, encoding='utf-8')
        result.check_returncode()
        print(json.dumps(dict(completed=i, total=len(jobs), model=name)), flush=True)
    raise SystemExit(0)
if args.isolation_check:
    isolation_guard()
from utils.replays import module
from utils.statistics import read
import torch
torch.set_num_threads(4)
from utils.paths import training_cache
c = module(ROOT / 'models/controls/common.py', 'common')
c.PREP = training_cache()
ts = c.tasks('source')
if args.select_from_source:
    assert not args.smoke
    if args.stage == '04_federated':
        module(ROOT / 'models/controls/run_federated.py', 'registered_fl').main()
    else:
        module(ROOT / 'models/controls/run_readouts.py', 'registered_readouts').main(args.stage)
else:
    inventory = read(ROOT / 'configs/control_models.csv')
    target = inventory[inventory.stage.eq(args.stage) & inventory.method.eq(args.method) & inventory.seed.eq(args.seed)]
    if len(target) != 1:
        parser.error('Choose a method present in manifests/control_models.csv for this stage.')
    ref = target.iloc[0]
    saved = torch.load(ROOT / ref.path, map_location='cpu', weights_only=True)
    if args.stage == '04_federated':
        f = module(ROOT / 'models/controls/run_federated.py', 'registered_fl')
        cnorm = f.core_norm(ts)
        norm = f.prepare_numeric(args.seed, ts)
        data = f.training_data(ts, norm, cnorm)
        if args.smoke:
            f.ROUNDS = 1
        f.train_one(args.method, args.seed, ts, data, norm, cnorm)
        trained = c.ROOT / args.stage / args.method / f'seed_{args.seed}/model.pt'
        if not args.smoke:
            actual = torch.load(trained, map_location='cpu', weights_only=True)
            assert c.state_hash(actual['encoder']) == c.state_hash(saved['encoder'])
            for k in saved['heads']:
                assert c.state_hash(actual['heads'][k]) == c.state_hash(saved['heads'][k])
    else:
        r = module(ROOT / 'models/controls/run_readouts.py', 'registered_readouts')
        config = next((v for v in r.configs_for(args.stage) if v['name'] == args.method))
        x = c.raw_features('source', args.seed, config['kind'])
        w = c.weights(ts, None)
        norm = c.normalization(x, w)
        if args.smoke:
            c.CHECKPOINTS = (1,)
        epoch = 1 if args.smoke else int(ref.selected_epoch)
        c.train_head_run(config, args.seed, None, x, norm, ts, selected_epoch=epoch)
        trained = c.ROOT / args.stage / args.method / f'seed_{args.seed}/all/epoch_{epoch}.pt'
        if not args.smoke:
            actual = torch.load(trained, map_location='cpu', weights_only=True)
            assert c.state_hash(actual['state']) == c.state_hash(saved['state']), 'Retrained state differs from the registered checkpoint'
    report = dict(passed=True, stage=args.stage, method=args.method, seed=args.seed, smoke=args.smoke, full_registered_training=not args.smoke, source_records_only=True, model=str(trained))
    dest = out / 'validation'
    dest.mkdir(exist_ok=True)
    (dest / f'training_{args.stage}_{args.method}_{args.seed}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
