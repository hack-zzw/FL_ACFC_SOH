"""Train a Table1 reference strategy from its registered configuration."""
import argparse, dataclasses, json, time
from utils.cli import *
parser = argparse.ArgumentParser()
parser.add_argument('--configuration', choices=['centralized', 'fedavg', 'material_balanced', 'clustered', 'fedprox_mu_1e-4', 'fedprox_mu_1e-2', 'fedprox_mu_1e-1'], default='fedavg')
parser.add_argument('--seed', type=int, choices=[20260903, 20260904, 20260905], default=20260903)
parser.add_argument('--output')
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--isolation-check', action='store_true')
args = parser.parse_args()
out = output_path(args.output or str(ROOT / 'results/generated/training_native'))
if args.isolation_check:
    isolation_guard()
import torch, pandas as pd
from configs.schema import 实验配置
from dataloader.loader import 读取数据, 按电池划分训练验证
from dataloader.preprocessing import 拟合材料标准化器
from utils.runtime import 设置随机种子, 选择设备
from models.federated import 训练集中式, 训练联邦, 训练材料平衡共享编码器, 训练材料来源共享编码器
from models.comparisons import _拟合聚类, _簇标签
torch.set_num_threads(4)
strategy = 'fedprox' if args.configuration.startswith('fedprox') else args.configuration
sub = Path(args.configuration) / f'fedprox_seed_{args.seed}.pt' if strategy == 'fedprox' else Path(f'{strategy}_seed_{args.seed}.pt')
reference = torch.load(ROOT / 'checkpoints/native' / sub, map_location='cpu', weights_only=True)
cfg = 实验配置(**{k: v for k, v in reference['config'].items() if k in {f.name for f in dataclasses.fields(实验配置)}})
frame = 读取数据(ROOT / 'data/features', 划分种子=cfg.数据划分种子)
source, _ = 按电池划分训练验证(frame)
scaler = 拟合材料标准化器(source)
device = 选择设备(cfg.设备)
if args.smoke:
    cfg.联邦轮数 = cfg.本地训练轮数 = 1
设置随机种子(args.seed)
start = time.perf_counter()
if strategy == 'centralized':
    global_state, states = (训练集中式(source, scaler, cfg, device), {})
elif strategy in ['fedavg', 'fedprox']:
    global_state, states, _ = 训练联邦(source, scaler, cfg, device, 材料感知=False, 使用FedProx=strategy == 'fedprox')
elif strategy == 'material_balanced':
    global_state, states, _ = 训练材料平衡共享编码器(source, scaler, cfg, device)
else:
    k = source.material.astype(str).str.cat(source.source.astype(str), sep='|').nunique()
    sc, km = _拟合聚类(source, k, 随机种子=args.seed)
    labels = _簇标签(source, sc, km)
    routed = source.copy()
    routed['source'] = 'cluster_' + routed.battery_id.map(lambda b: labels[b]).astype(str)
    global_state, states, _ = 训练材料来源共享编码器(routed, scaler, cfg, device)
dest = out / args.configuration
dest.mkdir(exist_ok=True)
torch.save(dict(global_state=global_state, material_states=states, config=cfg.转字典(), seed=args.seed, strategy=strategy), dest / f'model_{args.seed}.pt')
same = lambda a, b: set(a) == set(b) and all((torch.equal(a[k], b[k]) for k in a))
exact = same(global_state, reference['global_state']) and set(states) == set(reference['material_states']) and all((same(s, reference['material_states'][d]) for d, s in states.items()))
if not args.smoke:
    assert exact, 'Retrained reference parameters differ from the registered checkpoint'
report = dict(passed=True, configuration=args.configuration, seed=args.seed, smoke=args.smoke, elapsed_seconds=time.perf_counter() - start, source_batteries=source.battery_id.nunique(), registered_parameter_state_bit_exact=exact)
(dest / f'verification_{args.seed}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
