"""Fit the registered conditional heads using source batteries and a frozen source encoder."""
import argparse, copy, json, time
from utils.cli import *
parser = argparse.ArgumentParser()
parser.add_argument('--domain', choices=['HUST', 'NCA', 'XJTU'], default='HUST')
parser.add_argument('--seed', type=int, choices=[20260903, 20260904, 20260905], default=20260903)
parser.add_argument('--output')
parser.add_argument('--isolation-check', action='store_true')
args = parser.parse_args()
out = output_path(args.output or str(ROOT / 'results/generated/training_domain'))
if args.isolation_check:
    isolation_guard()
import numpy as np, pandas as pd, torch
from configs.schema import 实验配置
from dataloader.loader import 读取数据, 按电池划分训练验证
from models.encoder import 按配置创建模型
from utils.state import restore_scaler
from models.domain import training as fitting
torch.set_num_threads(4)
domain = {'HUST': 'LFP|HUST', 'NCA': 'NCA|TJU', 'XJTU': 'NCM|XJTU'}[args.domain]
fitting.DOMAIN = domain
saved = torch.load(ROOT / f'checkpoints/main/seed_{args.seed}.pt', map_location='cpu', weights_only=True)
record = saved['conditional_heads'][domain]
if args.domain == 'XJTU':
    record = torch.load(ROOT / f'checkpoints/initialization/XJTU_prefix_{args.seed}.pt', map_location='cpu', weights_only=True)
cfg = 实验配置(**saved['config'])
model = 按配置创建模型(cfg).to('cuda').eval()
model.load_state_dict(saved['domain_states'][domain])
for p in model.parameters():
    p.requires_grad_(False)
scaler = restore_scaler(saved['scaler'])
frame = 读取数据(ROOT / 'data/features', 划分种子=20260903)
frame['domain'] = frame.material + '|' + frame.source
frame = frame[frame.partition.eq('source') & frame.domain.eq(domain)]
role = {'HUST': 'HUST_window', 'NCA': 'NCA_observation_linkage', 'XJTU': 'XJTU_window_canonical'}[args.domain]
frame = frame.merge(pd.read_pickle(ROOT / f'data/windows/source/{role}.pkl'), on=['battery_id', 'cycle_index'], validate='one_to_one')
prefix, _ = 按电池划分训练验证(frame)
tasks = fitting.build(prefix, model, scaler, sampled=True)
normalization = fitting.normalize(tasks)
for name in ['mean', 'sd']:
    np.testing.assert_array_equal(np.asarray(normalization[name]), np.asarray(record['normalization'][name]))
start = time.perf_counter()
head, log = fitting.fit(tasks, int(record['width']), args.seed, cfg)
state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
maximum = max((float(torch.max(abs(state[k] - record['state'][k]))) for k in state))
dest = out / args.domain
dest.mkdir(exist_ok=True)
torch.save(dict(state=state, normalization=normalization, domain=domain, seed=args.seed), dest / f'head_{args.seed}.pt')
log.to_csv(dest / f'training_{args.seed}.csv', index=False)
report = dict(passed=maximum == 0.0, domain=domain, seed=args.seed, elapsed_seconds=time.perf_counter() - start, source_batteries=prefix.battery_id.nunique(), normalization_bit_exact=True, maximum_parameter_difference=maximum, scope='registered initial conditional head; XJTU continuation and residual fitting are separate stages')
(dest / f'verification_{args.seed}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
assert maximum == 0.0, report
print(json.dumps(report), flush=True)
