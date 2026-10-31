"""Train the source encoder and source-domain states with the registered settings."""
import argparse, copy, json, time, random
from utils.cli import *
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, choices=[20260903, 20260904, 20260905], default=20260903)
parser.add_argument('--output')
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--isolation-check', action='store_true')
args = parser.parse_args()
out = output_path(args.output or str(ROOT / 'results/generated/training_source'))
if args.isolation_check:
    isolation_guard()
import numpy as np, torch
import pandas as pd
from configs.schema import 实验配置
from dataloader.loader import 读取数据, 按电池划分训练验证
from models.federated import 训练材料来源共享编码器
from dataloader.preprocessing import 拟合材料标准化器
from utils.runtime import 设置随机种子, 选择设备
from models.external.source import scaler_record
torch.set_num_threads(4)
reference = torch.load(ROOT / f'checkpoints/source/source_seed_{args.seed}.pt', map_location='cpu', weights_only=True)
cfg = 实验配置(**reference['config'])
cfg.随机种子 = [args.seed]
cfg.额外参数['数据目录'] = str(ROOT / 'data/features')
frame = 读取数据(ROOT / 'data/features', 划分种子=cfg.数据划分种子)
source, _ = 按电池划分训练验证(frame)
scaler = 拟合材料标准化器(source)
if args.smoke:
    cfg.联邦轮数 = cfg.本地训练轮数 = 1
设置随机种子(args.seed)
start = time.perf_counter()

def progress(r, *_):
    if r == 1 or r % 5 == 0:
        print(f'Source seed={args.seed}: round {r}/{cfg.联邦轮数}', flush=True)
global_state, domain_states, logs = 训练材料来源共享编码器(source, scaler, cfg, 选择设备(cfg.设备), round_callback=progress)
elapsed = time.perf_counter() - start
payload = dict(method='material_source_acfc', seed=args.seed, global_state=global_state, domain_states=domain_states, external_training=False, scaler=scaler_record(scaler), config=cfg.转字典(), torch_rng_state=torch.get_rng_state(), cuda_rng_states=torch.cuda.get_rng_state_all(), python_rng_state=random.getstate(), numpy_rng_state=dict(name=np.random.get_state()[0], keys=np.random.get_state()[1].tolist(), pos=np.random.get_state()[2], has_gauss=np.random.get_state()[3], cached_gaussian=np.random.get_state()[4]))
dest = out / 'source'
dest.mkdir(exist_ok=True)
path = dest / f'source_seed_{args.seed}.pt'
torch.save(payload, path)
(dest / 'source_scaler.json').write_text(json.dumps(scaler_record(scaler), indent=2), encoding='utf-8')
pd.DataFrame(logs).to_csv(dest / f'training_{args.seed}.csv', index=False)
same = lambda a, b: set(a) == set(b) and all((torch.equal(a[k], b[k]) for k in a))
exact = same(global_state, reference['global_state']) and all((same(v, reference['domain_states'][d]) for d, v in domain_states.items()))
if not args.smoke:
    assert exact, 'Source training completed, but checkpoint tensors differ from the published source initialization'
report = dict(passed=True, seed=args.seed, smoke=args.smoke, source_batteries=source.battery_id.nunique(), source_records=len(source), elapsed_seconds=elapsed, registered_source_state_bit_exact=exact, scope='source encoder and initial domain states; domain-fitting stages have separate entry points', checkpoint=str(path))
(dest / f'verification_{args.seed}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
