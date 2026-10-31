"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import numpy as np
import random as random
import torch as torch
from dataloader.preprocessing import 材料标准化器 as 材料标准化器
from dataloader.preprocessing import 标准化器 as 标准化器

def restore_scaler(record):

    def single(item):
        return 标准化器(item['mean'].numpy(), item['std'].numpy(), tuple(item['metadata_columns']))
    return 材料标准化器(single(record['global']), {key: single(item) for key, item in record['materials'].items()}) if record['materials'] else single(record['global'])

def restore_rng(payload):
    random.setstate(payload['python_rng_state'])
    state = payload['numpy_rng_state']
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    torch.set_rng_state(payload['torch_rng_state'])
    if payload['cuda_rng_states']:
        torch.cuda.set_rng_state_all(payload['cuda_rng_states'])
