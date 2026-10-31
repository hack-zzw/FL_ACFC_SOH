"""Persist and verify the registered source-only ACfC baseline."""
from __future__ import annotations
from utils.paths import ROOT as PACKAGE_ROOT
import hashlib
import json
from dataclasses import fields
from pathlib import Path
import numpy as np
from configs.schema import 实验配置, 特征列
from dataloader.preprocessing import 标准化器, 材料标准化器
PROJECT = PACKAGE_ROOT

def read_config(path: Path) -> 实验配置:
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    names = {field.name for field in fields(实验配置)}
    config = 实验配置(**{key: value for key, value in data.items() if key in names})
    config.启用物理约束 = False
    if any((getattr(config, name) != 0 for name in ('物理损失权重', '单调损失权重', '范围损失权重'))):
        raise ValueError('Frozen source requires zero training constraint weights')
    return config

def scaler_record(scaler) -> dict:

    def single(item):
        return {'mean': item.均值.tolist(), 'std': item.标准差.tolist(), 'metadata_columns': list(item.元数据列)}
    return {'global': single(scaler.全局), 'materials': {key: single(value) for key, value in scaler.分材料.items()}, 'feature_columns': 特征列 + ['cycle_index'], 'fitted_on': 'source_training_only'}

def load_scaler(path: Path):
    record = json.loads(path.read_text(encoding='utf-8'))

    def single(item):
        return 标准化器(np.asarray(item['mean'], dtype=np.float32), np.asarray(item['std'], dtype=np.float32), tuple(item['metadata_columns']))
    return 材料标准化器(single(record['global']), {key: single(value) for key, value in record['materials'].items()})

def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
