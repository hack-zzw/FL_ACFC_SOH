"""Shared helpers retained for final experiments and validation.

Historical search/runner code is recoverable from the organization backup.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from models.federated import 加载状态
from models.encoder import 按配置创建模型
from utils.runtime import 设置随机种子
COMMON_START = 20

def split_client(frame: pd.DataFrame, shots: int):
    support, query = ([], [])
    for _, battery in frame.groupby('battery_id', sort=True):
        battery = battery.sort_values('cycle_index').copy()
        if len(battery) < COMMON_START + 2:
            continue
        battery['record_index'] = np.arange(len(battery))
        support.append(battery.iloc[:shots])
        query.append(battery.iloc[max(shots, COMMON_START):])
    if not query:
        raise ValueError('Client has no battery with at least 22 valid records')
    return (pd.concat(support, ignore_index=True), pd.concat(query, ignore_index=True))

def initialized_model(config, state, device, seed):
    设置随机种子(seed)
    model = 按配置创建模型(config).to(device)
    if state is not None:
        加载状态(model, state, device)
    return model
