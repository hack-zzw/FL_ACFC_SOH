"""NASA support-budget scaling for the no-PINN federated ACfC client."""
from __future__ import annotations
import numpy as np
import pandas as pd
from models.external.nasa_training import validate_support
SEEDS = (20260903, 20260904, 20260905)
SHOTS = (5, 10, 20, 40, 60)
METHODS = ('source_federated', 'source_local', 'source_frozen_local', 'random_federated')
HORIZON = 20

def split_scale(clients, shots, mode='capped', horizon=HORIZON):
    """Split chronological prefixes without allowing support/query overlap."""
    support, all_after, fixed, inventory = ({}, {}, {}, [])
    for client, frame in sorted(clients.items()):
        s_rows, a_rows, f_rows = ([], [], [])
        for battery_id, battery in frame.groupby('battery_id', sort=True):
            battery = battery.sort_values('cycle_index').copy().reset_index(drop=True)
            battery['record_index'] = np.arange(len(battery))
            eligible = len(battery) >= 22
            actual = min(shots, len(battery) - 2) if eligible else 0
            s_rows.append(battery.iloc[:actual]) if eligible else None
            a_rows.append(battery.iloc[actual:]) if eligible else None
            f_rows.append(battery.iloc[actual:actual + horizon]) if eligible else None
            inventory.append(dict(client=client, battery_id=battery_id, shots=shots, records=len(battery), eligible=eligible, support=actual, exact_budget=actual == shots, query_all=len(battery) - actual if eligible else 0, query_fixed=min(horizon, max(0, len(battery) - actual)) if eligible else 0))
        if not s_rows:
            raise ValueError(f'No eligible batteries for {client}, shots={shots}')
        support[client] = pd.concat(s_rows, ignore_index=True)
        all_after[client] = pd.concat(a_rows, ignore_index=True)
        fixed[client] = pd.concat(f_rows, ignore_index=True)
    validate_support(support, shots)
    validate_split(support, all_after, fixed, shots, horizon)
    return (support, all_after, fixed, pd.DataFrame(inventory))

def validate_split(support, query, fixed, shots, horizon):
    for client in support:
        for battery_id, s in support[client].groupby('battery_id'):
            s_idx = set(s.record_index.astype(int))
            a = query[client][query[client].battery_id == battery_id]
            f = fixed[client][fixed[client].battery_id == battery_id]
            if len(s_idx) > shots or s_idx != set(range(len(s_idx))):
                raise AssertionError('Invalid chronological support prefix')
            if s_idx.intersection(set(a.record_index.astype(int))) or s_idx.intersection(set(f.record_index.astype(int))):
                raise AssertionError('Support/query overlap')
            if len(f) > horizon or (len(f) and f.record_index.min() != len(s_idx)):
                raise AssertionError('Invalid fixed-horizon query')
