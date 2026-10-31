"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import numpy as np
import pandas as pd
from models.domain import support_prior as small
import torch as torch

def append_low(tasks, frame, scale):
    for t in tasks:
        g = frame[frame.battery_id == t['battery_id']].sort_values('cycle_index')
        support = g.iloc[:20]
        q = g.set_index('cycle_index', drop=False).loc[t['query'].cycle_index]
        selected = pd.concat([support, q], ignore_index=True)
        x = small.relative(support, selected.drop(columns=['soh', 'capacity']), ['low38_40_Q'])[:, 0] / scale
        valid = np.isfinite(x) & selected.low_window_valid.to_numpy(bool)
        extra = np.column_stack([np.where(valid, np.clip(x, -10, 10), 0.0), (~valid).astype(float)])
        assert len(extra) == len(t['tensor'])
        t['tensor'] = torch.cat([t['tensor'], torch.from_numpy(extra.astype(np.float32)).to('cuda')], dim=1)
        t['low_query_valid'] = valid[20:] if support.low_window_valid.sum() >= 10 else np.zeros(len(q), bool)
