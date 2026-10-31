"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
from models.domain import routing as scope

def transport(frame, cover):
    result = frame.copy()
    metadata = []
    for bid, g in frame.groupby('battery_id', sort=True):
        g = g.sort_values('cycle_index')
        s = g.iloc[:20]
        q = g.iloc[20:].drop(columns=['soh', 'capacity'])
        G, H, qa = scope.masks(cover, s, q)
        if not H.any():
            continue
        shift = max(0.0, qa['support_temperature'] - cover['temperature_upper'])
        for col in ['cc_temperature_mean', 'cv_temperature_mean']:
            result.loc[g.index, col] = result.loc[g.index, col] - shift
        metadata.append(dict(battery_id=bid, temperature_shift=shift, original_support_temperature=qa['support_temperature'], destination_support_temperature=cover['temperature_upper']))
    return (result, metadata)
