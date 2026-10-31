"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import numpy as np
from dataloader.reference_features import predict as predict

def apply_reference(base, frame, model):
    """The predictor receives current features and20 support labels only."""
    result = base.copy()
    for bid, b in frame.groupby('battery_id', sort=True):
        b = b.sort_values('cycle_index')
        support = b.iloc[:20]
        query = b.iloc[20:]
        raw, pred = predict(model, support, query.drop(columns=['soh', 'capacity']))
        idx = result.index[result.battery_id == bid]
        old = result.loc[idx].sort_values('cycle_index')
        np.testing.assert_array_equal(old.cycle_index, query.cycle_index)
        np.testing.assert_array_equal(old.true_soh.to_numpy(), query.soh.to_numpy())
        result.loc[old.index, 'pred_soh'] = 0.5 * old.pred_soh.to_numpy() + 0.5 * pred
        result.loc[old.index, 'raw_pred_soh'] = 0.5 * old.raw_pred_soh.to_numpy() + 0.5 * raw
    return result
