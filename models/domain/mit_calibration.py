"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
from models.domain import mit_residual as method
import numpy as np
import pandas as pd
from dataloader.reference_features import predict as predict_reference
DOMAIN = 'LFP|MIT'

def apply_quality(bundle, support, query_features, mixed_raw, current_final):
    if {'soh', 'capacity', 'true_soh'} & set(query_features):
        raise ValueError('Query labels forbidden')
    record = bundle['ordered_quality_residuals'][DOMAIN]
    rr, _ = predict_reference(bundle['reference_heads'][DOMAIN], support, query_features)
    ar = 2 * np.asarray(mixed_raw, float) - rr
    branch = pd.DataFrame(dict(cycle_index=query_features.cycle_index.to_numpy(), acfc_raw=ar, reference_raw=rr, incumbent=np.asarray(current_final, float)))
    return method.correct(record, support, query_features, branch)
