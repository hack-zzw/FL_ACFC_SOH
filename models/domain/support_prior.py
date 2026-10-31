"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import numpy as np

def relative(support, features, columns):
    if {'soh', 'capacity', 'true_soh'} & set(features):
        raise ValueError('Query labels forbidden')
    ref = np.nanmedian(support[list(columns)].to_numpy(float), axis=0)
    x = features[list(columns)].to_numpy(float)
    return (x - ref) / np.maximum(abs(ref), 1e-06)
