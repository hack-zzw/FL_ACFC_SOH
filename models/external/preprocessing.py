"""Convert NASA ARC and Oxford battery MAT files to the project's feature schema.

The converter is deliberately separate from the model code. It reads the raw
archives, extracts one feature row per observed aging cycle, and writes only
CSV files plus an external-data manifest under the requested output directory.
"""
from __future__ import annotations
import numpy as np
FEATURES = ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness', 'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy', 'current mean', 'current std', 'current kurtosis', 'current skewness', 'CV Q', 'CV charge time', 'current slope', 'current entropy']

def _array(value) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)

def _finite_profile(voltage, current, time):
    voltage = _array(voltage)
    current = _array(current)
    time = _array(time)
    count = min(len(voltage), len(current), len(time))
    voltage, current, time = (voltage[:count], current[:count], time[:count])
    mask = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(time)
    voltage, current, time = (voltage[mask], current[mask], time[mask])
    if len(time) < 12:
        return None
    order = np.argsort(time, kind='stable')
    voltage, current, time = (voltage[order], current[order], time[order])
    keep = np.r_[True, np.diff(time) > 0]
    voltage, current, time = (voltage[keep], current[keep], time[keep])
    return (voltage, current, time)
