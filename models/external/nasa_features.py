"""Shared helpers retained for final experiments and validation.

Historical search/runner code is recoverable from the organization backup.
"""
import numpy as np
import torch
from models.external.nasa_readout import hierarchical_head
PHYSICAL = ['charge_Ah', 'energy_Wh', 'duration_s', 'cc_Ah', 'cv_Ah', 'cc_s', 'cv_s', 'time_voltage_mean', 'time_voltage_std', 'time_current_mean', 'time_current_std', 'voltage_V_per_s', 'current_A_per_s', 'time_3p8_4p0_s', 'time_4p0_4p1_s', 'cv_observed', 'cycle_index']
POSITIVE = {'CC Q', 'CV Q', 'CC charge time', 'CV charge time', 'cycle_index', *PHYSICAL[:11], 'time_3p8_4p0_s', 'time_4p0_4p1_s'}

def head_with_alpha(sx, qx, y, sids, qids, prior, bias, strength, alpha):
    pred, fit = hierarchical_head(sx, qx, y, sids, qids, prior, bias, strength, 0.0)
    fit['delta'] = (np.asarray(fit['delta']) * alpha).tolist()
    shifts = {}
    for bid in sorted(set(sids)):
        mask = torch.tensor([b == bid for b in sids], device=y.device)
        shifts[bid] = y.double()[mask].mean() - bias - sx.double()[mask].mean(0) @ prior.double()
    base = qx.double() @ prior.double() + bias + torch.stack([shifts[b] for b in qids])
    return (base + alpha * (pred - base), fit)
