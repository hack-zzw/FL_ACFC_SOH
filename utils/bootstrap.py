from pathlib import Path
import numpy as np
import pandas as pd
from utils.statistics import DOMAINS
VARIANTS = ('projected', 'clip_only')
ARMS = ('M1_shared_mapping', 'Mmid_material_mapping', 'M2_domain_mapping')
ARM = ARMS[1]

def material_pairs(batteries, out):
    rng = np.random.default_rng(20261027)
    template = batteries[(batteries.method == ARMS[0]) & (batteries.variant == VARIANTS[0])].sort_values('battery_id')
    ids = template.battery_id.to_numpy()
    domains = template.domain.to_numpy()
    groups = [np.flatnonzero(domains == d) for d in DOMAINS]
    draws = np.column_stack([rng.choice(g, size=(10000, len(g)), replace=True) for g in groups])
    rows, individual = ([], [])
    for variant in VARIANTS:
        for before, after in ((ARMS[0], ARM), (ARM, ARMS[2]), (ARMS[0], ARMS[2])):
            a = batteries[(batteries.method == before) & (batteries.variant == variant)].set_index('battery_id').loc[ids]
            b = batteries[(batteries.method == after) & (batteries.variant == variant)].set_index('battery_id').loc[ids]
            for metric in ('mae', 'r2'):
                delta = b[metric].to_numpy() - a[metric].to_numpy()
                samples = delta[draws].mean(axis=1)
                intervals = np.percentile(samples, [2.5, 97.5])
                rows.append({'variant': variant, 'before': before, 'after': after, 'metric': metric, 'estimand': 'equal75battery_mean_after_minus_before', 'difference': float(delta.mean()), 'pointwise_conditional_95_lower': float(intervals[0]), 'pointwise_conditional_95_upper': float(intervals[1]), 'batteries': 75, 'bootstrap_draws': 10000, 'primary_adjacent_comparison': (before, after) != (ARMS[0], ARMS[2])})
            for bid in ids:
                individual.append({'variant': variant, 'before': before, 'after': after, 'battery_id': bid, 'domain': a.loc[bid, 'domain'], 'mae_difference': b.loc[bid, 'mae'] - a.loc[bid, 'mae'], 'r2_difference': b.loc[bid, 'r2'] - a.loc[bid, 'r2']})
    pd.DataFrame(rows).to_csv(Path(out) / 'paired_bootstrap_intervals.csv', index=False)
    pd.DataFrame(individual).to_csv(Path(out) / 'paired_battery_differences.csv', index=False)

def contrast(b, reference, candidate, variant):
    a = b[(b.method == reference) & (b.variant == variant)].set_index('battery_id')
    c = b[(b.method == candidate) & (b.variant == variant)].set_index('battery_id').reindex(a.index)
    assert len(a) == len(c) == 75
    out = []
    rng = np.random.default_rng(20261028)
    for metric_name in ['mae', 'r2']:
        delta = (c[metric_name] - a[metric_name]).to_numpy()
        boot = np.zeros(5000)
        for domain in DOMAINS:
            ix = np.flatnonzero(a.domain.to_numpy() == domain)
            boot += delta[ix][rng.integers(len(ix), size=(5000, len(ix)))].sum(1) / 75
        lo, hi = np.quantile(boot, [0.025, 0.975])
        out.append(dict(reference=reference, candidate=candidate, variant=variant, metric=metric_name, paired_difference=float(delta.mean()), conditional95_low=float(lo), conditional95_high=float(hi), batteries=75, bootstrap_draws=5000))
    return out

def factorial(battery):
    effects = []
    for variant in VARIANTS:
        table = battery[battery.variant.eq(variant)].pivot(index='battery_id', columns='method', values='mae')
        domain = battery[battery.variant.eq(variant) & battery.method.eq('F00')].set_index('battery_id').domain.reindex(table.index)
        rng = np.random.default_rng(20261028)
        contrasts = {'domain_head_at_flat': table.F01 - table.F00, 'domain_head_at_hierarchy': table.F11 - table.F10, 'encoder_hierarchy_at_global_head': table.F10 - table.F00, 'encoder_hierarchy_at_domain_head': table.F11 - table.F01, 'interaction_difference_of_differences': table.F11 - table.F10 - (table.F01 - table.F00)}
        for name, delta in contrasts.items():
            boot = np.zeros(5000)
            values = delta.to_numpy()
            for dd in DOMAINS:
                ix = np.flatnonzero(domain.to_numpy() == dd)
                boot += values[ix][rng.integers(len(ix), size=(5000, len(ix)))].sum(1) / 75
            lo, hi = np.quantile(boot, [0.025, 0.975])
            effects.append(dict(variant=variant, contrast=name, battery_macro_mae_difference=float(delta.mean()), conditional95_low=float(lo), conditional95_high=float(hi), bootstrap_draws=5000))
    return pd.DataFrame(effects)
