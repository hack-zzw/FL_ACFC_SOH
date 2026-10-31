from utils.paths import ROOT
import os
R = __import__('pathlib').Path(os.environ.get('SOH_RUN_OUTPUT', str(ROOT / 'results/generated')))
P = R / 'figure_inputs'
T = R / 'metrics/paper'
M = R / 'metrics/material_readout'
B = R / 'metrics/controls'
for name in ['figures', 'source_data']:
    (R / name).mkdir(parents=True, exist_ok=True)
'Claim-led reconstruction of frozen quantitative evidence, without refitting.'
from pathlib import Path
import sys, json, csv
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
sys.stdout.reconfigure(encoding='utf-8')
DOMAINS = ['LFP|HUST', 'LFP|MIT', 'NCA|TJU', 'NCM|TJU', 'NCM|XJTU']
SHORT = ['HUST', 'MIT', 'NCA/TJU', 'NCM/TJU', 'XJTU']
TICKS = ['HUST', 'MIT', 'NCA/\nTJU', 'NCM/\nTJU', 'XJTU']
BLUE = '#2F5D8A'
TEAL = '#4C9692'
COPPER = '#C78963'
INK = '#303840'
GRAY = '#8FA2B3'
DOMAIN_COLORS = ['#477AA8', '#5BA59E', '#C99B52', '#9B83B4', '#8DA16C']
METHODS = ['FedAvg', 'FedProx', 'Material heads', 'Material × clusters', 'Centralized (17D)', 'Paper method']
COLORS = {'Paper method': BLUE, 'Centralized (17D)': '#5E9D97', 'Material × clusters': '#A18BB5', 'Material heads': '#D1AA62', 'FedProx': '#8BA7C3', 'FedAvg': '#B2BAC5'}
DISPLAY = {'Paper method': 'Complete method', 'Material heads': 'Chemistry heads', 'Material × clusters': 'Chemistry × clusters'}
ARMS = ['uniform_support_budget', 'no_monotonic_projection']
VARS = ['projected', 'clip_only']
VC = {'projected': '#244D72', 'clip_only': '#BE814C'}
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.titlesize': 8, 'axes.titleweight': 'normal', 'axes.linewidth': 0.65, 'axes.spines.right': False, 'axes.spines.top': False, 'lines.linewidth': 1.0, 'xtick.major.width': 0.65, 'ytick.major.width': 0.65, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 600, 'figure.dpi': 120, 'legend.frameon': False, 'text.color': INK, 'axes.labelcolor': INK})
export = []
counts = {}

def load(path):
    return pd.read_csv(path, float_precision='round_trip')

def fig(height):
    return plt.figure(figsize=(6.6929133858, height / 25.4))

def tag(ax, label):
    ax.set_gid(label)
    ax.annotate(label, (0, 1), xycoords='axes fraction', xytext=(-15, 9), textcoords='offset points', fontsize=9, fontweight='bold', ha='left', va='bottom', annotation_clip=False)

def save(f, name, axes, letters, exclude=None):
    f.canvas.draw()
    f.savefig(R / 'figures' / f'{name}.pdf', dpi=600, facecolor='white')
    f.savefig(R / 'figures' / f'{name}.svg', dpi=600, facecolor='white')
    f.savefig(R / 'figures' / f'{name}.png', dpi=600, facecolor='white')
    f.savefig(R / 'figures' / f'{name}.tiff', dpi=600, facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(f)
    print(name + ' exported', flush=True)

def identity(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], color=GRAY, lw=0.8, ls='--', zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Observed SOH')
    ax.set_ylabel('Predicted SOH')
metric = load(T / 'primary_battery_metrics.csv')
ci = load(T / 'conditional_battery_bootstrap_CI.csv')
primary = load(P / 'primary_predictions.csv.gz')
native = load(P / 'native_predictions.csv.gz')
assert len(primary) == 58214 and primary.battery_id.nunique() == 75
assert len(metric) == 75 and metric.domain.nunique() == 5
effects = load(B / 'planned_paired_effects.csv')
factor = load(B / 'federated_factorial_paired_effects.csv')

def effect(stage, reference, candidate, var):
    q = effects[(effects.stage == stage) & (effects.reference == reference) & (effects.candidate == candidate) & (effects.variant == var) & (effects.metric == 'mae')].iloc[0]
    return [float(q[x]) * 100 for x in ['paired_difference', 'conditional95_low', 'conditional95_high']]

def feffect(key, var):
    q = factor[(factor.contrast == key) & (factor.variant == var)].iloc[0]
    return [float(q[x]) * 100 for x in ['battery_macro_mae_difference', 'conditional95_low', 'conditional95_high']]
'Assemble the frozen Results into three self-contained main figures.'
from pathlib import Path
import hashlib, json, shutil
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'axes.titlesize': 7.5})
export.clear()
counts.clear()
NAMES = ['LFP/HUST', 'LFP/MIT', 'NCA/TJU', 'NCM/TJU', 'NCM/XJTU']
TICKS = ['LFP/\nHUST', 'LFP/\nMIT', 'NCA/\nTJU', 'NCM/\nTJU', 'NCM/\nXJTU']

def save(f, name, axes, letters, exclude=None, rows=None, columns=None):
    f.canvas.draw()
    f.savefig(R / 'figures' / f'{name}.pdf', dpi=600, facecolor='white')
    f.savefig(R / 'figures' / f'{name}.svg', dpi=600, facecolor='white')
    f.savefig(R / 'figures' / f'{name}.png', dpi=600, facecolor='white')
    f.savefig(R / 'figures' / f'{name}.tiff', dpi=600, facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(f)
    print(name + ' exported', flush=True)

def figure_main_accuracy():
    f = fig(200)
    axes = [f.add_axes([x, 0.755, 0.35, 0.155]) for x in [0.105, 0.61]]
    axes.extend([f.add_axes([x, 0.495, 0.35, 0.17]) for x in [0.105, 0.61]])
    axes.append(f.add_axes([0.105, 0.3, 0.855, 0.08]))
    width = 0.126
    gap = (0.855 - 5 * width) / 4
    axes.extend([f.add_axes([0.105 + i * (width + gap), 0.08, width, 0.13]) for i in range(5)])
    for a, l in zip(axes, 'abcdefghij'):
        tag(a, l)
    for j, (col, label) in enumerate([('mae', 'Battery MAE (SOH pp)'), ('r2', 'Battery R²')]):
        a = axes[j]
        for i, d in enumerate(DOMAINS):
            g = metric[metric.domain.eq(d)].sort_values('battery_id')
            scale = 100 if col == 'mae' else 1
            values = g[col].to_numpy() * scale
            a.scatter(i + np.linspace(-0.17, 0.17, len(g)), values, s=9, color=DOMAIN_COLORS[i], alpha=0.8, linewidths=0, zorder=2)
            q = ci[ci.domain.eq(d) & ci.metric.eq(col)].iloc[0]
            mean, lo, hi = [float(q[k]) * scale for k in ['estimate', 'low95', 'high95']]
            a.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt='D', ms=3.7, color=BLUE, capsize=2, lw=1, zorder=4)
            a.text(i, 1.045, f'{mean:.3f}' if col == 'mae' else f'{mean:.4f}', transform=a.get_xaxis_transform(), ha='center', va='bottom', fontsize=6.2, color=BLUE)
            export.append({'figure': 'Fig2', 'panel': 'ab'[j], 'domain': d, 'metric': col, 'estimate': mean, 'low95': lo, 'high95': hi, 'n': len(g)})
        a.set_xticks(range(5), [f'{n}\n(n={int(metric.domain.eq(d).sum())})' for n, d in zip(TICKS, DOMAINS)])
        a.set_xlim(-0.55, 4.55)
        a.set_ylabel(label)
        if col == 'r2':
            a.axhline(0.9, color=TEAL, ls='--', lw=0.7)
            a.set_ylim(min(0, float(metric.r2.min()) - 0.03), 1.075)
        else:
            a.set_ylim(0, float(metric.mae.max()) * 118)
    f.legend(handles=[Line2D([], [], marker='o', color=GRAY, ls='', ms=3, label='Individual battery'), Line2D([], [], marker='D', color=BLUE, ls='', ms=3.5, label='Domain mean and conditional 95% interval')], loc='upper center', bbox_to_anchor=(0.55, 0.992), ncol=2, columnspacing=2, handlelength=1.5)
    a = axes[2]
    hb = a.hexbin(primary.true_soh, primary.pred_soh, gridsize=42, mincnt=1, bins='log', cmap='Blues', linewidths=0, rasterized=True)
    assert int(hb.get_array().sum()) == 58214
    lo = min(primary.true_soh.min(), primary.pred_soh.min()) - 0.025
    hi = max(primary.true_soh.max(), primary.pred_soh.max()) + 0.025
    identity(a, lo, hi)
    a.set_title('58,214 queries; pooled R² = 0.9635', pad=10, fontsize=6.7)
    cax = f.add_axes([0.47, 0.495, 0.011, 0.17])
    cb = f.colorbar(hb, cax=cax)
    ticks = [v for v in [1, 10, 100, 1000] if v <= hb.get_array().max()]
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(v) for v in ticks])
    cb.minorticks_off()
    cb.ax.tick_params(labelsize=6.2, length=2)
    cb.set_label('Query count', fontsize=6.5, labelpad=1)
    a = axes[3]
    maxerror = 0
    ecdf_counts = {}
    for name in METHODS:
        g = native[native.method.eq(name)]
        assert len(g) == 58214 and g.battery_id.nunique() == 75
        e = np.sort(np.abs(g.pred_soh.to_numpy() - g.true_soh.to_numpy()) * 100)
        a.plot(np.r_[0, e], np.r_[0, np.arange(1, len(e) + 1) / len(e) * 100], color=COLORS[name], lw=1.6 if name == 'Paper method' else 0.9, label=DISPLAY.get(name, name), zorder=5 if name == 'Paper method' else 2)
        maxerror = max(maxerror, float(e.max()))
        ecdf_counts[name] = len(g)
    a.set_xscale('symlog', linthresh=1, linscale=0.55)
    a.set_xlim(0, maxerror * 1.03)
    a.set_ylim(0, 102)
    a.set_xticks([0, 1, 2, 5, 10, 20], ['0', '1', '2', '5', '10', '20'])
    a.set_xlabel('Absolute error (SOH pp; symlog)')
    a.set_ylabel('Queries within error (%)')
    f.legend(*a.get_legend_handles_labels(), loc='center', bbox_to_anchor=(0.55, 0.425), ncol=3, fontsize=6.3, handlelength=1.8, columnspacing=2.2, labelspacing=0.4)
    th = load(T / 'absolute_error_thresholds.csv')
    vals = np.array([[float(th[th.domain.eq(d) & th.threshold_pp.eq(t)].percent.iloc[0]) for t in [0.5, 1, 2, 3]] for d in DOMAINS])
    a = axes[4]
    a.imshow(vals.T, cmap='Blues', vmin=0, vmax=100, aspect='auto', interpolation='nearest')
    a.set_yticks(range(4), ['≤0.5', '≤1', '≤2', '≤3'])
    a.set_ylabel('Error (SOH pp)', fontsize=6.5)
    a.set_xticks(range(5), NAMES)
    a.tick_params(length=0)
    a.set_title('Query coverage within each absolute-error threshold', pad=7, fontsize=7)
    for i in range(4):
        for j in range(5):
            a.text(j, i, f'{vals[j, i]:.1f}%', ha='center', va='center', fontsize=6.5, color='white' if vals[j, i] >= 62 else INK)
    support = load(P / 'support_observations.csv')
    examples = load(P / 'representative_batteries.csv')
    representatives = []
    for i, d in enumerate(DOMAINS):
        a = axes[i + 5]
        q = examples[examples.domain.eq(d)].iloc[0]
        g = primary[primary.battery_id.eq(q.battery_id)].sort_values('cycle_index')
        s = support[support.battery_id.eq(q.battery_id)].sort_values('cycle_index')
        assert len(s) == 20 and len(g) > 0
        a.axvspan(0, s.cycle_index.max(), color=TEAL, alpha=0.14, lw=0)
        a.plot(s.cycle_index, s.soh, color=TEAL, lw=1)
        a.plot(g.cycle_index, g.true_soh, color=INK, lw=0.9)
        a.plot(g.cycle_index, g.pred_soh, color=BLUE, lw=1)
        a.set_title(NAMES[i], fontsize=6.7, pad=8)
        a.text(0.96, 0.94, f'R²={q.r2:.3f}', transform=a.transAxes, ha='right', va='top', fontsize=6.2, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.5})
        a.set_xlim(left=0)
        a.set_xlabel('Retained record', fontsize=6.2, labelpad=2)
        a.xaxis.set_major_locator(MaxNLocator(2, integer=True))
        a.yaxis.set_major_locator(MaxNLocator(3))
        a.tick_params(labelsize=6.2, pad=1.5)
        if i == 0:
            a.set_ylabel('SOH', fontsize=6.5, labelpad=2)
        representatives.append({'domain': d, 'battery_id': q.battery_id, 'queries': len(g), 'support': len(s), 'r2': float(q.r2)})
    f.legend(handles=[Line2D([], [], color=TEAL, label='Support observations'), Line2D([], [], color=INK, label='Observed query SOH'), Line2D([], [], color=BLUE, label='Predicted query SOH')], loc='lower center', bbox_to_anchor=(0.55, 0.002), ncol=3, fontsize=6.3, columnspacing=2)
    counts['Fig2'] = {'distribution_batteries_per_metric': 75, 'hexbin_queries': 58214, 'ecdf_queries_per_pipeline': ecdf_counts, 'coverage_cells': 20, 'representatives': representatives}
    save(f, 'Fig2', axes, 'abcdefghij', [cax], rows=[['a', 'b'], ['c', 'd'], list('fghij')], columns=[['a', 'c'], ['b', 'd']])

def reduction_bars(a, labels, fn, panel, title):
    x = np.arange(len(labels), dtype=float)
    for var, offset in [('projected', -0.17), ('clip_only', 0.17)]:
        rows = [fn(key, var) for _, key in labels]
        means = -np.array([r[0] for r in rows])
        low = -np.array([r[2] for r in rows])
        high = -np.array([r[1] for r in rows])
        assert np.all(low <= means) and np.all(means <= high)
        a.bar(x + offset, means, width=0.31, color=VC[var] if var == 'projected' else '#F1DCCB', edgecolor=VC[var], linewidth=0.8, yerr=np.vstack([means - low, high - means]), error_kw={'ecolor': INK, 'elinewidth': 0.7, 'capsize': 1.8, 'capthick': 0.7}, zorder=3)
        for (_, key), m, lo, hi in zip(labels, means, low, high):
            export.append({'figure': 'Fig3', 'panel': panel, 'contrast': str(key), 'variant': var, 'mae_reduction_pp': m, 'low95': lo, 'high95': hi, 'n': 75, 'bootstrap_draws': 5000})
    a.axhline(0, color=GRAY, lw=0.7, zorder=2)
    a.grid(axis='y', color='#E5E9EC', lw=0.45)
    a.set_axisbelow(True)
    a.set_xticks(x, [l for l, _ in labels])
    a.tick_params(axis='x', length=0, pad=4, labelsize=6.3)
    a.set_ylabel('MAE reduction (SOH pp)', fontsize=6.7, labelpad=3)
    a.set_title(title, fontsize=7.5, pad=10)
    a.yaxis.set_major_locator(MaxNLocator(4))
    a.margins(x=0.12, y=0.2)

def figure_boundary_controls():
    f = fig(184)
    axes = [f.add_axes([x, y, 0.35, 0.19]) for y in [0.71, 0.415, 0.12] for x in [0.105, 0.61]]
    for a, l in zip(axes, 'abcdef'):
        tag(a, l)
    dm = load(M / 'domain_metrics.csv')
    arms = ['M1_shared_mapping', 'Mmid_material_mapping', 'M2_domain_mapping']
    matrix = np.array([[float(dm[(dm.method == arm) & (dm.variant == var) & (dm.domain == d)].battery_macro_r2.iloc[0]) for d in DOMAINS] for var in VARS for arm in arms])
    assert matrix.shape == (6, 5) and np.isfinite(matrix).all()
    a = axes[0]
    a.imshow(matrix, cmap='Blues', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    a.set_xticks(range(5), TICKS)
    a.set_yticks(range(6), ['Shared: RM', 'Chem.: RM', 'Domain: RM', 'Shared: clip', 'Chem.: clip', 'Domain: clip'])
    a.tick_params(length=0, pad=2, labelsize=6.2)
    a.set_title('Mapping granularity: domain-mean R²', pad=10, fontsize=7.1)
    for i in range(6):
        for j in range(5):
            a.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', fontsize=6.5, color='white' if matrix[i, j] > 0.63 else INK)
    pairs = load(M / 'paired_battery_differences.csv')
    intervals = load(M / 'paired_bootstrap_intervals.csv')
    a = axes[1]
    for i, var in enumerate(VARS):
        g = pairs[(pairs.variant == var) & (pairs.before == 'Mmid_material_mapping') & (pairs.after == 'M2_domain_mapping')].sort_values('battery_id')
        assert len(g) == 75
        vals = -g.mae_difference.to_numpy() * 100
        a.scatter(i + (np.arange(75) * 0.6180339887498949 % 1 - 0.5) * 0.3, vals, s=7, c=VC[var], alpha=0.4, linewidths=0, zorder=2)
        q = intervals[(intervals.variant == var) & (intervals.before == 'Mmid_material_mapping') & (intervals.after == 'M2_domain_mapping') & (intervals.metric == 'mae')].iloc[0]
        mean = -float(q.difference) * 100
        lo = -float(q.pointwise_conditional_95_upper) * 100
        hi = -float(q.pointwise_conditional_95_lower) * 100
        a.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt='D', color=INK, ms=4, lw=1.1, capsize=3, zorder=4)
        export.append({'figure': 'Fig3', 'panel': 'b', 'contrast': 'Chemistry minus domain', 'variant': var, 'mae_reduction_pp': mean, 'low95': lo, 'high95': hi, 'n': 75, 'bootstrap_draws': 10000})
    a.axhline(0, color=GRAY, ls='--', lw=0.7)
    a.set_xlim(-0.55, 1.55)
    a.set_xticks([0, 1], ['RM + clipping', 'Clipping only'])
    a.set_ylabel('MAE reduction (SOH pp)', fontsize=6.7, labelpad=3)
    a.set_title('Chemistry → domain: all 75 batteries', pad=10, fontsize=7.1)
    a.yaxis.set_major_locator(MaxNLocator(5))
    reduction_bars(axes[2], [('Shared →\ndomain', ('shared', 'domain')), ('Chemistry →\ndomain', ('material', 'domain'))], lambda k, v: effect('01_capacity', *k, v), 'c', 'Matched parameters (83,205)')
    reduction_bars(axes[3], [('Heads\nclient', 'domain_head_at_flat'), ('Heads\nhierarchy', 'domain_head_at_hierarchy'), ('Hierarchy\nglobal', 'encoder_hierarchy_at_global_head'), ('Hierarchy\ndomain', 'encoder_hierarchy_at_domain_head')], feffect, 'd', 'Information-matched federated controls')
    reduction_bars(axes[4], [('Learned vs\nrandom', 'random'), ('Learned vs\nno embedding', 'raw')], lambda k, v: effect('02_representation', k, 'learned', v), 'e', 'Learned representations')
    reduction_bars(axes[5], [(f'{i}', str(i)) for i in range(1, 6)], lambda k, v: effect('03_semantic_groups', f'pseudo_{k}', 'true_domain', v), 'f', 'Chemistry–source vs pseudo-grouping')
    axes[5].set_xlabel('Size-matched pseudo-grouping', fontsize=6.5, labelpad=3)
    f.legend(handles=[Patch(facecolor=VC['projected'], edgecolor=VC['projected'], label='Running minimum + clipping'), Patch(facecolor=VC['clip_only'], edgecolor=VC['clip_only'], label='Clipping only')], loc='upper center', bbox_to_anchor=(0.55, 0.992), ncol=2, columnspacing=2, handlelength=1.9, fontsize=6.7)
    counts['Fig3'] = {'domain_r2_cells': 30, 'paired_battery_points': 150, 'bar_effects': 26, 'batteries_per_effect': 75, 'bars_show': 'reference minus candidate MAE in percentage points', 'zero_crossing_intervals_preserved': True}
    save(f, 'Fig3', axes, 'abcdef', rows=[['a', 'b'], ['c', 'd'], ['e', 'f']], columns=[['a', 'c', 'e'], ['b', 'd', 'f']])
'Restyle the existing scientific panels without changing data or intervals.'
from pathlib import Path
import json, hashlib
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'axes.titlesize': 7.5})

def fig(height):
    return plt.figure(figsize=(6.6929133858, height / 25.4))
BLUE = '#225B91'
TEAL = '#44935B'
INK = '#303840'
GRAY = '#9BA8AF'
DOMAIN_COLORS = ['#5795C7', '#1E4C9C', '#345D82', '#7DBA7F', '#196937']
VC = {'projected': BLUE, 'clip_only': '#7DBA7F'}
COLORS = {'Paper method': BLUE, 'Centralized (17D)': '#196937', 'Material × clusters': '#44935B', 'Material heads': '#78AAC8', 'FedProx': '#91B4C8', 'FedAvg': '#A8AFB5'}
DISPLAY['FedProx'] = 'FedProx (default)'
export.clear()
counts.clear()

def save(f, name, axes, letters, exclude=None, rows=None, columns=None):
    for a in axes:
        a.grid(False, which='both', axis='both')
    f.canvas.draw()
    f.savefig(R / 'figures' / f'{name}.pdf', dpi=600, facecolor='white')
    f.savefig(R / 'figures' / f'{name}.svg', dpi=600, facecolor='white')
    f.savefig(R / 'figures' / f'{name}.png', dpi=600, facecolor='white')
    f.savefig(R / 'figures' / f'{name}.tiff', dpi=600, facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(f)
    print(name + ' exported', flush=True)

def reduction_bars(a, labels, fn, panel, title):
    x = np.arange(len(labels), dtype=float)
    for var, offset in [('projected', -0.17), ('clip_only', 0.17)]:
        rows = [fn(key, var) for _, key in labels]
        means = -np.array([r[0] for r in rows])
        low = -np.array([r[2] for r in rows])
        high = -np.array([r[1] for r in rows])
        assert np.all(low <= means) and np.all(means <= high)
        a.bar(x + offset, means, width=0.31, color=VC[var], edgecolor='none', linewidth=0, yerr=np.vstack([means - low, high - means]), error_kw={'ecolor': '#535B61', 'elinewidth': 0.55, 'capsize': 1.5, 'capthick': 0.55}, zorder=3)
        for (_, key), m, lo, hi in zip(labels, means, low, high):
            export.append({'figure': 'Fig3', 'panel': panel, 'contrast': str(key), 'variant': var, 'mae_reduction_pp': m, 'low95': lo, 'high95': hi, 'n': 75, 'bootstrap_draws': 5000})
    a.set_xticks(x, [l for l, _ in labels])
    a.tick_params(axis='x', length=0, pad=4, labelsize=6.3)
    a.set_ylabel('MAE reduction (SOH pp)', fontsize=6.7, labelpad=3)
    a.set_title(title, fontsize=7.5, pad=10)
    a.yaxis.set_major_locator(MaxNLocator(4))
    a.margins(x=0.12, y=0.2)
    if a.get_ylim()[0] < 0:
        a.axhline(0, color='#A0A8AE', lw=0.55, ls='--', zorder=1)

def figure_external():
    n = load(P / 'NASA_fixed_query.csv.gz')
    o = load(P / 'Oxford.csv.gz')
    nt = load(T / 'table5_NASA.csv')
    ot = load(T / 'table6_Oxford.csv')
    for k, g in n.groupby('shots'):
        assert len(g) == 200
        assert abs(np.abs(g.prediction - g.soh).mean() - float(nt[nt.shots.eq(k)].mae.iloc[0])) < 1e-12
    f = fig(138)
    gs = f.add_gridspec(2, 2, left=0.115, right=0.975, bottom=0.12, top=0.9, wspace=0.45, hspace=0.75)
    axes = [f.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    a, b, c, d = axes
    for ax, l in zip(axes, 'abcd'):
        tag(ax, l)
    a.plot(nt.shots, nt.mae * 100, 'o-', color=BLUE, ms=3.5)
    a.set_xticks([20, 40, 60])
    a.set_ylim(0, 5.4)
    a.set_xlabel('NASA support labels per battery')
    a.set_ylabel('Pooled MAE (SOH pp)')
    for x, y in zip(nt.shots, nt.mae * 100):
        a.annotate(f'{y:.2f}', (x, y), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=7)
    for bid, g in o.groupby('battery_id'):
        vals = [float(np.abs((h := g[g.shots.eq(k)]).raw_prediction - h.soh).mean() * 100) for k in ot.shots]
        b.plot(ot.shots, vals, color=GRAY, lw=0.65, alpha=0.6)
    b.plot(ot.shots, ot.mae * 100, 'o-', color=BLUE, ms=3.5)
    b.set_xticks([0, 5, 10, 20])
    b.set_ylim(0, 12.5)
    b.set_xlabel('Oxford support labels per battery')
    b.set_ylabel('Battery-macro MAE (SOH pp)')
    b.text(0.97, 0.92, 'Grey: individual batteries', transform=b.transAxes, ha='right', va='top', fontsize=6.5)
    b.annotate('0.350', (20, float(ot[ot.shots.eq(20)].mae.iloc[0] * 100)), xytext=(-4, 9), textcoords='offset points', ha='right', fontsize=7)
    sel = n[n.shots.isin([20, 60])]
    lo = min(float(sel.soh.min()), float(sel.prediction.min())) - 0.025
    hi = max(float(sel.soh.max()), float(sel.prediction.max())) + 0.025
    identity(c, lo, hi)
    for shots, color, marker in [(20, TEAL, '^'), (60, BLUE, 'o')]:
        g = n[n.shots.eq(shots)]
        c.scatter(g.soh, g.prediction, s=6, c=color, marker=marker, alpha=0.68, linewidths=0, label=f'{shots} labels')
    c.legend(loc='upper left', fontsize=6.5, handlelength=1)
    c.text(0.97, 0.06, '200 common queries', transform=c.transAxes, ha='right', fontsize=6.5)
    g = o[o.shots.eq(20)]
    assert len(g) == 111
    lo = min(float(g.soh.min()), float(g.raw_prediction.min())) - 0.015
    hi = max(float(g.soh.max()), float(g.raw_prediction.max())) + 0.015
    identity(d, lo, hi)
    for (bid, h), marker, color in zip(g.groupby('battery_id', sort=True), ['o', 's', '^'], [BLUE, TEAL, '#345D82']):
        d.scatter(h.soh, h.raw_prediction, s=8, c=color, marker=marker, alpha=0.65, linewidths=0, label=bid.rsplit('_', 1)[-1])
    d.legend(loc='upper left', fontsize=6.5, handlelength=1)
    d.text(0.97, 0.06, '111 queries / 3 batteries', transform=d.transAxes, ha='right', fontsize=6.5)
    counts['Fig4'] = {'NASA_queries_at_each_budget': 200, 'NASA_batteries': 10, 'Oxford_queries_at_each_budget': 111, 'Oxford_batteries': 3, 'redrawn': True}
    save(f, 'Fig4', axes, 'abcd')

def main():
    figure_main_accuracy()
    figure_boundary_controls()
    figure_external()
    pd.DataFrame(export).to_csv(R / 'source_data/plotted_summaries_and_effects.csv', index=False)
