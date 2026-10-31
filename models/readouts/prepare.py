"""Freeze folds and design, check causal/label guards, then cache observations."""
import pickle
import platform
import numpy as np
import pandas as pd
import torch
from common import *
from dataloader.loader import 读取数据

def main():
    torch.set_num_threads(4)
    PREP.mkdir(parents=True, exist_ok=True)
    if (OUT / 'prepared/completion.json').exists():
        print('Prepared data already exists;use verified cache.', flush=True)
        return
    assert verify(R)['passed']
    inv = pd.read_csv(AUDIT / 'source_feature_asset_inventory.csv')
    inv['path'] = [str(PACKAGE / v) for v in inv.path]
    for item in inv.itertuples():
        assert sha(item.path) == item.sha256
    protocol = json.loads((AUDIT / 'minimal_comparator_protocol.json').read_text(encoding='utf-8'))
    dump(OUT / 'registration.json', dict(protocol=protocol, protocol_sha256=sha(AUDIT / 'minimal_comparator_protocol.json'), input_whitelist_sha256=sha(AUDIT / 'matched_comparator_input_whitelist.json'), source_cache_hashes=inv[['role', 'path', 'sha256']].to_dict('records'), source_training=False, refinements=dict(numeric_dtype='float32 cached design;float64 normalization/covariance/solve', statistics_weights='training source domain-equal,battery-equal,query-record-equal within battery', fold_normalization='post-support query records of training source batteries only', primary_CV_prediction='projected with same support20 intercept adaptation as target', optional_source_coverage_feature_summaries='not appended;all actual source observables are available', source_aggregation='single-process sufficient-statistics simulation;no secure protocol claim'), numeric_schema=numeric_schema(), base_numeric_dimensions=len(numeric_schema()), embedding_dimensions=36, normalized_design_dimensions=2 * (len(numeric_schema()) + 36), target_scoring_not_started=True, python=platform.python_version(), torch=torch.__version__, prepare_script_sha256=sha(__file__), common_script_sha256=sha(Path(__file__).with_name('common.py'))))
    core = 读取数据(R / 'data/features', 划分种子=20260903)
    core['domain'] = core.material + '|' + core.source
    truth = []
    allmeta = []
    guards = []
    folds = []
    for source, label in [(True, 'source'), (False, 'target')]:
        frame = merge_observables(core[core.partition.eq('source' if source else 'target_test')], source)
        arrays = []
        tasks = []
        offset = 0
        for bid, g in frame.groupby('battery_id', sort=True):
            g = g.sort_values('cycle_index').reset_index(drop=True)
            support = g.iloc[:20].copy()
            obs = g.drop(columns=['soh', 'capacity'])
            domain = str(g.domain.iloc[0])
            a = design(obs, support, domain)
            assert len(g) > 20
            task = dict(battery_id=bid, domain=domain, client_id=str(g.client_id.iloc[0]), start=offset, end=offset + len(g), support_y=support.soh.to_numpy(float), support_median=float(support.soh.median()), support_last=float(support.soh.iloc[-1]), cycle_index=g.cycle_index.to_numpy(), fold=int(hashlib.sha256(bid.encode('utf-8')).hexdigest(), 16) % 5, support_sha256=hashlib.sha256(pd.util.hash_pandas_object(support[CORE + ['soh']], index=False).values.tobytes()).hexdigest())
            if source:
                task['query_y'] = g.soh.iloc[20:].to_numpy(float)
                folds.append({k: task[k] for k in ['battery_id', 'domain', 'client_id', 'fold', 'support_sha256']})
            else:
                truth.append(g.iloc[20:][['battery_id', 'cycle_index', 'domain', 'soh']].rename(columns={'soh': 'true_soh'}))
                checks = 0
                for cut in sorted(set([21, 20 + (len(g) - 20) // 2])):
                    np.testing.assert_array_equal(design(obs.iloc[:cut], support, domain), a[:cut])
                    checks += 1
                try:
                    design(obs.assign(true_soh=-999.0), support, domain)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Query labels accepted')
                cut = 20 + max(1, (len(g) - 20) // 2)
                altered = obs.copy()
                numeric_cols = list(set(CORE + EXTRA + list(QV)) & set(altered))
                altered.loc[altered.index >= cut, numeric_cols] = 9999.0
                b = design(altered, support, domain)
                np.testing.assert_array_equal(a[:cut], b[:cut])
                guards.append(dict(battery_id=bid, domain=domain, prefix_checks=checks, query_label_rejected=True, future_invariant=True))
            tasks.append(task)
            arrays.append(a)
            offset += len(g)
        a = np.concatenate(arrays)
        np.save(PREP / f'{label}_numeric.npy', a)
        with (PREP / f'{label}_tasks.pkl').open('wb') as f:
            pickle.dump(tasks, f, protocol=5)
        frame.drop(columns=['capacity', 'soh']).to_pickle(PREP / f'{label}_observations.pkl')
        allmeta.append(dict(partition=label, batteries=len(tasks), records=offset, post_support_records=sum((t['end'] - t['start'] - 20 for t in tasks))))
        print(json.dumps(allmeta[-1]), flush=True)
    pd.DataFrame(folds).to_csv(OUT / 'source_battery_folds.csv', index=False)
    pd.DataFrame(guards).to_csv(OUT / 'feature_guard_checks.csv', index=False)
    pd.concat(truth).to_csv(OUT / 'prepared/target_truth.csv', index=False)
    pd.DataFrame([dict(index=i, name=name) for i, name in enumerate(numeric_schema() + [f'encoder:{j}' for j in range(36)])]).to_csv(OUT / 'design_schema.csv', index=False)
    fold_table = pd.DataFrame(folds)
    assert (fold_table.groupby(['domain', 'fold']).size() > 0).all()
    hashes = [dict(path=str(p.relative_to(OUT)), sha256=sha(p)) for p in sorted(PREP.glob('*')) if p.is_file()]
    dump(OUT / 'prepared/completion.json', dict(passed=True, partitions=allmeta, feature_guard_batteries=len(guards), prefix_checks=sum((g['prefix_checks'] for g in guards)), source_folds_sha256=sha(OUT / 'source_battery_folds.csv'), files=hashes, no_target_query_labels_in_task_or_observation_inputs=True))
    print('Preparation complete;no SOH readout fitted and no target metrics computed.', flush=True)
if __name__ == '__main__':
    main()
