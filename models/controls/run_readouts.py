"""Source-only five-fold selection, final refit, then locked target scoring."""
import argparse
import gc
from common import *

def configs_for(stage):
    if stage == '01_capacity':
        return [dict(stage=stage, name=n, scope=n, kind='learned') for n in ['shared', 'material', 'domain']]
    if stage == '02_representation':
        return [dict(stage=stage, name=k, scope='domain', kind=k) for k in ['raw', 'random', 'learned']]
    if stage == '03_semantic_groups':
        return [dict(stage=stage, name='true_domain', scope='domain', kind='learned')] + [dict(stage=stage, name=f'pseudo_{i}', scope='domain', kind='learned', pseudo=i) for i in range(1, 6)]
    raise ValueError(stage)

def reused_domain(c, seed, fold):
    if c['stage'] == '01_capacity' or c['name'] not in ('learned', 'true_domain'):
        return None
    folder = 'all' if fold is None else f'fold{fold}'
    src = ROOT / '01_capacity/domain' / f'seed_{seed}' / folder
    if not (src / 'completion.json').exists():
        return None
    import shutil
    dst = ROOT / c['stage'] / c['name'] / f'seed_{seed}' / folder
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file() and path.name != 'completion.json':
            shutil.copy2(path, dst / path.name)
    result = js(src / 'completion.json')
    result['config'] = c
    result['reused_from'] = str(src)
    dump(dst / 'completion.json', result)
    return result

def main(stage):
    assert_protocol()
    configs = configs_for(stage)
    dest = ROOT / stage
    dest.mkdir(exist_ok=True)
    if (dest / 'completion.json').exists():
        print(stage + ' already completed', flush=True)
        return
    ts = tasks('source')
    registration = dest / 'stage_registration.json'
    if not registration.exists():
        dump(registration, dict(registered_at=stamp(), configs=configs, source_folds=5, script_sha256=sha(__file__), common_sha256=sha(Path(__file__).with_name('common.py')), source_cv='head-level; inherited frozen encoder and source scaler; not complete out-of-fold model training', selection_metric='domain-equal battery-macro MAE with fixed projected output', checkpoint_grid=list(CHECKPOINTS), optimizer='AdamW lr1e-3 weight_decay1e-4', batch=4096, loss='Huber delta .05', calibration='delta=sum(support_y-pred_support)/40', feature_whitelist_inherited=True, protocol_sha256=sha(PROJECT / 'configs/controls/protocol.json')))
    assert js(registration)['common_sha256'] == sha(Path(__file__).with_name('common.py'))
    selections = {c['name']: {} for c in configs}
    cvrows = []
    for s in SEEDS:
        for c in configs:
            x = None
            records = []
            for fold in range(5):
                result = reused_domain(c, s, fold)
                if result is None:
                    if x is None:
                        x = raw_features('source', s, c['kind'])
                    w = weights(ts, fold)
                    norm = norm_for(x, w, c['kind'], s, fold)
                    result = train_head_run(c, s, fold, x, norm, ts)
                records.extend(result['scores'])
            frame = pd.DataFrame(records)
            scores = frame.groupby('epoch').score.mean()
            best = float(scores.min())
            chosen = min((int(e) for e, v in scores.items() if v <= best + 1e-12))
            selections[c['name']][str(s)] = chosen
            cvrows.extend((dict(arm=c['name'], **r) for r in records))
            result = reused_domain(c, s, None)
            if result is None:
                if x is None:
                    x = raw_features('source', s, c['kind'])
                w = weights(ts, None)
                norm = norm_for(x, w, c['kind'], s, None)
                train_head_run(c, s, None, x, norm, ts, selected_epoch=chosen)
            del x
            gc.collect()
    dump(dest / 'source_selections.json', selections)
    pd.DataFrame(cvrows).to_csv(dest / 'source_validation_scores.csv', index=False)
    evaluate_heads(stage, configs, selections)
    dump(dest / 'completion.json', dict(passed=True, completed_at=stamp(), source_selected_epochs=selections, evaluation=js(dest / 'evaluation_completion.json'), main_project_and_paper_modified=False))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['01_capacity', '02_representation', '03_semantic_groups'])
    main(parser.parse_args().stage)
