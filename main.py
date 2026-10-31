"""Reproduce the paper: saved models -> fresh predictions -> statistics -> figures."""
import argparse
import json
import time
from utils.cli import ROOT, isolation_guard, output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['all', 'main', 'native', 'readouts', 'controls', 'ablations', 'external'], default='all', help='Default: all paper experiments and figures.')
    parser.add_argument('--output', help='Optional subdirectory under results/generated/.')
    parser.add_argument('--no-figures', action='store_true')
    parser.add_argument('--strict-main', action='store_true', help='Require bit-for-bit equality for the three main components.')
    parser.add_argument('--verify-only', action='store_true', help='Verify packaged file checksums without running inference.')
    args = parser.parse_args()
    isolation_guard()
    from models.predict import verify
    checked = verify(ROOT)
    if args.verify_only:
        print(json.dumps(checked), flush=True)
        return
    out = output_path(args.output)
    started = time.perf_counter()
    import torch
    torch.set_num_threads(4)
    from utils import replays
    suites = ['main', 'native', 'readouts', 'controls', 'ablations', 'external'] if args.suite == 'all' else [args.suite]
    for suite in suites:
        print(f'Running {suite} ...', flush=True)
        if suite == 'main':
            from models.predict import replay
            replay(ROOT, out / 'main', tolerance=0. if args.strict_main else 1e-6)
        else:
            getattr(replays, suite)(out)
    if args.suite == 'all':
        from utils.results import main as compute_results
        compute_results(out, figures=not args.no_figures)
    report = dict(passed=True, suite=args.suite, elapsed_seconds=time.perf_counter()-started,
                  model_inference_performed=True, statistics_from_current_predictions=args.suite == 'all',
                  historical_workspace_reads_blocked=True, output=str(out))
    (out / 'run_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
