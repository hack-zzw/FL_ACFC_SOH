"""Redraw quantitative paper figures from a completed local reproduction run."""
import argparse
import json
from utils.cli import isolation_guard, output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Completed run directory under results/generated/.')
    args = parser.parse_args()
    isolation_guard()
    output = output_path(args.output)
    report = output / 'validation/result_recomputation.json'
    if not report.is_file() or not json.loads(report.read_text(encoding='utf-8'))['passed']:
        raise SystemExit('Run python main.py first. Plotting uses that run\'s predictions and statistics.')
    from utils.figures import main as draw
    draw()


if __name__ == '__main__':
    main()
