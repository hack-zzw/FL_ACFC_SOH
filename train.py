"""Run an available training task with its fixed published configuration."""
import argparse
import runpy
import sys
from utils.cli import isolation_guard


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Example: python train.py controls --stage 01_capacity --method domain --seed 20260903. The source task trains the source encoder; it is not the complete final-method training chain.')
    parser.add_argument('task', choices=['source', 'heads', 'readouts', 'controls', 'comparisons', 'external', 'prepare'])
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    isolation_guard()
    # The selected task parses its own options. A fresh process keeps its
    # random-state initialization and task-specific module globals isolated.
    sys.argv = [f'train.py {args.task}'] + rest
    runpy.run_module('utils.tasks.' + args.task, run_name='__main__')


if __name__ == '__main__':
    main()
