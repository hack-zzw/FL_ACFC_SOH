"""Shared command setup; generated files stay under results/generated/."""
import os
import sys
from pathlib import Path
from utils.paths import ROOT

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '4')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')


def isolation_guard():
    """Reject accidental reads from the author's old workspace and backups."""
    package = str(ROOT.resolve()).lower()
    blocked = [str(p).lower() for p in ROOT.parents if p.name.lower() == 'claude_work']

    def audit(event, args):
        if event not in ('open', 'os.listdir', 'os.scandir') or not args:
            return
        if not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        name = os.path.abspath(os.fsdecode(args[0])).lower()
        if any(name.startswith(p + os.sep) for p in blocked):
            if name != package and not name.startswith(package + os.sep):
                raise RuntimeError(f'Read outside this reproduction project: {name}')
        if 'claude_work_backups' in name:
            raise RuntimeError(f'Historical backup access blocked: {name}')

    sys.addaudithook(audit)


def output_path(value=None):
    base = (ROOT / 'results/generated').resolve()
    path = Path(value).resolve() if value else base
    if not path.is_relative_to(base):
        raise ValueError('Choose an output directory under this project results/generated/.')
    path.mkdir(parents=True, exist_ok=True)
    os.environ['SOH_RUN_OUTPUT'] = str(path)
    return path
