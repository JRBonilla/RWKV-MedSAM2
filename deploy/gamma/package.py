"""Package matching evaluation source and frozen plans; weights and data stay separate."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SUFFIXES = {'.py', '.yaml', '.yml', '.ini', '.json', '.cu', '.cpp', '.h', '.cuh'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plans', type=Path, required=True, help='Directory containing finalized positive.json and full_scan.json')
    parser.add_argument('--checkpoints', type=Path, required=True, help='Pinned five-model checkpoint specification')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or Path(str(args.output) + '.sha256').exists():
        raise FileExistsError(args.output)
    with tempfile.TemporaryDirectory() as temp:
        package = Path(temp) / 'gamma-performance'
        repo = package / 'repo'
        for directory in ('rwkv_medsam2', 'ext', 'dripp'):
            for source in (ROOT / directory).rglob('*'):
                if any(part in ('.git', '__pycache__', '.venv', 'checkpoints') or part.endswith('.egg-info') for part in source.parts):
                    continue
                if source.is_file() and not source.is_symlink() and source.suffix in SUFFIXES:
                    target = repo / source.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        for relative in ('notebooks/strict_video_protocol_lib.py', 'deploy/gamma/env.sh', 'deploy/gamma/performance-campaign.py'):
            target = repo / relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        engine = repo / 'rwkv_medsam2/evaluation'
        engine_files = {p.relative_to(engine).as_posix(): sha256(p) for p in sorted(engine.rglob('*.py'))
                        if 'tests' not in p.relative_to(engine).parts}
        from rwkv_medsam2.evaluation.performance.data import read_plan
        (package / 'plans').mkdir()
        for name in ('positive', 'full_scan'):
            path = args.plans / (name + '.json')
            plan = read_plan(path)
            if plan['engine_files'] != engine_files:
                raise ValueError('Plan/source mismatch: package the exact frozen source; do not relabel old measurements')
            shutil.copy2(path, package / 'plans' / path.name)
        shutil.copy2(args.checkpoints, package / 'checkpoints.json')
        for name in ('start.sh', 'tensorboard.sh', 'README.md'):
            shutil.copy2(ROOT / 'deploy/gamma' / name, package / name)
        files = [p for p in sorted(package.rglob('*')) if p.is_file()]
        (package / 'package-files.json').write_text(json.dumps({p.relative_to(package).as_posix(): sha256(p) for p in files}, indent=2) + '\n')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.output, 'x:gz') as archive:
            archive.add(package, arcname=package.name)
    Path(str(args.output) + '.sha256').write_text(sha256(args.output) + '  ' + args.output.name + '\n')
    print(f'Wrote {args.output}. No models were loaded or run.')


if __name__ == '__main__':
    main()
