"""Bounded CPU surface jobs, usable during GPU inference or as a postpass."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import time

from ..shared.data import atomic_json, digest, file_hash, validate_result
from .inputs import InputCache
from .records import verify_archive

_CACHE = None


@lru_cache(maxsize=1)
def analysis_identity():
    from surface_distance import metrics, lookup_tables
    from . import surfaces, inputs
    return dict(schema=1, versions={p: importlib.metadata.version(p) for p in
                ('numpy', 'Pillow', 'SimpleITK', 'scipy', 'surface-distance')},
                sources={Path(p).name: file_hash(p) for p in
                         (metrics.__file__, lookup_tables.__file__, surfaces.__file__, inputs.__file__, __file__)},
                tolerances=[1., 2., 3.])


def worker_init(cache_bytes):
    global _CACHE
    # Spawn rather than fork: CUDA state must not be inherited from inference.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    _CACHE = InputCache(cache_bytes)


def process_surface(row, path, case, root, destination, reuse_path=None):
    from .surfaces import analyze
    cache = _CACHE if _CACHE is not None else InputCache(0)
    started = time.monotonic()
    validate_result(row, case, row['mode'], row['run_id'])
    identity = analysis_identity()
    expected = dict(analysis_identity=identity, result_sha256=file_hash(path),
                    archive_sha256=row['prediction_archive']['sha256'], case_sha256=digest(case),
                    run_id=row['run_id'], unit_key=row['unit_key'], model=row['model'])
    reused = False
    for candidate in dict.fromkeys(p for p in (destination, reuse_path) if p is not None):
        if not Path(candidate).exists():
            continue
        result = json.loads(Path(candidate).read_text())
        if any(result.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Saved surface identity mismatch: {candidate}')
        saved_hash = result.pop('surface_sha256', None)
        if saved_hash != digest(result):
            raise ValueError(f'Saved surface result changed: {candidate}')
        result['surface_sha256'] = saved_hash
        verify_archive(path, row)
        cache.verify_case(case, root)
        reused = True
        break
    else:
        result = analyze(row, path, case, root, cache=cache)
        result.update(expected)
        result['surface_sha256'] = digest(result)
    if not Path(destination).exists():
        atomic_json(destination, result)
    # Per-slice data stay on disk; neither the queue nor aggregations retain them.
    compact = {k: result[k] for k in ('case', 'model', 'mode', 'sequence_condition',
                'scope', 'units', 'status', 'hd95', 'assd', 'nsd', 'unit_key')}
    return dict(result=compact, reused=reused, seconds=time.monotonic() - started,
                cache_stats=dict(cache.stats), path=str(destination))


class SurfacePool:
    def __init__(self, workers=1, cache_bytes=512 * 1024**2):
        if workers < 1 or cache_bytes < 0:
            raise ValueError('Require positive surface workers and nonnegative cache size')
        analysis_identity()  # Fail on missing dependencies before model loading.
        self.capacity = 2 * workers
        self.pending = set()
        self.executor = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=worker_init, initargs=(cache_bytes,))

    def collect(self, block=False):
        if not self.pending:
            return []
        done, _ = wait(self.pending, timeout=None if block else 0, return_when=FIRST_COMPLETED)
        self.pending.difference_update(done)
        return [f.result() for f in done]  # Propagate worker failures; never silently skip.

    def submit(self, *args):
        if len(self.pending) >= self.capacity:
            raise RuntimeError('Surface queue is full; collect completed work first')
        self.pending.add(self.executor.submit(process_surface, *args))

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
