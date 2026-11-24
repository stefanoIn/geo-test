"""Bounded, single-process comparison against an explicit pre-optimisation Git revision."""
import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import geopandas as gpd
import pandas as pd
import psutil
import pyarrow.parquet as pq
import shapely
import geography.enrichment as enrichment
from experiments.geographical_exposure.enrich_metadata import DEFAULT_INVENTORIES, representative_centres


def measure(function, frame, regions, label):
    # Rebuild prepared geometry state for both methods; no warmed predicate cache advantage.
    shapely.destroy_prepared(regions.geometry.to_numpy())
    shapely.prepare(regions.geometry.to_numpy())
    tree = shapely.STRtree(regions.geometry)
    gc.collect()
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stopped = threading.Event()
    started = time.perf_counter()
    def monitor():
        last = started
        while not stopped.wait(.05):
            peak[0] = max(peak[0], process.memory_info().rss)
            if time.perf_counter() - last > 30:
                print(f'{label}: {time.perf_counter()-started:.0f}s elapsed', flush=True)
                last = time.perf_counter()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        result = function(frame, regions, tree)
        elapsed = time.perf_counter() - started
        peak[0] = max(peak[0], process.memory_info().rss)
    finally:
        stopped.set()
        thread.join()
    stats = dict(seconds=elapsed, records_per_second=len(frame)/elapsed,
                 baseline_rss_bytes=baseline, sampled_peak_rss_bytes=peak[0],
                 incremental_peak_rss_bytes=peak[0]-baseline)
    print(label, stats, flush=True)
    return result, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, help='Git revision containing the original _assign')
    parser.add_argument('--output', type=Path, default=Path('reports/pretraining_overlap/assignment_benchmark.json'))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    git = ['git', '-c', f'safe.directory={repo.as_posix()}', '-C', str(repo)]
    revision = subprocess.check_output(git+['rev-parse', args.reference], text=True).strip()
    source = subprocess.check_output(git+['show', f'{revision}:geography/enrichment.py'], text=True)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_assign')
    namespace = vars(enrichment).copy()
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<original _assign>', 'exec'), namespace)
    original = namespace['_assign']
    regions = gpd.read_parquet('data/environmental/resolve_2017_validated.parquet')
    enrichment.validate_regions(regions)
    report = dict(reference=revision, records_per_inventory=65536,
                  methodology='One timing per implementation/inventory; identical batch; prepared state reset; excludes loading, validation, hashing and output I/O.',
                  memory='RSS sampled every 50ms by a monitoring thread; same process, allocator retention may affect baselines; original result retained for exact comparison.',
                  optimized_source_sha256=hashlib.sha256(Path(enrichment.__file__).read_bytes()).hexdigest(),
                  versions=dict(pandas=pd.__version__, shapely=shapely.__version__), inventories={})
    for kind in ['ssl4eo', 'terramesh']:
        frame = representative_centres(kind, DEFAULT_INVENTORIES[kind], 65536)
        assert len(frame) == 65536
        expected, old_stats = measure(original, frame, regions, kind+' original')
        actual, new_stats = measure(enrichment._assign, frame, regions, kind+' optimized')
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        info = dict(original=old_stats, optimized=new_stats,
                    speedup=old_stats['seconds']/new_stats['seconds'], exact_equivalence=True,
                    batch_sha256=hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).values.tobytes()).hexdigest(),
                    source_counts=frame.source.value_counts().to_dict(),
                    assignment_counts=actual.assignment_status.value_counts().to_dict())
        # Read only selected completed-cache rows; never rewrite the cached enrichment.
        cache = Path('data/metadata/resolve_full/ssl4eo_enriched.parquet')
        if kind == 'ssl4eo' and cache.exists():
            cached = pd.read_parquet(cache, filters=[('source_row','in',frame.source_row.tolist())])
            cached = cached.sort_values('source_row').reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, cached, check_exact=True)
            info['completed_cache_equivalence'] = True
        report['inventories'][kind] = info
        print(kind, 'EXACT equivalence; speedup', info['speedup'], flush=True)
        del actual, expected, frame
        gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with enrichment.pending_output(args.output) as pending:
        pending.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved',args.output,flush=True)


if __name__ == '__main__':
    main()
