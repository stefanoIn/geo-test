"""Enrich inventories and report centre-based ecological composition.

The default uses bounded validation subsets. --full explicitly enriches complete
inventories; repair comparisons always use the bounded subsets.
"""
from contextlib import closing
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from tqdm.auto import tqdm
from geography.metadata import iter_centres
from geography.enrichment import (prepare_resolve, enrich_inventory, compare_repairs,
                                  fingerprint, classified_regions, pending_output)


DEFAULT_INVENTORIES = {
    'ssl4eo': Path('pretrained_models/pretraining_locations/SSL4EO-S12/ssl4eo-s12_center_coords.csv'),
    'terramesh': Path('pretrained_models/pretraining_locations/TerraMesh/train_metadata.parquet'),
    'stac': Path('data/sen1floods11/catalog/sen1floods11_hand_labeled_label/collection.json'),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolve-source', type=Path, default=Path('data/environmental/resolve_2017/Ecoregions2017.shp'))
    parser.add_argument('--validated', type=Path, default=Path('data/environmental/resolve_2017_validated.parquet'))
    parser.add_argument('--dataset', choices=list(DEFAULT_INVENTORIES), nargs='+',
                        help='Inventories to process (default: all; stac is Sen1Floods11).')
    parser.add_argument('--inventory', action='append', metavar='KIND=PATH',
                        help='Override inventory paths. Without --dataset, process only these overrides.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--sample-records', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=65536)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args(argv)
    if args.sample_records <= 0 or args.batch_size <= 0:
        parser.error('Sample and batch sizes must be positive')
    overrides = {}
    for spec in args.inventory or []:
        kind, separator, path = spec.partition('=')
        if not separator or not path or kind not in DEFAULT_INVENTORIES:
            parser.error('--inventory must be ssl4eo=PATH, terramesh=PATH or stac=PATH')
        if kind in overrides:
            parser.error(f'Duplicate inventory override: {kind}')
        overrides[kind] = Path(path)
    selected = args.dataset or list(overrides) or list(DEFAULT_INVENTORIES)
    if len(selected) != len(set(selected)):
        parser.error('Each dataset can be selected only once')
    if set(overrides) - set(selected):
        parser.error('Inventory overrides must belong to the selected datasets')
    paths = DEFAULT_INVENTORIES | overrides
    args.inventory = [f'{kind}={paths[kind]}' for kind in selected]
    if args.output_dir is None:
        args.output_dir = Path('data/metadata/resolve_full' if args.full else 'data/metadata/resolve_validation')
    return args


def parquet_rows(path):
    with pq.ParquetFile(path) as parquet:
        return parquet.metadata.num_rows


def representative_centres(kind, path, size):
    """Bounded TerraMesh sampling across row groups; no full Parquet scan."""
    if kind == 'terramesh':
        with pq.ParquetFile(path) as parquet:
            group_count = parquet.num_row_groups
        # Row groups span the file's source ordering; sampling is not uniform globally.
        groups = np.unique(np.linspace(0, group_count-1, min(16,group_count), dtype=int))
        quota = max(1, int(np.ceil(size / len(groups))))
        pieces = []
        for group in groups:
            piece = next(iter_centres(kind, path, batch_size=quota, row_groups={int(group)}))
            pieces.append(piece)
        result = pd.concat(pieces, ignore_index=True)
        if len(result) > size:
            result = result.sample(size, random_state=42).sort_values('source_row').reset_index(drop=True)
        return result
    frames = list(iter_centres(kind, path))
    result = pd.concat(frames, ignore_index=True)
    if kind != 'stac' and len(result) > size:
        result = result.sample(size, random_state=42).sort_values('source_row').reset_index(drop=True)
    return result


def _describe(path, out):
    """Aggregate via disk-backed SQL; plot a bounded sample of centres."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use(['science', 'no-latex'])
    out.mkdir(parents=True, exist_ok=True)
    scope=json.loads(path.with_suffix('.provenance.json').read_text())['signature']['inputs']['scope']
    title=f'{path.stem} ({scope.replace("_", " ")})'
    centres = []
    with tempfile.TemporaryDirectory() as temp:
        with closing(sqlite3.connect(str(Path(temp)/'counts.sqlite'))) as db, pq.ParquetFile(path) as parquet, \
             tqdm(total=parquet.metadata.num_rows, desc=f'Aggregate {path.stem}', unit='records') as progress:
            for batch in parquet.iter_batches(batch_size=65536):
                frame = batch.to_pandas()
                frame[['center_lon','center_lat','biome_id','biome_name','biome_color',
                       'ecoregion_id','ecoregion_name','ecoregion_color','assignment_status']].to_sql(
                           'records', db, if_exists='append', index=False)
                # Bounded reservoir of plotting candidates, independent of category totals.
                centres.append(frame.sample(min(128,len(frame)), random_state=42))
                progress.update(len(frame))
            count = db.execute('select count(*) from records').fetchone()[0]
            statuses = dict(db.execute('select assignment_status,count(*) from records group by assignment_status'))
            summary = dict(records=count, assignment_status_counts=statuses,
                           uniquely_classified_records=sum(statuses.get(k,0) for k in ['classified','boundary']))
            tables = {}
            for category in ['biome','ecoregion']:
                valid = "assignment_status != 'unclassified' and " + category + '_id is not null'
                table = pd.read_sql_query(f'''select {category}_id, {category}_name, {category}_color,
                    count(*) as records from records where {valid}
                    group by {category}_id, {category}_name, {category}_color order by records desc''', db)
                unique = pd.read_sql_query(f'''select {category}_id,count(*) as unique_centres from
                    (select distinct {category}_id,center_lon,center_lat from records where {valid})
                    group by {category}_id''', db)
                table = table.merge(unique, on=category+'_id')
                table['fraction_all_records'] = table.records/count
                table.to_csv(out/f'{category}_counts.csv', index=False)
                tables[category] = table
                summary['represented_'+category+'s'] = len(table)
                shown=table.head(20).iloc[::-1]
                fig,ax=plt.subplots(figsize=(10,max(3,.3*len(shown))))
                ax.barh(shown[category+'_name'], shown.fraction_all_records, color=shown[category+'_color'])
                ax.set(xlabel='Fraction of all inventory records', title=f'{title}: {category} composition'+(' (top 20)' if len(table)>20 else ''))
                fig.tight_layout()
                fig.savefig(out/f'{category}_distribution.pdf',bbox_inches='tight')
                plt.close(fig)
    points=pd.concat(centres,ignore_index=True)
    if len(points)>10000:
        points=points.sample(10000,random_state=42)
    finite=np.isfinite(points.center_lon)&np.isfinite(points.center_lat)
    colours=points.biome_color.fillna('#999999').copy()
    colours.loc[points.assignment_status == 'unclassified'] = '#999999'
    fig,ax=plt.subplots(figsize=(10,5))
    ax.scatter(points.loc[finite,'center_lon'],points.loc[finite,'center_lat'],s=3,c=colours[finite],alpha=.6,rasterized=True)
    ax.set(xlim=(-180,180),ylim=(-90,90),xlabel='Longitude',ylabel='Latitude',
           title=f'{title}: sampled centres ({len(points):,}); colour denotes biome')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'centres.pdf');plt.close(fig)
    summary['plot_records']=len(points)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return tables['biome']


def describe(path, out):
    """Stage reports so interruption during aggregation preserves completed reports."""
    path, out = Path(path), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.describe-', dir=out.parent) as temporary:
        staging = Path(temporary)
        try:
            result = _describe(path, staging)
            out.mkdir(parents=True, exist_ok=True)
            # The summary is the completion marker and is published last.
            (out / 'summary.json').unlink(missing_ok=True)
            for file in sorted(staging.iterdir(), key=lambda p: p.name == 'summary.json'):
                file.replace(out / file.name)
            return result
        finally:
            import matplotlib.pyplot as plt
            plt.close('all')


def main():
    args = parse_args()
    start=time.perf_counter()
    regions=prepare_resolve(args.resolve_source,args.validated)
    original=gpd.read_file(args.resolve_source)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    all_biomes=[];report={}
    for spec in args.inventory:
        kind,source=spec.split('=',1);source=Path(source)
        print(f'Validating {kind}: {source}',flush=True)
        sample=representative_centres(kind,source,args.sample_records)
        comparison=compare_repairs(sample,original,regions)
        with pending_output(args.output_dir/f'{kind}_repair_comparison.parquet') as pending:
            comparison.to_parquet(pending,compression='zstd',index=False)
        with pending_output(args.output_dir/f'{kind}_assignment_discrepancies.csv') as pending:
            comparison[comparison.comparison_status!='unchanged'].to_csv(pending,index=False)
        invalid_tree=shapely.STRtree(original.loc[~original.is_valid].geometry)
        points=shapely.points(sample.center_lon,sample.center_lat)
        candidate_count=len(np.unique(invalid_tree.query(points)[0]))
        print(f'Checking source fingerprints for {kind}...', flush=True)
        provenance=dict(source=fingerprint(source),kind=kind,
            centre_method='existing inventory coordinates; STAC explicit centre or planar WGS84 footprint centroid',
            scope='full' if args.full or kind=='stac' else 'validation_subset',
            sample_records=args.sample_records if not args.full else None,
            sampling='all STAC items; SSL4EO fixed-seed random sample; TerraMesh evenly selected row-group prefixes',
            record_identity='source fingerprint plus source_row; original sample_id and tar/source_item retained')
        output=args.output_dir/f'{kind}_enriched.parquet'
        info=enrich_inventory(iter_centres(kind,source,args.batch_size) if args.full else
                              (sample.iloc[i:i+args.batch_size] for i in range(0,len(sample),args.batch_size)),
                              args.validated,output,provenance, total_records=(len(sample) if not args.full else
                                  parquet_rows(source) if kind == 'terramesh' else None))
        reloaded=pd.read_parquet(output) if not args.full else None
        if reloaded is not None:
            pd.testing.assert_frame_equal(reloaded[list(sample.columns)],sample.reset_index(drop=True),
                                          check_dtype=False, check_exact=True)
            assert reloaded.center_lon.dtype == np.dtype('float64')
            assert reloaded.center_lat.dtype == np.dtype('float64')
        info['comparison_counts']=comparison.comparison_status.value_counts().to_dict()
        info['comparison_records']=len(sample)
        info['sample_id_duplicates_in_subset']=int(sample.sample_id.duplicated().sum())
        info['repair_bbox_candidate_records']=candidate_count
        info['source_counts_in_subset']=sample.source.value_counts().to_dict()
        report[kind]=info
        table=describe(output,args.output_dir/f'{kind}_figures');table['inventory']=kind;all_biomes.append(table)
        print(kind,info['comparison_counts'],info['assignment_status_counts'],flush=True)
    import matplotlib.pyplot as plt
    combined=pd.concat(all_biomes,ignore_index=True)
    distribution=combined.pivot(index='biome_name',columns='inventory',values='fraction_all_records').fillna(0)
    fig,ax=plt.subplots(figsize=(10,max(4,.4*len(distribution))))
    distribution.plot.barh(ax=ax);ax.set(xlabel='Fraction of all records',ylabel='',title='Centre-based biome composition (subset scope where indicated)')
    fig.tight_layout()
    with pending_output(args.output_dir/'biome_comparison.pdf') as pending:
        fig.savefig(pending,bbox_inches='tight')
    plt.close(fig)
    report['elapsed_seconds']=time.perf_counter()-start
    try:
        import psutil
        report['peak_working_set_bytes']=psutil.Process().memory_info().peak_wset
    except (ImportError,AttributeError):
        report['peak_working_set_bytes']=None
    with pending_output(args.output_dir/'validation_summary.json') as pending:
        pending.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Completed in {report["elapsed_seconds"]:.1f}s',flush=True)


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted. Completed outputs are retained; rerun the same command to reuse valid caches.', file=sys.stderr)
        sys.exit(130)
