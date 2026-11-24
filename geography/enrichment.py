"""RESOLVE preparation and centre-based ecological assignment in WGS84."""
from pathlib import Path
from contextlib import contextmanager
from tqdm.auto import tqdm
from collections import Counter
import hashlib
import json
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Geod
import shapely
from shapely.geometry import Point
from shapely.geometry.polygon import orient

SCHEMA_VERSION = 1
STATUSES = ['classified', 'boundary', 'shared_boundary', 'multiple_matches',
            'unclassified', 'unmatched', 'invalid_coordinates', 'predicate_error']
GEOD = Geod(ellps='WGS84')
FIELDS = {'biome_id': 'BIOME_NUM', 'biome_name': 'BIOME_NAME', 'biome_color': 'COLOR_BIO',
          'ecoregion_id': 'ECO_ID', 'ecoregion_name': 'ECO_NAME',
          'ecoregion_color': 'COLOR', 'realm': 'REALM'}
METHOD = 'make_valid linework; invalid only; recursive polygon extraction; no simplification or reprojection'


@contextmanager
def pending_output(output):
    """Publish a complete file atomically; remove staging files on interruption."""
    output = Path(output)
    temporary = output.with_name(output.stem + '.pending' + output.suffix)
    try:
        yield temporary
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(path):
    """Hash source contents, including shapefile sidecars or linked STAC documents."""
    path = Path(path)
    files = [path]
    if path.suffix.lower() == '.shp':
        files = sorted(p for p in path.parent.glob(path.stem + '.*') if p.suffix.lower() in
                       {'.shp', '.shx', '.dbf', '.prj', '.cpg'})
    elif path.suffix.lower() == '.json':
        files, pending, seen = [], [path.resolve()], set()
        while pending:
            p = pending.pop()
            if p in seen:
                continue
            seen.add(p)
            files.append(p)
            doc = json.loads(p.read_text(encoding='utf-8'))
            pending.extend((p.parent / link['href']).resolve() for link in doc.get('links', [])
                           if link['rel'] in {'item', 'child'})
        files.sort()
    result = {}
    for file in files:
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(block)
        result[str(file.resolve())] = digest.hexdigest()
    return result


def polygon_parts(geometry):
    if geometry.geom_type == 'Polygon':
        return [geometry]
    return [polygon for part in getattr(geometry, 'geoms', []) for polygon in polygon_parts(part)]


def geodetic_area(geometry):
    # Invalid originals provide an algebraic reference, not a true area baseline.
    return sum(abs(GEOD.geometry_area_perimeter(orient(p, sign=1))[0]) for p in polygon_parts(geometry))


def classified_regions(regions):
    names = regions.BIOME_NAME.astype('string').fillna('').str.strip().str.upper()
    return regions.BIOME_NUM.between(1, 14) & ~names.isin(['', 'N/A', 'NA', 'NONE'])


def validate_regions(regions):
    if regions.crs is None or regions.crs.to_epsg() != 4326:
        raise ValueError('RESOLVE assignment requires WGS84 geographical coordinates (EPSG:4326)')
    if set(FIELDS.values()) - set(regions.columns):
        raise ValueError('Missing RESOLVE ecological attributes')
    if regions.ECO_ID.isna().any() or regions.ECO_ID.duplicated().any():
        raise ValueError('Ecological IDs must be present and unique')
    if (regions.geometry.isna().any() or regions.geometry.is_empty.any()
            or not regions.geometry.is_valid.all()
            or not regions.geom_type.isin(['Polygon', 'MultiPolygon']).all()):
        raise ValueError('RESOLVE geometries must be valid, nonempty polygons')
    bounds = regions.total_bounds
    if not np.isfinite(bounds).all() or bounds[0] < -180.000001 or bounds[2] > 180.000001 or bounds[1] < -90.000001 or bounds[3] > 90.000001:
        raise ValueError('Invalid geographical bounds')
    for _, row in regions.iterrows():
        for polygon in polygon_parts(row.geometry):
            for ring in [polygon.exterior, *polygon.interiors]:
                if np.any(np.abs(np.diff(np.asarray(ring.coords)[:, 0])) > 180):
                    raise ValueError(f'Unwrapped antimeridian edge requires review: ECO_ID={row.ECO_ID}')
    classified = regions[classified_regions(regions)]
    for _, group in classified.groupby('BIOME_NUM'):
        if any(group[column].nunique(dropna=False) != 1 for column in ['BIOME_NAME', 'COLOR_BIO']):
            raise ValueError('Inconsistent biome names or colours')


def prepare_resolve(source, output):
    """Persist a validated copy and repair audit; never overwrite source files."""
    source, output = Path(source), Path(output)
    if output.resolve() == source.resolve() or output.suffix != '.parquet':
        raise ValueError('Use a separate .parquet output')
    manifest_path = output.with_suffix('.provenance.json')
    signature = dict(source=fingerprint(source), method=METHOD, schema_version=SCHEMA_VERSION,
                     shapely=shapely.__version__, geos=shapely.geos_version_string)
    if output.exists() and manifest_path.exists() and output.with_suffix('.repair.csv').exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get('signature') == signature and previous.get('output_fingerprint') == fingerprint(output):
            regions = gpd.read_parquet(output)
            validate_regions(regions)
            return regions
    started = time.perf_counter()
    original = gpd.read_file(source)
    if original.crs is None or original.crs.to_epsg() != 4326:
        raise ValueError('Expected original RESOLVE geographical CRS EPSG:4326')
    regions = original.copy()
    invalid = ~original.geometry.is_valid
    audit = []
    for index, row in original[invalid].iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            raise ValueError('Missing source geometry requires manual review')
        raw = shapely.make_valid(geom, method='linework')
        parts = polygon_parts(raw)
        if not parts:
            raise ValueError(f'Repair produced no polygon: {row.ECO_ID}')
        repaired = raw if raw.geom_type in {'Polygon', 'MultiPolygon'} else shapely.union_all(parts)
        before, after = geodetic_area(geom), geodetic_area(repaired)
        delta = float(np.max(np.abs(np.array(geom.bounds) - repaired.bounds)))
        change = abs(after - before) / max(before, 1)
        audit.append(dict(ECO_ID=row.ECO_ID, ECO_NAME=row.ECO_NAME,
                          reason=shapely.is_valid_reason(geom), original_type=geom.geom_type,
                          raw_type=raw.geom_type, repaired_type=repaired.geom_type,
                          original_parts=len(polygon_parts(geom)), repaired_parts=len(polygon_parts(repaired)),
                          nonpolygon_components_removed=raw.geom_type == 'GeometryCollection' and
                          any(part.geom_type not in {'Polygon', 'MultiPolygon'} for part in raw.geoms),
                          original_algebraic_area_m2=before, repaired_geodetic_area_m2=after,
                          relative_area_change=change, bounds_delta_degrees=delta,
                          valid_after=repaired.is_valid, review_required=change > 0.0001 or delta > 1e-8))
        regions.at[index, 'geometry'] = repaired
        if len(audit) % 10 == 0:
            print(f'Repaired {len(audit)}/{int(invalid.sum())} invalid regions', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pending_output(output.with_suffix('.repair.csv')) as pending:
        pd.DataFrame(audit, columns=list(audit[0]) if audit else ['ECO_ID', 'review_required']).to_csv(
            pending, index=False)
    validate_regions(regions)
    pd.testing.assert_frame_equal(original.drop(columns='geometry'), regions.drop(columns='geometry'))
    assert (original.loc[~invalid].geometry.to_wkb() == regions.loc[~invalid].geometry.to_wkb()).all()
    if any(row['review_required'] for row in audit):
        raise ValueError('Repair changes exceed review thresholds; inspect the repair audit before assignment')
    with pending_output(output) as temporary:
        regions.to_parquet(temporary, compression='zstd', index=False)
        reloaded = gpd.read_parquet(temporary)
        validate_regions(reloaded)
        pd.testing.assert_frame_equal(regions.drop(columns='geometry').reset_index(drop=True), reloaded.drop(columns='geometry'))
        assert list(regions.geometry.to_wkb()) == list(reloaded.geometry.to_wkb())
    assert signature['source'] == fingerprint(source), 'Original RESOLVE files changed'
    with pending_output(manifest_path) as pending:
        pending.write_text(json.dumps(dict(signature=signature, records=len(regions),
            repaired=int(invalid.sum()), seconds=time.perf_counter()-started, bytes=output.stat().st_size,
            output_fingerprint=fingerprint(output),
            limitations=['Invalid-original areas are algebraic reference values only.',
                         'Planar WGS84 predicates; unwrapped antimeridian edges rejected.',
                         'Classification describes the centre, not the entire footprint.']), indent=2), encoding='utf-8')
    return regions


def _assign(frame, regions, tree, tolerate_errors=False):
    result = frame.copy().reset_index(drop=True)
    if 'geometry' in result.columns:
        raise ValueError('Enrichment inputs must not contain footprint geometries')
    for key in FIELDS:
        result[key] = pd.Series(pd.NA, index=result.index, dtype='Int64' if key.endswith('_id') else 'string')
    result['assignment_status'] = 'invalid_coordinates'
    result['matched_ecoregion_ids'] = '[]'
    lon = pd.to_numeric(result.center_lon, errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    lat = pd.to_numeric(result.center_lat, errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(lon) & np.isfinite(lat) & (abs(lon) <= 180) & (abs(lat) <= 90)
    positions = np.flatnonzero(valid)
    points = shapely.points(lon[valid], lat[valid])
    candidates = tree.query(points)
    geometries = regions.geometry.to_numpy()
    classified = classified_regions(regions).to_numpy()
    attributes = {
        key: pd.array([pd.NA if pd.isna(value) else int(value) if key.endswith('_id') else str(value)
                       for value in regions[column]],
                      dtype='Int64' if key.endswith('_id') else 'string')
        for key, column in FIELDS.items()
    }
    single_ids = np.array([json.dumps([int(value)]) for value in regions.ECO_ID], dtype=object)
    failed = np.zeros(len(points), dtype=bool)
    if not tolerate_errors:
        hits = shapely.covers(geometries[candidates[1]], points[candidates[0]])
    else:
        # Invalid-original diagnostics isolate predicate failures per candidate.
        hits = np.zeros(candidates.shape[1], dtype=bool)
        for k, (i, j) in enumerate(candidates.T):
            try:
                hits[k] = geometries[j].covers(points[i])
            except shapely.errors.GEOSException:
                failed[i] = True
    point_ids, region_ids = candidates[:, hits]
    keep = ~failed[point_ids]
    point_ids, region_ids = point_ids[keep], region_ids[keep]
    counts = np.bincount(point_ids, minlength=len(points))
    result.loc[positions, 'assignment_status'] = 'unmatched'
    result.loc[positions[failed], 'assignment_status'] = 'predicate_error'

    unique = counts[point_ids] == 1
    pi, ri = point_ids[unique], region_ids[unique]
    destination = positions[pi]
    result.loc[destination, 'matched_ecoregion_ids'] = single_ids[ri]
    try:
        boundary = shapely.touches(geometries[ri], points[pi])
        boundary_failed = np.zeros(len(pi), dtype=bool)
    except shapely.errors.GEOSException:
        if not tolerate_errors:
            raise
        boundary = np.zeros(len(pi), dtype=bool)
        boundary_failed = np.zeros(len(pi), dtype=bool)
        for k, (i, j) in enumerate(zip(pi, ri)):
            try:
                boundary[k] = geometries[j].touches(points[i])
            except shapely.errors.GEOSException:
                boundary_failed[k] = True
    statuses = np.where(~classified[ri], 'unclassified', np.where(boundary, 'boundary', 'classified')).astype(object)
    statuses[boundary_failed] = 'predicate_error'
    result.loc[destination, 'assignment_status'] = statuses
    for key, values in attributes.items():
        result.loc[destination[~boundary_failed], key] = values.take(ri[~boundary_failed])

    # Only ambiguous matches need group-level decisions and sorted ID lists.
    multiple = counts[point_ids] > 1
    mpi, mri = point_ids[multiple], region_ids[multiple]
    order = np.argsort(mpi, kind='stable')
    mpi, mri = mpi[order], mri[order]
    starts = np.r_[0, np.flatnonzero(np.diff(mpi)) + 1, len(mpi)]
    for start, end in zip(starts[:-1], starts[1:]):
        if start == end:
            continue
        i = mpi[start]
        position = positions[i]
        ids = mri[start:end]
        group = regions.iloc[ids]
        result.at[position, 'matched_ecoregion_ids'] = json.dumps(sorted(int(v) for v in group.ECO_ID))
        try:
            boundary = shapely.touches(geometries[ids], points[i])
        except shapely.errors.GEOSException:
            if not tolerate_errors:
                raise
            result.at[position, 'assignment_status'] = 'predicate_error'
            continue
        status = 'shared_boundary' if boundary.all() else 'multiple_matches'
        if classified[ids].all() and group.BIOME_NUM.nunique() == 1:
            for key in ['biome_id', 'biome_name', 'biome_color']:
                result.at[position, key] = attributes[key][ids[0]]
        if group.REALM.nunique(dropna=False) == 1 and pd.notna(group.REALM.iloc[0]):
            result.at[position, 'realm'] = attributes['realm'][ids[0]]
        result.at[position, 'assignment_status'] = status
    result['assignment_status'] = pd.Categorical(result.assignment_status, categories=STATUSES)
    return result


def assign_centres(frame, regions):
    """Assign with covers; retain all boundary matches and never use nearest regions."""
    validate_regions(regions)
    shapely.prepare(regions.geometry.to_numpy())
    return _assign(frame, regions, shapely.STRtree(regions.geometry))


def compare_repairs(frame, original, repaired):
    """Compare successful original predicates; invalid originals are diagnostic only."""
    validate_regions(repaired)
    shapely.prepare(original.geometry.to_numpy())
    shapely.prepare(repaired.geometry.to_numpy())
    before = _assign(frame, original, shapely.STRtree(original.geometry), tolerate_errors=True)
    after = _assign(frame, repaired, shapely.STRtree(repaired.geometry))
    compared = ['assignment_status', 'matched_ecoregion_ids', *FIELDS]
    details = frame.copy().reset_index(drop=True)
    for column in compared:
        details['before_' + column] = before[column]
        details['after_' + column] = after[column]
    success = before.assignment_status != 'predicate_error'
    equal = pd.Series(True, index=before.index)
    for column in compared:
        equal &= before[column].astype('string').fillna('<null>') == after[column].astype('string').fillna('<null>')
    details['comparison_status'] = np.where(~success, 'original_predicate_error', np.where(equal, 'unchanged', 'changed'))
    return details


def enrich_inventory(batches, regions_path, output, provenance, *, total_records=None):
    """Incrementally persist geometry-free batches; input record order is retained."""
    output, regions_path = Path(output), Path(regions_path)
    if output.resolve() == regions_path.resolve() or str(output.resolve()) in provenance.get('source', {}):
        raise ValueError('Output must differ from all input files')
    manifest = output.with_suffix('.provenance.json')
    signature = dict(schema_version=SCHEMA_VERSION, inputs=provenance,
                     resolve=fingerprint(regions_path), predicate='covers; touches distinguishes boundaries',
                     shapely=shapely.__version__)
    if output.exists() and manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous.get('signature') == signature and previous.get('output_fingerprint') == fingerprint(output):
            print(f'Using cached enrichment: {output}', flush=True)
            return previous
    regions = gpd.read_parquet(regions_path)
    validate_regions(regions)
    shapely.prepare(regions.geometry.to_numpy())
    tree = shapely.STRtree(regions.geometry)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pending_output(output) as temporary, tqdm(total=total_records, desc=f"Enrich {output.stem}", unit="records") as progress:
        writer, count, statuses = None, 0, Counter()
        last_source_row = -1
        started = time.perf_counter()
        try:
            for frame in batches:
                if frame.empty:
                    continue
                if 'source_row' in frame:
                    indices = frame.source_row.to_numpy()
                    if indices[0] <= last_source_row or np.any(np.diff(indices) <= 0):
                        raise ValueError('Source rows must be unique and increasing across batches')
                    last_source_row = int(indices[-1])
                assigned = _assign(frame, regions, tree)
                pd.testing.assert_frame_equal(assigned[list(frame.columns)], frame.reset_index(drop=True))
                statuses.update(assigned.assignment_status.astype(str))
                # Fixed categories keep the Arrow schema consistent across batches.
                table = pa.Table.from_pandas(assigned, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression='zstd', use_dictionary=True)
                writer.write_table(table.cast(writer.schema))
                count += len(frame)
                progress.update(len(frame))
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise ValueError('Empty inventory')
        with pq.ParquetFile(temporary) as parquet:
            assert parquet.metadata.num_rows == count
    info = dict(signature=signature, geometry_repair_method=METHOD,
                resolve_provenance=str(regions_path.with_suffix('.provenance.json')),
                input_records=count, output_records=count,
                assignment_status_counts=dict(statuses), seconds=time.perf_counter()-started,
                bytes=output.stat().st_size, output_fingerprint=fingerprint(output))
    with pending_output(manifest) as pending:
        pending.write_text(json.dumps(info, indent=2), encoding='utf-8')
    return info
