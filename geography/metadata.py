"""Inventory adapters independent of training and evaluation pipelines."""
from pathlib import Path
import csv
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform

def load_metadata(kind, path, batch_size=65536):
    """Yield common-schema tables; geometry decoding is deferred to candidates.

    Fields: sample_id, source, center_lon/lat, geometry, split, plus original
    provenance fields. Add future formats here, without dataset classes.
    """
    path = Path(path)
    if kind == "ssl4eo":
        for frame in pd.read_csv(path, header=None, names=["sample_id", "center_lon", "center_lat"],
                                 dtype={"sample_id": str}, chunksize=batch_size):
            frame["source"] = "croma_ssl4eo"
            frame["split"] = "inventory_unspecified"
            frame["geometry"] = None
            yield frame
    elif kind == "terramesh":
        parquet = pq.ParquetFile(path)
        columns = ["sample_id", "tar", "split", "center_lon", "center_lat", "geometry",
                   "crs", "bounds", "S1_time", "S2_time", "S1_product_id", "S2L1C_product_id"]
        missing = set(columns[:9]) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        for batch in parquet.iter_batches(batch_size=batch_size,
                                         columns=[c for c in columns if c in parquet.schema_arrow.names]):
            frame = batch.to_pandas()
            if not frame["split"].eq("train").all():
                raise ValueError("Exposure inventory must contain TerraMesh train rows only")
            frame["source"] = frame.tar.map(lambda s: "terramind_ssl4eo" if s.startswith("ssl4eos12")
                                           else "terramind_majortom" if s.startswith("majortom") else None)
            if frame.source.isna().any():
                raise ValueError("Unknown TerraMesh source prefix")
            yield frame
    elif kind == "stac":
        rows = []
        visited = set()
        for link in json.loads(path.read_text())["links"]:
            if link["rel"] != "item":
                continue
            item_path = (path.parent / link["href"]).resolve()
            if item_path in visited:
                continue
            visited.add(item_path)
            item = json.loads(item_path.read_text())
            geom = shape(item["geometry"])
            if not geom.is_valid or geom.is_empty:
                raise ValueError(f"Invalid STAC geometry: {item['id']}")
            rows.append(dict(sample_id=item["id"].removesuffix("_label"), source="sen1floods11",
                             center_lon=geom.centroid.x, center_lat=geom.centroid.y,
                             geometry=geom, split=None))
        yield pd.DataFrame(rows)
    else:
        raise ValueError(f"Unknown metadata format: {kind}")


def load_downstream(catalog, splits, raster_dir):
    from .exposure import projection, metric_polygon

    frame = next(load_metadata("stac", catalog))
    if frame.sample_id.duplicated().any():
        raise ValueError("Duplicate downstream IDs")
    membership = {}
    for split, path in splits.items():
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.reader(stream):
                if not row:
                    continue
                chip = Path(row[1].replace("\\", "/")).name.removesuffix("_LabelHand.tif")
                if chip in membership:
                    raise ValueError(f"Repeated chip within/across splits: {chip}")
                membership[chip] = split
    if set(membership) - set(frame.sample_id):
        raise ValueError("Split IDs missing from STAC catalogue")
    frame["split"] = frame.sample_id.map(membership)
    frame = frame[frame.split.notna()].copy()
    checks = []
    # Compare one actual raster per split, not just nominal catalogue dimensions.
    for _, group in frame.groupby("split"):
        row = group.iloc[0]
        with rasterio.open(Path(raster_dir) / f"{row.sample_id}_LabelHand.tif") as raster:
            corners = [raster.transform * xy for xy in
                       [(0, 0), (raster.width, 0), (raster.width, raster.height), (0, raster.height)]]
            from shapely.geometry import Polygon
            actual = transform(Transformer.from_crs(raster.crs, 4326, always_xy=True).transform,
                               Polygon(corners))
        project = projection(row.center_lon, row.center_lat)
        a, b = metric_polygon(row.geometry, project), metric_polygon(actual, project)
        difference = a.symmetric_difference(b).area / b.area
        checks.append({"chip": row.sample_id, "split": row.split, "relative_area_difference": difference})
        if difference > 0.001:
            raise ValueError(f"STAC/raster mismatch: {checks[-1]}")
    return frame, checks




def iter_centres(kind, path, batch_size=65536, row_groups=None):
    """Yield geometry-free records with source-row identity and WGS84 centres.

    STAC centres use the footprint's planar longitude/latitude centroid when no
    explicit centre exists. This preserves the historical exposure convention.
    TerraMesh row groups may be selected for bounded validation only.
    """
    path = Path(path)
    offset = 0
    if kind == 'ssl4eo':
        for frame in pd.read_csv(path, header=None, names=['sample_id', 'center_lon', 'center_lat'],
                                 dtype={'sample_id': str}, chunksize=batch_size):
            frame['source'] = 'ssl4eo_s12_original'
            frame['source_row'] = np.arange(offset, offset + len(frame), dtype=np.int64)
            offset += len(frame)
            yield frame
    elif kind == 'terramesh':
        parquet = pq.ParquetFile(path)
        columns = ['sample_id', 'tar', 'split', 'center_lon', 'center_lat']
        if 'zarr' in parquet.schema_arrow.names:
            columns.append('zarr')
        if set(columns) - set(parquet.schema_arrow.names):
            raise ValueError('TerraMesh requires sample_id, tar, split and centre coordinates')
        for group in range(parquet.num_row_groups):
            length = parquet.metadata.row_group(group).num_rows
            if row_groups is not None and group not in row_groups:
                offset += length
                continue
            for batch in parquet.iter_batches(batch_size=batch_size, row_groups=[group], columns=columns):
                frame = batch.to_pandas()
                frame['sample_id'] = frame.sample_id.astype(str)
                frame['source'] = frame.tar.str.split('_shard_').str[0]
                frame['source_row'] = np.arange(offset, offset + len(frame), dtype=np.int64)
                offset += len(frame)
                yield frame
    elif kind == 'stac':
        seen = set()
        rows = []
        def items(catalog):
            catalog = catalog.resolve()
            if catalog in seen:
                return
            seen.add(catalog)
            document = json.loads(catalog.read_text(encoding='utf-8'))
            if document.get('type') == 'Feature':
                yield catalog, document
            else:
                for link in document.get('links', []):
                    if link['rel'] in {'item', 'child'}:
                        yield from items(catalog.parent / link['href'])
        for item_path, item in items(path):
            props = item.get('properties', {})
            if 'center_lon' in props and 'center_lat' in props:
                lon, lat = props['center_lon'], props['center_lat']
            else:
                geometry = shape(item['geometry']) if item.get('geometry') else None
                if (geometry is None or geometry.is_empty or not geometry.is_valid
                        or geometry.bounds[2] - geometry.bounds[0] > 180):
                    lon, lat = np.nan, np.nan
                else:
                    lon, lat = geometry.centroid.x, geometry.centroid.y
            rows.append(dict(sample_id=str(item['id']), center_lon=lon, center_lat=lat,
                             source='stac', source_item=str(item_path), source_row=offset))
            offset += 1
            if len(rows) == batch_size:
                yield pd.DataFrame(rows)
                rows = []
        if rows:
            yield pd.DataFrame(rows)
    else:
        raise ValueError(f'Unsupported centre metadata format: {kind}')
