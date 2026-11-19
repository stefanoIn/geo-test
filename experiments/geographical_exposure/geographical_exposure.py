"""Standalone geographical exposure audit (no model inference).

From GeoTest: python experiments/geographical_exposure/geographical_exposure.py --chip Ghana_313799
Omit --chip to analyse all split members; --map-chip chooses the one local map.
Run --self-test for small synthetic checks. Paths are relative to the current directory.

Distances use a chip-centred WGS84 azimuthal-equidistant projection, with
densified edges. They are metric footprint distances, not degree distances or
mathematically exact ellipsoidal boundary minima. Candidate retrieval is global,
not truncated at the overlap/proximity radius. The output records this distinction.
Nearby counts are centre-to-centre haversine counts, not footprint-buffer counts.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from pyproj import Geod, Transformer
from sklearn.neighbors import BallTree
from shapely import from_wkb, segmentize
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union

R = 6371008.8
GEOD = Geod(ellps="WGS84")
COLORS = {"croma_ssl4eo": "#0072B2", "terramind_ssl4eo": "#009E73",
          "terramind_majortom": "#D55E00"}
LABELS = {"croma_ssl4eo": "CROMA — SSL4EO original",
          "terramind_ssl4eo": "TerraMind — SSL4EO v1.1",
          "terramind_majortom": "TerraMind — MajorTOM", "terramind": "TerraMind union"}


def projection(lon, lat):
    crs = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m"
    return Transformer.from_crs(4326, crs, always_xy=True).transform


def metric_polygon(geometry, project):
    # Dense geographic edges retain the footprint shape when projected.
    return transform(project, segmentize(geometry, 0.001))


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


def footprint(row):
    if row.source != "croma_ssl4eo":
        geom = from_wkb(row.geometry)
    else:
        lon, lat = row.center_lon, row.center_lat
        if not -80 <= lat <= 84:
            raise ValueError("SSL4EO centre outside supported UTM latitude range")
        epsg = (32600 if lat >= 0 else 32700) + min(60, int((lon + 180) // 6) + 1)
        forward = Transformer.from_crs(4326, epsg, always_xy=True)
        reverse = Transformer.from_crs(epsg, 4326, always_xy=True)
        x, y = forward.transform(lon, lat)
        geom = transform(reverse.transform, segmentize(box(x-1320, y-1320, x+1320, y+1320), 100))
    if geom.is_empty or not geom.is_valid:
        raise ValueError(f"Invalid footprint: {row.sample_id}")
    if geom.bounds[2] - geom.bounds[0] > 180:
        raise ValueError("Candidate crosses antimeridian; needs longitude unwrapping")
    return geom


def enclosing_radii(frame):
    """Conservative geodesic enclosing radii without decoding millions of WKBs.

    UTM has minimum scale .9996 in its valid zone. Rectangle half-diagonal
    divided by this scale bounds the ground distance from its midpoint.
    Add the geodesic offset from catalogue centre to that midpoint.
    Bounds are trusted source metadata, checked against decoded candidates.
    """
    if frame.source.eq("croma_ssl4eo").all():
        return np.full(len(frame), np.sqrt(2) * 1320 / .9996 * 1.02)
    bounds = np.stack(frame.bounds)
    if not np.isfinite(bounds).all() or np.any(bounds[:, 2:] < bounds[:, :2]):
        raise ValueError("Invalid projected bounds")
    radius = np.hypot(bounds[:, 2]-bounds[:, 0], bounds[:, 3]-bounds[:, 1]) / (2*.9996)
    for epsg in frame.crs.unique():
        if not (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760):
            raise ValueError(f"Unsupported bounds CRS {epsg}; add an explicit radius calculation")
        ix = np.flatnonzero(frame.crs.to_numpy() == epsg)
        mid = (bounds[ix, :2]+bounds[ix, 2:])/2
        lon, lat = Transformer.from_crs(int(epsg), 4326, always_xy=True).transform(mid[:, 0], mid[:, 1])
        radius[ix] += GEOD.inv(frame.center_lon.to_numpy()[ix], frame.center_lat.to_numpy()[ix], lon, lat)[2]
    return radius * 1.02


def coverage(target, polygons):
    if not polygons:
        return 0.0
    return min(1.0, target.intersection(unary_union(polygons)).area / target.area)


def analyse(downstream, inventory, radii, map_chip, view_km):
    states, maps, counts = {}, {}, {}
    for chip in downstream.itertuples():
        project = projection(chip.center_lon, chip.center_lat)
        target = metric_polygon(chip.geometry, project)
        # Path-length bound along lon/lat enclosing rectangle also covers edges.
        b = chip.geometry.bounds
        bound = R*np.radians(max(abs(b[0]-chip.center_lon), abs(b[2]-chip.center_lon)) +
                             max(abs(b[1]-chip.center_lat), abs(b[3]-chip.center_lat))) * 1.02
        states[chip.sample_id] = dict(chip=chip, project=project, target=target, radius=bound, sources={})
    processed = 0
    for frame in inventory:
        if not np.isfinite(frame[["center_lon", "center_lat"]].to_numpy()).all():
            raise ValueError("Non-finite centre")
        rr = enclosing_radii(frame)
        for source in frame.source.unique():
            indices = np.flatnonzero(frame.source.to_numpy() == source)
            part = frame.iloc[indices].reset_index(drop=True)
            sizes = rr[indices]
            coords = np.radians(part[["center_lat", "center_lon"]].to_numpy())
            tree = BallTree(coords, metric="haversine")
            max_size = sizes.max()
            counts[source] = counts.get(source, 0) + len(part)
            decoded = {}
            for key, state in states.items():
                chip, project, target = state["chip"], state["project"], state["target"]
                stats = state["sources"].setdefault(source, dict(best=float("inf"), nearest=None,
                      overlaps=[], overlap_records=0, sites=set(), record_counts=np.zeros(len(radii), dtype=int),
                      nearby_sites=[set() for _ in radii]))
                point = np.radians([[chip.center_lat, chip.center_lon]])

                def evaluate(j):
                    j = int(j)
                    row = part.iloc[j]
                    if j not in decoded:
                        decoded[j] = footprint(row)
                    geom = decoded[j]
                    # Catch bounds/geometry mismatch before accepting the radius guarantee.
                    xy = np.asarray(segmentize(geom, .001).exterior.coords)
                    distances = GEOD.inv(np.full(len(xy), row.center_lon), np.full(len(xy), row.center_lat),
                                         xy[:, 0], xy[:, 1])[2]
                    if distances.max() > sizes[j]:
                        raise ValueError(f"Footprint exceeds metadata-derived radius: {row.sample_id}")
                    projected = metric_polygon(geom, project)
                    distance = target.distance(projected)
                    if distance < stats["best"]:
                        stats["best"], stats["nearest"] = distance, row
                    return row, geom, projected

                # Seed the nearest upper bound for this source, then search all
                # candidates that could beat it. This works across batch boundaries.
                seed = int(tree.query(point, k=1)[1][0, 0])
                evaluate(seed)
                # Mean-radius spherical distances differ from WGS84 by <1%;
                # use a conservative 2% reduction for pruning, never for results.
                reach = (stats["best"] + state["radius"] + max_size) / .98
                reach = max(reach, (state["radius"] + max_size)/.98,
                            max(radii)*1000, (view_km*1000+max_size)/.98 if key == map_chip else 0)
                ids, dd = tree.query_radius(point, r=min(np.pi, reach/R), return_distance=True)
                for j, d in zip(ids[0], dd[0]*R):
                    row = part.iloc[int(j)]
                    site = (float(row.center_lon), float(row.center_lat))
                    for n, radius in enumerate(radii):
                        if d <= radius*1000:
                            stats["record_counts"][n] += 1
                            stats["nearby_sites"][n].add(site)
                    lower = .98*d - state["radius"] - sizes[j]
                    overlap_candidate = lower <= 0
                    map_candidate = key == map_chip and .98*d-sizes[j] <= view_km*1000
                    if lower > stats["best"] and not overlap_candidate and not map_candidate:
                        continue
                    row, geom, projected = evaluate(j)
                    if overlap_candidate and target.intersects(projected):
                        intersection = target.intersection(projected)
                        if intersection.area > 0:
                            stats["overlaps"].append(intersection)
                            stats["overlap_records"] += 1
                            stats["sites"].add(site)
                    if map_candidate:
                        maps[(source, geom.wkb)] = (source, projected, target.intersection(projected).area > 0)
        processed += len(frame)
        if processed % (65536*10) < len(frame):
            print(f"Scanned {processed:,} records", flush=True)
    rows = []
    for key, state in states.items():
        groups = dict(state["sources"])
        tm = [v for k, v in groups.items() if k.startswith("terramind_")]
        if tm:
            best = min(tm, key=lambda s: s["best"])
            groups["terramind"] = dict(best, overlaps=sum([s["overlaps"] for s in tm], []),
                overlap_records=sum(s["overlap_records"] for s in tm), sites=set().union(*(s["sites"] for s in tm)),
                record_counts=sum((s["record_counts"] for s in tm)),
                nearby_sites=[set().union(*(s["nearby_sites"][i] for s in tm)) for i in range(len(radii))])
        for source, stats in groups.items():
            nearest = stats["nearest"]
            row = dict(chip_id=key, split=state["chip"].split, pretraining_inventory=source,
                       footprint_precision="approximate" if source == "croma_ssl4eo" else "supplied",
                       physical_overlap=bool(stats["overlaps"]), coverage_fraction=coverage(state["target"], stats["overlaps"]),
                       nearest_footprint_km=stats["best"]/1000, nearest_sample_id=nearest.sample_id,
                       nearest_source=nearest.source, nearest_lon=nearest.center_lon, nearest_lat=nearest.center_lat,
                       overlapping_records=stats["overlap_records"], overlapping_unique_centres=len(stats["sites"]))
            for field in ["S1_time", "S2_time", "S1_product_id", "S2L1C_product_id"]:
                row["nearest_"+field] = nearest.get(field, None)
            for i, radius in enumerate(radii):
                row[f"records_within_{radius:g}km"] = int(stats["record_counts"][i])
                row[f"unique_centres_within_{radius:g}km"] = len(stats["nearby_sites"][i])
            rows.append(row)
    return pd.DataFrame(rows), maps, states, counts


def plot_results(summary, maps, state, out, view_km):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Patch, Patch as Legend
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, model, sources in zip(axes, ["croma_ssl4eo", "terramind"],
                                  [["croma_ssl4eo"], ["terramind_ssl4eo", "terramind_majortom"]]):
        for source, geom, hit in maps.values():
            if source not in sources:
                continue
            ax.add_patch(Patch(np.asarray(geom.exterior.coords)/1000, facecolor=COLORS[source],
                              edgecolor=COLORS[source], alpha=.32 if hit else .12, lw=1.3 if hit else .5))
        ax.add_patch(Patch(np.asarray(state["target"].exterior.coords)/1000, fill=False, edgecolor="black", lw=2.5))
        row = summary[(summary.chip_id == state["chip"].sample_id) & (summary.pretraining_inventory == model)].iloc[0]
        qualifier = " (approximate footprints)" if model == "croma_ssl4eo" else " (supplied footprints)"
        ax.set_title(f"{LABELS[model]}{qualifier}\nCoverage {row.coverage_fraction:.1%} · nearest {row.nearest_footprint_km:.3f} km", fontsize=10)
        ax.set(xlim=(-view_km, view_km), ylim=(-view_km, view_km), xlabel="East of chip centre (km)", aspect="equal")
        ax.grid(alpha=.15)
        ax.legend(handles=[Legend(facecolor=COLORS[s], alpha=.5, label=LABELS[s]) for s in sources], loc="lower left", fontsize=8)
    axes[0].set_ylabel("North of chip centre (km)")
    fig.suptitle(f"Geographical pretraining exposure · {state['chip'].sample_id}\nBlack: downstream footprint; stronger outlines: positive-area intersections", fontsize=12)
    fig.tight_layout()
    fig.savefig(out/"local_exposure.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for source, group in summary.groupby("pretraining_inventory", sort=False):
        x = np.sort(group.nearest_footprint_km.to_numpy())
        axes[1].plot(x, np.arange(1,len(x)+1)/len(x), drawstyle="steps-post", marker="o", markersize=4,
                     label=LABELS[source], color=COLORS.get(source,"#555555"))
    grouped = summary.groupby("pretraining_inventory", sort=False).coverage_fraction.mean()
    axes[0].bar(range(len(grouped)), grouped.values*100, color=[COLORS.get(s,"#555555") for s in grouped.index])
    axes[0].set_xticks(range(len(grouped)), [LABELS[s] for s in grouped.index], rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("Mean downstream coverage (%)")
    axes[0].set_ylim(0, 100)
    axes[1].set(xlabel="Nearest footprint distance (km)", ylabel="Fraction of analysed chips", ylim=(0,1.05))
    axes[1].legend(fontsize=7)
    fig.suptitle(f"Metadata exposure · {summary.chip_id.nunique()} downstream chips · CROMA footprints approximate\nGeographical exposure is not evidence of label leakage", fontsize=11)
    fig.tight_layout()
    fig.savefig(out/"exposure_summary.png", dpi=180)
    plt.close(fig)


def self_test():
    target = box(0,0,10,10)
    assert coverage(target, [box(0,0,6,10), box(4,0,8,10)]) == .8
    assert target.distance(box(13,0,20,10)) == 3
    assert target.distance(box(10,0,20,10)) == 0
    assert target.intersection(box(10,0,20,10)).area == 0
    assert coverage(target, [target,target]) == 1
    forward = projection(0,0)
    assert abs(forward(1,0)[0]-GEOD.inv(0,0,1,0)[2]) < .001
    # A larger, more distant-centred patch can be the nearest footprint.
    reverse = Transformer.from_crs(32631, 4326, always_xy=True)
    records = []
    for i, (offset, half, source) in enumerate([
            (5000, 100, "terramind_ssl4eo"), (15000, 12000, "terramind_ssl4eo"),
            (0, 300, "terramind_majortom"), (0, 300, "terramind_majortom")]):
        bounds = [500000+offset-half, -half, 500000+offset+half, half]
        geom = transform(reverse.transform, box(*bounds))
        lon, lat = reverse.transform(500000+offset, 0)
        records.append(dict(sample_id=str(i), source=source, center_lon=lon, center_lat=lat,
                            geometry=geom.wkb, crs=32631, bounds=bounds, split="train"))
    target_geo = transform(reverse.transform, box(499500,-500,500500,500))
    downstream = pd.DataFrame([dict(sample_id="synthetic", split="test", geometry=target_geo,
                                    center_lon=3., center_lat=0.)])
    data = pd.DataFrame(records)
    output, _, _, _ = analyse(downstream, [data.iloc[:1],data.iloc[1:]], [1], "synthetic", 1)
    tm = output.set_index("pretraining_inventory")
    assert tm.loc["terramind_ssl4eo", "nearest_sample_id"] == "1"
    assert tm.loc["terramind_ssl4eo", "nearest_footprint_km"] > 1  # outside proximity radius
    assert tm.loc["terramind_majortom", "records_within_1km"] == 2
    assert tm.loc["terramind_majortom", "unique_centres_within_1km"] == 1
    assert abs(tm.loc["terramind", "coverage_fraction"]-.36) < 1e-4
    together, _, _, _ = analyse(downstream, [data], [1], "synthetic", 1)
    np.testing.assert_allclose(output.nearest_footprint_km, together.nearest_footprint_km)
    # Independent exhaustive footprint comparison on this tiny inventory.
    project = projection(3,0)
    target_m = metric_polygon(target_geo, project)
    expected = min(target_m.distance(metric_polygon(from_wkb(r["geometry"]), project)) for r in records[:2])
    assert abs(tm.loc["terramind_ssl4eo", "nearest_footprint_km"]*1000-expected) < 1e-6
    # A huge patch with a distant centre still overlaps: no fixed-radius miss.
    enlarged = data.copy(deep=True)
    bounds = [499000, -16000, 531000, 16000]
    enlarged.at[1, "bounds"] = bounds
    enlarged.at[1, "geometry"] = transform(reverse.transform, box(*bounds)).wkb
    enlarged_result, _, _, _ = analyse(downstream, [enlarged], [1], "synthetic", 1)
    total = enlarged_result.set_index("pretraining_inventory").loc["terramind"]
    assert total.physical_overlap and abs(total.coverage_fraction-1) < 1e-8
    print("Passed: intersection/union, distance, touching, repeated sites, global nearest vs brute force, and batch invariance")


def world_map(downstream, ssl4eo, terramesh, output):
    """Actual sample centres at every zoom, with online terrain/imagery basemaps.

    Read only centre/source columns. Coincident acquisitions overlap visually.
    Embedded Natural Earth outlines remain available as an offline fallback.
    """
    from shapely.geometry import mapping
    import base64
    import gzip
    coordinates = {source: [] for source in COLORS}
    def accumulate(source, lon, lat):
        xy = np.column_stack([lon, lat])
        if not np.isfinite(xy).all() or (np.abs(xy[:,0]) > 180).any() or (np.abs(xy[:,1]) > 90).any():
            raise ValueError("Invalid world-map coordinates")
        coordinates[source].append(xy)

    for frame in load_metadata("ssl4eo", ssl4eo):
        accumulate("croma_ssl4eo", frame.center_lon.to_numpy(), frame.center_lat.to_numpy())
    parquet = pq.ParquetFile(terramesh)
    for batch in parquet.iter_batches(batch_size=131072, columns=["tar","split","center_lon","center_lat"]):
        frame = batch.to_pandas()
        if not frame.split.eq("train").all():
            raise ValueError("World map requires TerraMesh training metadata")
        matched = np.zeros(len(frame), dtype=bool)
        for prefix, source in [("ssl4eos12","terramind_ssl4eo"),("majortom","terramind_majortom")]:
            mask = frame.tar.str.startswith(prefix).to_numpy()
            matched |= mask
            accumulate(source, frame.loc[mask,"center_lon"].to_numpy(), frame.loc[mask,"center_lat"].to_numpy())
        if not matched.all():
            raise ValueError("Unknown TerraMesh source")
    features = [dict(type="Feature", geometry=mapping(row.geometry), properties=dict(
        chip_id=row.sample_id, split=row.split, lon=row.center_lon, lat=row.center_lat))
        for row in downstream.itertuples()]
    layers = {source: dict(label=LABELS[source], color=COLORS[source],
                          total=sum(len(xy) for xy in coordinates[source])) for source in COLORS}
    # Lossless float64 coordinates, sorted by latitude for viewport lookup.
    # Compress columns separately in one buffer; decode only when the source is first enabled.
    for source in layers:
        xy = np.concatenate(coordinates.pop(source))
        xy = xy[np.argsort(xy[:, 1], kind="stable")]
        raw = np.ascontiguousarray(xy.T, dtype="<f8").tobytes()
        # Byte shuffle preserves every float64 bit while improving compression.
        shuffled = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 8).T.copy().tobytes()
        packed = gzip.compress(shuffled, compresslevel=6, mtime=0)
        assert len(xy) == layers[source]["total"]
        restored = np.frombuffer(gzip.decompress(packed), dtype=np.uint8).reshape(8, -1).T.copy().tobytes()
        assert restored == raw
        layers[source]["coordinates_gzip"] = base64.b64encode(packed).decode("ascii")
        print(f"Packed {source}: {len(xy):,} exact centres ({len(packed)/1e6:.1f} MB)", flush=True)
        del xy, raw, packed, shuffled, restored
    payload = json.dumps(dict(chips=features, layers=layers), separators=(",",":"),
                         ensure_ascii=False).replace("</", "<\\/")
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GeoTest · geographical inventories</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>body{margin:0;font:14px system-ui;color:#192632}header{padding:14px 22px;border-bottom:1px solid #ddd}
h1{font-size:20px;margin:0 0 5px}p{margin:4px 0}#map{height:calc(100vh - 155px);min-height:400px}
.note{font-size:12px;color:#4d5963}button,select{padding:5px;margin:4px;border:1px solid #aaa;background:white;border-radius:4px}
.leaflet-control-layers{max-height:70vh;overflow:auto}footer{padding:8px 22px;font-size:12px}</style></head>
<body><header><h1>Geographical inventories</h1><p id="description"></p>
<span class="note">Actual sample centres at every zoom; no spatial aggregation or sampling.
Repeated acquisitions at identical coordinates overlap. Centres are not footprint coverage.</span><br>
<button onclick="map.setView([15,0],2)">World view</button><label>Find chip <select id="chips"><option value="">Select…</option></select></label>
<span id="status" class="note">Loading map…</span></header><div id="map"></div>
<footer>Colours: blue — CROMA / SSL4EO original; green — TerraMind / SSL4EO v1.1; orange — TerraMind / MajorTOM.
Black outlines — downstream chips. Catalogue exposure does not establish label leakage. Topographic and satellite basemaps require internet. Offline outlines are available in the layer control.</footer>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script id="inventory" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('inventory').textContent);
document.getElementById('description').textContent=`${data.chips.length} Sen1Floods11 chips · toggle inventories and splits using the layer control at top right`;
if(typeof L==='undefined'){document.getElementById('status').textContent='Embedded map library unavailable; regenerate the HTML.';throw Error('Leaflet unavailable');}
const map=L.map('map',{preferCanvas:true,minZoom:2}).setView([15,0],2);
map.createPane('basemap');map.getPane('basemap').style.zIndex=200;
const offline=L.geoJSON(__BASEMAP__,{pane:'basemap',interactive:false,style:{color:'#aeb8bc',weight:.6,fillColor:'#eef0eb',fillOpacity:1},attribution:'Natural Earth - public domain - 1:110m'});
map.getContainer().style.background='#e6eff4';
const tileRoot='https://server.arcgisonline.com/ArcGIS/rest/services/';
const tiles=(service,attribution)=>L.tileLayer(tileRoot+service+'/MapServer/tile/{z}/{y}/{x}',{maxZoom:19,attribution});
const topo=tiles('World_Topo_Map','Tiles &copy; Esri - Esri, HERE, Garmin, USGS and the GIS User Community');
const imagery=tiles('World_Imagery','Imagery &copy; Esri - Maxar, Earthstar Geographics and the GIS User Community');
const labels=tiles('Reference/World_Boundaries_and_Places','Labels &copy; Esri, HERE, Garmin');
const satellite=L.layerGroup([imagery,labels]);
const baseLayers={'Topographic - places and terrain':topo,'Satellite - imagery and labels':satellite,'Offline - country outlines only':offline};
topo.addTo(map);
for(const layer of [topo,imagery,labels])layer.on('tileerror',()=>{
 document.getElementById('status').textContent='Basemap tile unavailable: check your connection or choose another basemap at top right.';
});

map.createPane('downstream');map.getPane('downstream').style.zIndex=450;
const overlays={};const chipLayers={};
const safe=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
for(const [key,source] of Object.entries(data.layers)){
 const ExactLayer=L.Layer.extend({
  onAdd:function(m){this.m=m;this.c=L.DomUtil.create('canvas','leaflet-layer leaflet-zoom-hide');this.c.style.pointerEvents='none';
   m.getPanes().overlayPane.appendChild(this.c);m.on('moveend zoomend resize',this.draw,this);this.draw();},
  onRemove:function(m){m.off('moveend zoomend resize',this.draw,this);this.c.remove();this.m=null;},
  load:function(){if(this.pending)return this.pending;
   this.pending=(async()=>{const binary=atob(source.coordinates_gzip);const bytes=new Uint8Array(binary.length);
    for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
    const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    const buffer=await new Response(stream).arrayBuffer(), shuffled=new Uint8Array(buffer);
    const length=source.total*2, raw=new Uint8Array(buffer.byteLength);
    for(let byte=0;byte<8;byte++)for(let i=0;i<length;i++)raw[i*8+byte]=shuffled[byte*length+i];
    this.points=new Float64Array(raw.buffer);
    if(this.points.length!==source.total*2)throw Error('Coordinate count mismatch');
    source.coordinates_gzip=null;})();return this.pending;},
  draw:async function(){const m=this.m;if(!m)return;
   if(!this.points){document.getElementById('status').textContent='Decoding actual coordinates (first activation only)…';
    try{await this.load();}catch(e){document.getElementById('status').textContent='Could not decode centres: '+e.message;return;}}
   if(this.m!==m)return;
   const size=m.getSize(), origin=m.containerPointToLayerPoint([0,0]);L.DomUtil.setPosition(this.c,origin);
   this.c.width=size.x;this.c.height=size.y;const ctx=this.c.getContext('2d');ctx.fillStyle=source.color;
   const p=this.points,n=source.total,b=m.getBounds();
   const lower=value=>{let lo=0,hi=n;while(lo<hi){const mid=(lo+hi)>>>1;if(p[n+mid]<value)lo=mid+1;else hi=mid;}return lo;};
   const start=lower(b.getSouth()),stop=lower(b.getNorth()+1e-10),scale=256*Math.pow(2,m.getZoom());
   const top=m.project(m.containerPointToLatLng([0,0]),m.getZoom());let drawn=0;
   // Larger, outlined centre markers at regional/local zoom; compact at world scale.
   const radius=m.getZoom()>=9?4:m.getZoom()>=6?3:1;
   const marker=document.createElement('canvas'),side=2*radius+4;
   marker.width=marker.height=side;const brush=marker.getContext('2d');
   brush.beginPath();brush.arc(side/2,side/2,radius,0,2*Math.PI);
   brush.fillStyle=source.color;brush.fill();brush.strokeStyle='#ffffff';brush.lineWidth=1.5;brush.stroke();
   for(let i=start;i<stop;i++){const lon=p[i],lat=p[n+i];if(lon<b.getWest()||lon>b.getEast())continue;
    const sin=Math.sin(Math.max(-85.05112878,Math.min(85.05112878,lat))*Math.PI/180);
    const x=(lon/360+.5)*scale-top.x,y=(.5-Math.log((1+sin)/(1-sin))/(4*Math.PI))*scale-top.y;
    if(m.getZoom()<6)ctx.fillRect(x-1,y-1,2,2);
    else ctx.drawImage(marker,x-side/2,y-side/2);drawn++;}
   document.getElementById('status').textContent='Actual centres · '+source.label+': '+drawn.toLocaleString()+' records in view (coincident centres overlap)';
  }
 });
 overlays[`<span style="color:${source.color}">●</span> ${safe(source.label)} (${source.total.toLocaleString()} records)`]=new ExactLayer();
}
for(const split of ['train','validation','test','bolivia']){
 const chips=data.chips.filter(f=>f.properties.split===split);if(!chips.length)continue;
 const group=L.layerGroup().addTo(map);chipLayers[split]=group;
 overlays[`Sen1Floods11 — ${safe(split)} (${chips.length})`]=group;
 for(const feature of chips){const p=feature.properties;const popup=`<b>${safe(p.chip_id)}</b><br>Split: ${safe(p.split)}<br>Centre: ${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}`;
 L.geoJSON(feature,{pane:'downstream',style:{color:'#111',weight:2,fillOpacity:.04}}).bindPopup(popup).addTo(group);
 L.circleMarker([p.lat,p.lon],{pane:'downstream',radius:4,color:'#111',weight:1.5,fillColor:'#fff',fillOpacity:.9}).bindPopup(popup).addTo(group);}
}
// Support custom downstream split names too.
for(const split of [...new Set(data.chips.map(f=>f.properties.split))].filter(s=>!chipLayers[s])){
 const group=L.geoJSON(data.chips.filter(f=>f.properties.split===split),{pane:'downstream',style:{color:'#111',weight:2},
 onEachFeature:(f,l)=>l.bindPopup(safe(f.properties.chip_id))}).addTo(map);chipLayers[split]=group;overlays[`Sen1Floods11 — ${safe(split)}`]=group;
}
L.control.layers(baseLayers,overlays,{collapsed:false}).addTo(map);L.control.scale().addTo(map);
const select=document.getElementById('chips');
data.chips.sort((a,b)=>a.properties.chip_id.localeCompare(b.properties.chip_id)).forEach((f,i)=>{const o=document.createElement('option');o.value=i;o.textContent=f.properties.chip_id;select.appendChild(o);});
select.onchange=()=>{if(select.value==='')return;const f=data.chips[Number(select.value)];chipLayers[f.properties.split].addTo(map);map.fitBounds(L.geoJSON(f).getBounds(),{padding:[50,50],maxZoom:13});};
document.getElementById('status').textContent='Ready · pretraining layers initially hidden; toggle them on at top right';
</script></body></html>'''
    output.mkdir(parents=True, exist_ok=True)
    path = output/"world_inventory_map.html"
    assets = Path(__file__).resolve().parent / "assets"
    world = json.loads((assets / "world_countries.geojson").read_text(encoding="utf-8"))
    leaflet_js = (assets / "leaflet.js").read_text(encoding="utf-8")
    leaflet_css = (assets / "leaflet.css").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">',
                        "<style>" + leaflet_css + "</style>")
    html = html.replace('<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>',
                        "<script>" + leaflet_js + "</script>")
    html = html.replace("__BASEMAP__", json.dumps(world, separators=(",", ":")))
    path.write_text(html.replace("__DATA__",payload),encoding="utf-8")
    print(f"Saved {path} ({path.stat().st_size/1e6:.2f} MB); {len(features)} chips; "
          + str({s:v['total'] for s,v in layers.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=Path("data/sen1floods11/catalog/sen1floods11_hand_labeled_label/collection.json"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/sen1floods11/splits/all_hand_labels"))
    parser.add_argument("--split", action="append", default=[], metavar="NAME=CSV", help="Override default splits (repeatable)")
    parser.add_argument("--raster-dir", type=Path, default=Path("data/sen1floods11/HandLabeled/LabelHand"))
    parser.add_argument("--ssl4eo", type=Path, default=Path("pretrained_models/pretraining_locations/SSL4EO-S12/ssl4eo-s12_center_coords.csv"))
    parser.add_argument("--terramesh", type=Path, default=Path("pretrained_models/pretraining_locations/TerraMesh/train_metadata.parquet"))
    parser.add_argument("--output", type=Path, default=Path("reports/pretraining_overlap"))
    parser.add_argument("--chip", action="append", help="Analyse only this chip (repeatable); scans full inventories")
    parser.add_argument("--map-chip", default=None)
    parser.add_argument("--view-km", type=float, default=10)
    parser.add_argument("--radii", type=float, nargs="+", default=[5,10,25,50,100])
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--world-map-only", action="store_true", help="All downstream chips and toggleable actual pretraining centres; skip overlap analysis")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.view_km <= 0 or args.batch_size <= 0 or min(args.radii) <= 0:
        parser.error("Radii and batch size must be positive")
    start = time.perf_counter()
    splits = dict(s.split("=",1) for s in args.split) if args.split else {
        k: args.splits_dir/f"flood_{v}_data.csv" for k,v in
        [("train","train"),("validation","valid"),("test","test"),("bolivia","bolivia")]}
    downstream, checks = load_downstream(args.catalog, splits, args.raster_dir)
    if args.world_map_only:
        if args.chip:
            parser.error("--world-map-only displays all split chips; omit --chip")
        world_map(downstream, args.ssl4eo, args.terramesh, args.output)
        return
    split_counts = downstream.groupby("split").size().to_dict()
    if args.chip:
        if set(args.chip)-set(downstream.sample_id):
            parser.error("Unknown requested chip")
        downstream = downstream[downstream.sample_id.isin(args.chip)]
    map_chip = args.map_chip or downstream.iloc[0].sample_id
    if map_chip not in set(downstream.sample_id):
        parser.error("Map chip must be in analysed selection")
    def inventory():
        for kind, path in [("ssl4eo",args.ssl4eo),("terramesh",args.terramesh)]:
            yield from load_metadata(kind, path, args.batch_size)
    print(f"Analysing {len(downstream)} chips; map: {map_chip}; split counts: {split_counts}", flush=True)
    summary, maps, states, counts = analyse(downstream, inventory(), sorted(set(args.radii)), map_chip, args.view_km)
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output/"exposure_summary.csv", index=False)
    plot_results(summary, maps, states[map_chip], args.output, args.view_km)
    provenance = dict(inputs={str(p): {"bytes":p.stat().st_size,"mtime_ns":p.stat().st_mtime_ns}
                             for p in [args.catalog,args.ssl4eo,args.terramesh]},
        splits={k:str(v) for k,v in splits.items()}, split_counts=split_counts, raster_checks=checks,
        inventory_records=counts, seconds=time.perf_counter()-start,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        distance_method="WGS84 chip-centred AEQD, edges densified to 0.001 degrees; global branch-and-bound candidates; metric distances, not exact ellipsoidal minima",
        nearby_method="Haversine centre distance, mean Earth radius 6371008.8 m; unique sites = exact coordinate pairs",
        overlap="Positive area; coverage uses union; TerraMind total unions both sources",
        limitations=["SSL4EO reconstructed footprints are approximate; candidate source raster verification not performed",
                     "TerraMesh current metadata revision is not verified against historical checkpoint pretraining",
                     "No acquisition-product matching to downstream imagery has been performed",
                     "UTM bounds required for TerraMesh; antimeridian candidates rejected explicitly",
                     "Source metadata bounds assumed to enclose supplied footprints; selected candidates checked",
                     "Exposure is not evidence of supervised label leakage"])
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print(summary[["chip_id","pretraining_inventory","coverage_fraction","nearest_footprint_km"]].to_string(index=False))
    print(f"Saved to {args.output}; elapsed {provenance['seconds']:.1f}s")


if __name__ == "__main__":
    main()
