"""Footprint exposure calculations; inventory semantics match the original analysis."""
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from sklearn.neighbors import BallTree
from shapely import from_wkb, segmentize
from shapely.geometry import box
from shapely.ops import transform, unary_union

R = 6371008.8
GEOD = Geod(ellps="WGS84")

def projection(lon, lat):
    crs = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m"
    return Transformer.from_crs(4326, crs, always_xy=True).transform


def metric_polygon(geometry, project):
    # Dense geographic edges retain the footprint shape when projected.
    return transform(project, segmentize(geometry, 0.001))


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


