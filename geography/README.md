# Geographical metadata and exposure

This package has no dependency on GeoTest training, encoders or evaluation. Paths
are explicit function arguments. Runtime dependencies are NumPy, pandas, PyArrow,
GeoPandas, Shapely >= 2.1, pyproj, rasterio and scikit-learn. The separate reporting
script additionally uses Matplotlib and SciencePlots.

- `metadata.iter_centres(kind, path, batch_size)` yields lightweight inventory
  records for `ssl4eo`, `terramesh` and `stac`. It does not select experiment roles
  or splits. Original STAC item IDs are preserved. Fallback centres are planar
  WGS84 footprint centroids, matching the existing exposure convention. Missing,
  invalid or dateline-crossing footprints without explicit centres retain their
  records with unavailable coordinates and an invalid-coordinate assignment.
- `enrichment.prepare_resolve(source, output)` repairs invalid geometries only,
  using linework make-valid in EPSG:4326. GeometryCollections are reduced to their
  polygonal components. No simplification or map projection is applied. All source
  attributes are preserved, including licensing and original area fields.
- `enrichment.enrich_inventory(batches, validated_path, output, provenance)` writes
  batches to compressed Parquet. Future adapters only need to supply sample IDs,
  WGS84 centre coordinates and source identifiers; no dataset class is required.
- `exposure` contains the existing footprint calculations. The original command
  at `experiments/geographical_exposure/geographical_exposure.py` and its flags
  remain available. Enrichment is not a prerequisite for exposure.

Enrichment outputs contain `sample_id`, `center_lon`, `center_lat`, `biome_id`,
`biome_name`, `biome_color`, `ecoregion_id`, `ecoregion_name`, `ecoregion_color`,
`realm` and categorical `assignment_status`, plus source retrieval fields and
`matched_ecoregion_ids`. No footprint geometries are copied. Original names and
HEX colours are retained. `source_row` together with the source fingerprint is
record identity; repeated sample IDs and acquisitions are not collapsed. TerraMesh
also retains `tar` and `zarr` when supplied.

Assignments use `covers`, with `touches` distinguishing boundaries. Multiple
matches remain explicit; no first-match or nearest-region fallback is used. A
shared biome is retained only when every candidate is classified in that biome.
Unclassified RESOLVE record attributes are retained but excluded from classified
category counts. Classification describes the centre, not footprint coverage.

Repair produces a GeoParquet file, `.repair.csv` audit and `.provenance.json`.
Review thresholds (relative geodetic area change > 0.0001 or bounds change > 1e-8
degrees) stop preparation; they are diagnostic thresholds, not scientific error
bounds. Areas of invalid originals are algebraic references only. Source hashes,
attributes, unchanged valid geometries and persistence are checked. Antimeridian
edges requiring longitude unwrapping are rejected. Longitude wrapping is not
performed; exact +/-180-degree boundary matches can depend on representation.

Run bounded validation from the repository directory:

```powershell
python experiments/geographical_exposure/enrich_metadata.py --resolve-source data/environmental/resolve_2017/Ecoregions2017.shp --validated data/environmental/resolve_2017_validated.parquet --inventory ssl4eo=pretrained_models/pretraining_locations/SSL4EO-S12/ssl4eo-s12_center_coords.csv --inventory terramesh=pretrained_models/pretraining_locations/TerraMesh/train_metadata.parquet --inventory stac=data/sen1floods11/catalog/sen1floods11_hand_labeled_label/collection.json --output-dir data/metadata/resolve_validation --sample-records 1024 --batch-size 256
python -m unittest tests.test_geography -v
```

Default validation samples SSL4EO with seed 42 and TerraMesh from prefixes of
row groups distributed through the file; the entire supplied STAC catalogue is
included. These are diagnostic subsets, not unbiased estimates of full-inventory
composition. Before/after repair comparisons report changed assignments and
original predicate failures separately. Zero observed changes in a subset do not
prove classification invariance elsewhere. CSV category counts, PDF plots and
`validation_summary.json` describe the scope actually processed.

Use `--full` and a separate output directory to enrich full inventories explicitly.
Only the repair comparison remains subset-based. Enrichment and descriptive
aggregation are batched; inventory coordinates are not duplicated as polygons.
Matching source/configuration fingerprints reuse completed enrichment outputs.
Geographical exposure and ecological membership do not prove historical model
exposure, representation quality or downstream label leakage.
