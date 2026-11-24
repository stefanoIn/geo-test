"""Targeted geometry and metadata enrichment regression tests."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from shapely.geometry import box, Polygon, GeometryCollection, LineString

from geography.enrichment import (assign_centres, polygon_parts, prepare_resolve,
                                  compare_repairs, enrich_inventory, validate_regions)


def regions(geometries):
    n = len(geometries)
    return gpd.GeoDataFrame(dict(ECO_ID=list(range(1, n+1)), ECO_NAME=[f'Ecoregion {i}' for i in range(n)],
        BIOME_NUM=[1]*n, BIOME_NAME=['Full biome name']*n, COLOR_BIO=['#123456']*n,
        COLOR=['#ABCDEF']*n, REALM=['Realm']*n), geometry=geometries, crs=4326)


def centres(coords):
    return pd.DataFrame(dict(sample_id=[str(i) for i in range(len(coords))],
                             center_lon=[v[0] for v in coords], center_lat=[v[1] for v in coords]))


class EnrichmentTests(unittest.TestCase):
    def test_vectorized_unique_matches_preserve_fields(self):
        r = regions([box(0,0,1,1)])
        r.loc[0, 'REALM'] = None
        r.index = [57]
        frame = centres([(.123456789012345, .5), (0., .5), (4., 4.)])
        frame.index = [10, 20, 30]
        result = assign_centres(frame, r)
        pd.testing.assert_frame_equal(result[list(frame)], frame.reset_index(drop=True), check_exact=True)
        self.assertEqual(list(result.assignment_status), ['classified', 'boundary', 'unmatched'])
        self.assertEqual(list(result.matched_ecoregion_ids), ['[1]', '[1]', '[]'])
        self.assertTrue(result.realm.isna().all())
        self.assertEqual(str(result.ecoregion_id.dtype), 'Int64')
        for frame in [centres([]), centres([(float('nan'), 0), (181, 0)])]:
            result = assign_centres(frame, r)
            self.assertTrue((result.assignment_status == 'invalid_coordinates').all())
            self.assertTrue(result.ecoregion_id.isna().all())

    def test_vectorized_predicate_error_semantics(self):
        import shapely
        from geography.enrichment import _assign
        r = regions([box(0,0,1,1)])
        tree = shapely.STRtree(r.geometry)
        frame = centres([(.5,.5)])
        for predicate, ids in [('covers', '[]'), ('touches', '[1]')]:
            with patch('shapely.' + predicate, side_effect=shapely.errors.GEOSException('diagnostic failure')):
                result = _assign(frame, r, tree, tolerate_errors=True)
                self.assertEqual(result.assignment_status.iloc[0], 'predicate_error')
                self.assertEqual(result.matched_ecoregion_ids.iloc[0], ids)
                self.assertTrue(pd.isna(result.ecoregion_id.iloc[0]))
                with self.assertRaises(shapely.errors.GEOSException):
                    _assign(frame, r, tree)

    def test_boundaries_missing_and_ambiguity(self):
        r = regions([box(0,0,1,1), box(1,0,2,1)])
        result = assign_centres(centres([(.5,.5),(1,.5),(0,.5),(5,5),(float('nan'),1),(181,1)]), r)
        self.assertEqual(list(result.assignment_status), ['classified','shared_boundary','boundary','unmatched','invalid_coordinates','invalid_coordinates'])
        self.assertTrue(pd.isna(result.ecoregion_id.iloc[1]))
        self.assertEqual(result.biome_id.iloc[1], 1)
        self.assertEqual(result.biome_name.iloc[0], 'Full biome name')
        self.assertEqual(result.ecoregion_color.iloc[0], '#ABCDEF')

    def test_overlapping_regions(self):
        r=regions([box(0,0,2,2),box(1,1,3,3)])
        r.loc[1,['BIOME_NUM','BIOME_NAME','COLOR_BIO']]=[2,'Other biome','#000000']
        result=assign_centres(centres([(1.5,1.5)]),r)
        self.assertEqual(result.assignment_status.iloc[0], 'multiple_matches')
        self.assertTrue(pd.isna(result.biome_id.iloc[0]))

    def test_unclassified(self):
        r=regions([box(0,0,1,1)])
        r.loc[0,['ECO_ID','BIOME_NUM','BIOME_NAME']]=[0,11,'N/A']
        result=assign_centres(centres([(.5,.5)]),r)
        self.assertEqual(result.assignment_status.iloc[0], 'unclassified')
        self.assertEqual(result.ecoregion_id.iloc[0], 0)

    def test_metadata_identity_and_batching(self):
        from geography.metadata import iter_centres
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'inventory.parquet'
            source=pd.DataFrame(dict(sample_id=['same','same'],tar=['a.tar','b.tar'],
                zarr=['a.zarr','b.zarr'],split=['train','val'],center_lon=[1.,2.],center_lat=[3.,4.]))
            source.to_parquet(path,row_group_size=1)
            small=pd.concat(iter_centres('terramesh',path,1),ignore_index=True)
            large=pd.concat(iter_centres('terramesh',path,100),ignore_index=True)
            pd.testing.assert_frame_equal(small,large)
            self.assertEqual(list(small.source_row),[0,1])
            self.assertEqual(list(small.sample_id),['same','same'])
            self.assertEqual(list(small.zarr),['a.zarr','b.zarr'])
            self.assertNotIn('geometry',small)
            catalogue=Path(d)/'collection.json'
            catalogue.write_text(json.dumps({'links':[{'rel':'item','href':'item.json'}]}))
            (Path(d)/'item.json').write_text(json.dumps({'type':'Feature','id':'original_label',
                'properties':{},'geometry':box(1,1,3,3).__geo_interface__}))
            stac=next(iter_centres('stac',catalogue))
            self.assertEqual(stac.sample_id.iloc[0],'original_label')
            self.assertEqual(stac.center_lon.iloc[0],2.)
            self.assertEqual(stac.center_lat.iloc[0],2.)
            (Path(d)/'item.json').write_text(json.dumps({'type':'Feature','id':'missing',
                'properties':{},'geometry':None}))
            missing=next(iter_centres('stac',catalogue))
            self.assertEqual(missing.sample_id.iloc[0],'missing')
            result=assign_centres(missing,regions([box(0,0,1,1)]))
            self.assertEqual(result.assignment_status.iloc[0],'invalid_coordinates')

    def test_existing_exposure_checks(self):
        from experiments.geographical_exposure.geographical_exposure import self_test
        self_test()

    def test_duplicate_source_rows_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/'regions.parquet'
            regions([box(0,0,1,1)]).to_parquet(target)
            batch = centres([(.5,.5)])
            batch['source_row'] = 0
            with self.assertRaises(ValueError):
                enrich_inventory([batch,batch],target,Path(d)/'output.parquet',{})
            self.assertFalse((Path(d)/'output.parquet').exists())

    def test_collections(self):
        g=GeometryCollection([box(0,0,1,1),GeometryCollection([box(2,2,3,3),LineString([(0,0),(1,1)])])])
        self.assertEqual(len(polygon_parts(g)),2)

    def test_repair_and_persistence(self):
        invalid=Polygon([(0,0),(3,0),(3,3),(2,3),(2,2),(1.5,2),(1.5,2.5),(1.5,2),(1,2),(1,3),(0,3),(0,0)])
        original=regions([invalid,box(5,5,6,6)])
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source.shp';output=Path(d)/'validated.parquet'
            original.to_file(source)
            before=source.read_bytes()
            fixed=prepare_resolve(source,output)
            self.assertEqual(before,source.read_bytes())
            self.assertTrue(fixed.is_valid.all())
            self.assertEqual(fixed.geometry.iloc[1].wkb,gpd.read_file(source).geometry.iloc[1].wkb)
            comparison=compare_repairs(centres([(.5,.5),(5.5,5.5),(1.5,2.25)]),gpd.read_file(source),fixed)
            self.assertIn('changed',list(comparison.comparison_status))
            pd.testing.assert_frame_equal(fixed,prepare_resolve(source,output))
            points=centres([(.5,.5),(4,4),(5.5,5.5)])
            for size in [1,3]:
                dest=Path(d)/f'enriched{size}.parquet'
                enrich_inventory((points.iloc[i:i+size] for i in range(0,len(points),size)),output,dest,{'test':True})
                saved=pd.read_parquet(dest)
                self.assertEqual(len(saved),len(points))
                self.assertNotIn('geometry',saved.columns)
                pd.testing.assert_frame_equal(saved[list(points)],points)
            pd.testing.assert_frame_equal(pd.read_parquet(Path(d)/'enriched1.parquet'),pd.read_parquet(Path(d)/'enriched3.parquet'))

    def test_invalid_regions_and_antimeridian(self):
        with self.assertRaises(ValueError):
            validate_regions(regions([Polygon([(0,0),(1,1),(1,0),(0,1),(0,0)])]))
        with self.assertRaises(ValueError):
            validate_regions(regions([box(-179,-1,179,1)]))


class EnrichmentCLITests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        from experiments.geographical_exposure.enrich_metadata import parse_args
        args = parse_args(['--full'])
        self.assertEqual([s.split('=')[0] for s in args.inventory], ['ssl4eo', 'terramesh', 'stac'])
        self.assertEqual(args.output_dir, Path('data/metadata/resolve_full'))
        self.assertEqual(parse_args([]).output_dir, Path('data/metadata/resolve_validation'))
        self.assertEqual(len(parse_args(['--dataset', 'terramesh']).inventory), 1)
        args = parse_args(['--inventory', 'ssl4eo=custom.csv', '--resolve-source', 'custom.shp',
                           '--validated', 'valid.parquet', '--output-dir', 'out'])
        self.assertEqual(args.inventory, ['ssl4eo=custom.csv'])
        self.assertEqual(args.resolve_source, Path('custom.shp'))
        self.assertEqual(args.validated, Path('valid.parquet'))
        for invalid in [['--inventory', 'bad'], ['--batch-size', '0'],
                        ['--dataset', 'stac', '--inventory', 'ssl4eo=other.csv'],
                        ['--inventory', 'stac=a', '--inventory', 'stac=b']]:
            with self.assertRaises(SystemExit):
                parse_args(invalid)

    def test_interrupted_enrichment_preserves_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            region_path = root/'regions.parquet'
            regions([box(0,0,1,1)]).to_parquet(region_path)
            output = root/'output.parquet'
            batch = centres([(.5,.5)])
            enrich_inventory([batch], region_path, output, {'version': 1})
            manifest = output.with_suffix('.provenance.json')
            saved = output.read_bytes(), manifest.read_bytes()
            def interrupted():
                yield batch
                raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                enrich_inventory(interrupted(), region_path, output, {'version': 2})
            self.assertEqual(saved, (output.read_bytes(), manifest.read_bytes()))
            self.assertFalse(list(root.glob('*.pending*')))
            # A matching cache must not consume the supplied generator.
            def unused():
                raise AssertionError('Cache unexpectedly recomputed')
                yield batch
            enrich_inventory(unused(), region_path, output, {'version': 1})
            fresh = root/'fresh.parquet'
            with self.assertRaises(KeyboardInterrupt):
                enrich_inventory(interrupted(), region_path, fresh, {})
            self.assertFalse(fresh.exists())
            self.assertFalse(fresh.with_suffix('.provenance.json').exists())
            self.assertFalse(list(root.glob('*.pending*')))

    def test_interrupted_reporting_and_atomic_files(self):
        from experiments.geographical_exposure.enrich_metadata import describe
        from geography.enrichment import pending_output
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root/'figures'; out.mkdir()
            summary = out/'summary.json'; summary.write_text('completed')
            def interrupted(path, staging):
                (staging/'partial.pdf').write_bytes(b'partial')
                raise KeyboardInterrupt
            with patch('experiments.geographical_exposure.enrich_metadata._describe', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    describe(root/'input.parquet', out)
            self.assertEqual(summary.read_text(), 'completed')
            self.assertEqual(list(root.iterdir()), [out])
            with self.assertRaises(KeyboardInterrupt):
                with pending_output(summary) as pending:
                    pending.write_text('partial')
                    raise KeyboardInterrupt
            self.assertEqual(summary.read_text(), 'completed')
            self.assertFalse(list(out.glob('*.pending*')))

    def test_tiny_all_inventory_cli(self):
        from experiments.geographical_exposure import enrich_metadata as cli
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root/'source.shp'
            regions([box(0,0,1,1)]).to_file(source)
            ssl = root/'ssl.csv'; ssl.write_text('1,0.5,0.5\n2,2.0,2.0\n')
            terra = root/'terra.parquet'
            pd.DataFrame(dict(sample_id=['t'], tar=['majortom_shard_1'], split=['train'],
                              center_lon=[.5], center_lat=[.5])).to_parquet(terra)
            stac = root/'collection.json'
            stac.write_text(json.dumps({'links': [{'rel':'item','href':'item.json'}]}))
            (root/'item.json').write_text(json.dumps({'type':'Feature', 'id':'chip',
                'properties':{}, 'geometry':box(.1,.1,.2,.2).__geo_interface__}))
            out = root/'results'
            argv = ['enrich_metadata.py', '--full', '--resolve-source', str(source),
                    '--validated', str(root/'valid.parquet'), '--output-dir', str(out),
                    '--sample-records', '2', '--batch-size', '1']
            with patch.object(cli, 'DEFAULT_INVENTORIES', {'ssl4eo':ssl, 'terramesh':terra, 'stac':stac}), \
                 patch('sys.argv', argv):
                cli.main()
            for kind, count in [('ssl4eo',2), ('terramesh',1), ('stac',1)]:
                with pq.ParquetFile(out/f'{kind}_enriched.parquet') as parquet:
                    self.assertEqual(parquet.metadata.num_rows, count)
                summary = json.loads((out/f'{kind}_figures/summary.json').read_text())
                self.assertEqual(summary['records'], count)
                self.assertTrue((out/f'{kind}_figures/biome_distribution.pdf').exists())
            self.assertTrue((out/'validation_summary.json').exists())
            self.assertFalse(list(root.rglob('*.pending*')))


if __name__ == '__main__':
    unittest.main()
