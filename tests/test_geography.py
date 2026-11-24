"""Targeted geometry and metadata enrichment regression tests."""
import json
import tempfile
import unittest
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


if __name__ == '__main__':
    unittest.main()
