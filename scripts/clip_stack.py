#!/usr/bin/env ipython
from pathlib import Path
from unidecode import unidecode
import numpy as np
import geopandas as gpd
import rioxarray
import xarray as xr
from glob import glob
from tools import tools


def pct_valid(arr, aoi_mask):
    return 100 * np.count_nonzero(np.isfinite(arr[aoi_mask])) / np.count_nonzero(aoi_mask)


min_cover = 60 # only keep days when at least this % of each basin is covered

sin_crs = tools.sin_proj().to_proj4()
dir_stacks = 'stacks' # the folder to look for the stacked data in

tiles = gpd.read_file('modis_tiles.gpkg')
basins = gpd.read_file('basins.gpkg')

for name in basins['name']:
    basin = basins.to_crs(sin_crs).loc[basins['name'] == name]
    geom = basin.geometry.buffer(1e4).envelope.values[0]

    # get only those tiles that intersect the envelope
    these_tiles = tiles.to_crs(sin_crs).loc[tiles.to_crs(sin_crs).intersects(geom)]

    clipped = []
    for _, tile in these_tiles.iterrows():
        clip_geom = tile.geometry.intersection(geom).envelope

        this_stack = []
        for fn_stack in sorted(glob(f"*{tile['tile']}*.nc", root_dir=dir_stacks)):
            ds = xr.open_dataset(Path(dir_stacks, fn_stack), decode_coords='all')

            this_clip = ds.rio.clip_box(**gpd.GeoSeries(clip_geom).bounds)
            this_stack.append(this_clip)

        clipped.append(xr.concat(this_stack, 'time'))

    merged = xr.combine_by_coords(clipped)

    _mask = merged['snow_cover'][0]
    _mask.data = np.ones(_mask.data.shape)
    mask = np.isfinite(_mask.rio.clip(basin.geometry, drop=False))

    merged['pct_coverage'] = xr.apply_ufunc(pct_valid, merged['snow_cover'],
                                            input_core_dims=[['y', 'x']],
                                            vectorize=True,
                                            kwargs={'mask': mask})

    merged = merged.where(merged['pct_coverage'] > min_cover, drop=True)
    merged = merged.rio.reproject(4326)

    merged.to_netcdf('tmp.nc', encoding={'snow_cover': {'zlib': True},
                                         'cgf_snow_cover': {'zlib': True}})

    # get the name ready for output
    fn = unidecode(name.lower().replace(' ', '_'))

    ds = xr.open_dataset('tmp.nc', decode_coords='all')
    ds.rio.write_crs(ds.spatial_ref.crs_wkt, inplace=True)
    ds.to_netcdf(f"{fn}.nc")
