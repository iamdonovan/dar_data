from pathlib import Path
import geopandas as gpd
import rioxarray
import xarray as xr
from glob import glob
from tools import tools

sin_crs = tools.sin_proj().to_proj4()
dir_stacks = 'stacks'

basins = gpd.read_file('basins.gpkg')
tiles = gpd.read_file('modis_tiles.gpkg')

basin = basins.loc[basins['name'] == 'Santa'].to_crs(sin_crs).buffer(1e5).values[0].envelope

# get only those tiles that intersect the envelope
tiles = tiles.to_crs(sin_crs).loc[tiles.to_crs(sin_crs).intersects(basin)]

clipped = []
for _, row in tiles.iterrows():
    clip_geom = row['geometry'].intersection(basin).envelope

    this_stack = []
    for fn_stack in sorted(glob(f"*{row['tile']}*.nc", root_dir=dir_stacks)):
        ds = xr.open_dataset(Path(dir_stacks, fn_stack), decode_coords='all')
        this_clip = ds.rio.clip_box(**gpd.GeoSeries(clip_geom).bounds)
        this_stack.append(this_clip)

    clipped.append(xr.concat(this_stack, 'time'))

merged = xr.combine_by_coords(clipped)
