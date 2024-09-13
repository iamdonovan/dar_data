import os
import re
from pathlib import Path
from unidecode import unidecode
import shapely
import xarray as xr
import geopandas as gpd
from pyproj import Proj
import earthaccess


def sin_proj(radius=6371007.181):
    return Proj(f"+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R={radius} +units=m +no_defs")


def parse_corners(meta):

    ul_re = re.compile(r'''UpperLeftPointMtrs=\((?P<x>[+-]?\d+\.\d+),(?P<y>[+-]?\d+\.\d+)\)''', re.VERBOSE)
    lr_re = re.compile(r'''LowerRightMtrs=\((?P<x>[+-]?\d+\.\d+),(?P<y>[+-]?\d+\.\d+)\)''', re.VERBOSE)

    ul_match = ul_re.search(meta)
    lr_match = lr_re.search(meta)

    ul = float(ul_match.group('x')), float(ul_match.group('y'))
    lr = float(lr_match.group('x')), float(lr_match.group('y'))

    return ul, lr


def parse_radius(meta):

    rad_re = re.compile(r'''ProjParams=\((?P<rad>\d+\.\d+)\,''', re.VERBOSE)
    rad_match = rad_re.search(meta)

    return float(rad_match.group('rad'))


def load_data(fn_data):

    ext = os.path.spliteext(fn_data)[-1]
    if ext == '.hdf':
        pass

    elif ext == '.h5':
        pass

    else:
        pass


def to_netcdf(fn_data):
    pass


def to_geotiff(fn_data, dataname):
    pass


def download_basin(name, ds_name, data_directory='.'):
    basins = gpd.read_file('../basins.gpkg').set_index('name')
    basin = basins.loc[name, 'geometry']

    search_area = shapely.geometry.polygon.orient(basin.minimum_rotated_rectangle, sign=1)

    earthaccess.login(strategy='netrc')

    results = earthaccess.search_data(
        short_name=ds_name,
        polygon=search_area.exterior.coords
    )

    earthaccess.download(
        results,
        Path(data_directory, f"{unidecode(name)}_{ds_name}")
    )
