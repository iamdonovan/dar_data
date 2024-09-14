import os
import re
from pathlib import Path

from earthaccess import download
from unidecode import unidecode
import shapely
import numpy as np
import h5py
import rioxarray
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


def parse_date_range(meta):

    start_re = re.compile(r'''GROUP\s+=\s+RANGEDATETIME''', re.VERBOSE)
    end_re = re.compile(r'''END_GROUP\s+=\s+RANGEDATETIME''', re.VERBOSE)

    group = meta[start_re.search(meta).start():end_re.search(meta).end()]

    val_re = re.compile(r'''VALUE\s+=\s+"(?P<val>.*?)"\n''', re.VERBOSE)
    matches = val_re.findall(group)

    start = np.datetime64(matches[2] + 'T' + matches[0])
    end = np.datetime64(matches[3] + 'T' + matches[1])

    return start + (end - start) / 2


def load_data(fn_data):

    parsed = {}

    ext = os.path.splitext(fn_data)[-1]

    if ext == '.hdf':
        ds = xr.open_dataset(fn_data, engine='netcdf4')
        parsed['meta'] = ds.attrs['StructMetadata.0']

        grid_names = list(ds.keys())

        data = {}
        for grid in grid_names:
            data[grid] = (['time', 'y', 'x'], np.expand_dims(ds[grid].values, axis=0))
        parsed['data'] = data

        meta = ds.attrs['CoreMetadata.0']

        parsed['date'] = parse_date_range(meta)

        cc_re_start = re.compile(r'''OBJECT\s+=\s+QAPERCENTCLOUDCOVER''', re.VERBOSE)
        cc_re_end = re.compile(r'''END_OBJECT\s+=\s+QAPERCENTCLOUDCOVER''', re.VERBOSE)

        cc_txt = meta[cc_re_start.search(meta).start():cc_re_end.search(meta).end()]
        cc_val = re.compile(r'''VALUE\s+=\s+(?P<val>\d+)\n''', re.VERBOSE)

        parsed['cloud_cover'] = float(cc_val.search(cc_txt).group('val'))

        row_dim = [n for n in ds.sizes if 'YDim' in n][0]
        col_dim = [n for n in ds.sizes if 'XDim' in n][0]

        rows, cols = ds.sizes[row_dim], ds.sizes[col_dim]

    elif ext == '.h5':
        with h5py.File(fn_data) as ds:
            parsed['meta'] = ds['HDFEOS INFORMATION/StructMetadata.0'][()].decode('UTF-8')

            data_fields = ds['HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields']
            grid_names = [n for n in data_fields.keys() if 'Projection' not in n]

            data = {}
            for grid in grid_names:
                data[grid] = (['time', 'y', 'x'], np.expand_dims(data_fields[grid][:], axis=0))
            parsed['data'] = data

        # there has to be a way to get this from h5py.File
        ds = xr.open_dataset(fn_data, engine='netcdf4')

        start_time = np.datetime64(ds.attrs['StartTime'])
        end_time = np.datetime64(ds.attrs['EndTime'])

        parsed['date'] = start_time + (end_time - start_time) / 2
        parsed['cloud_cover'] = ds.attrs['Cloud_Cover_Extent']

        rows, cols = parsed['attributes']['DataRows'], parsed['attributes']['DataColumns']

    else:
        pass

    ul, lr = parse_corners(parsed['meta'])
    radius = parse_radius(parsed['meta'])

    projstr = sin_proj(radius).to_proj4()

    dx = (lr[0] - ul[0]) / cols
    dy = (ul[1] - lr[1]) / rows

    xx = np.linspace(ul[0], lr[0], cols, endpoint=False)
    yy = np.linspace(ul[1], lr[1], rows, endpoint=False)

    parsed['spatial'] = {'ul': ul, 'lr': lr, 'proj': projstr, 'dx': dx, 'dy': dy}

    # TODO: create xarray Dataset, including crs variable

    return parsed


def to_netcdf(fn_data):
    pass


def to_geotiff(fn_data, dataname):
    pass


def download_basin(name, ds_name, data_directory='.'):
    basins = gpd.read_file('../basins.gpkg').set_index('name')
    basin = basins.loc[name, 'geometry']

    download_from_extent(basin, ds_name, Path(data_directory, unidecode(name)))


def download_from_extent(geom, ds_name, data_directory='.'):
    search_area = shapely.geometry.polygon.orient(geom.minimum_rotated_rectangle, sign=1)

    earthaccess.login(strategy='netrc')

    results = earthaccess.search_data(
        short_name=ds_name,
        polygon=search_area.exterior.coords
    )

    earthaccess.download(
        results,
        Path(data_directory, ds_name)
    )
