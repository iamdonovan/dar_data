import os
import re
from glob import glob
from pathlib import Path
import pandas as pd
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

    start = np.datetime64(matches[2] + 'T' + matches[0], 'ns')
    end = np.datetime64(matches[3] + 'T' + matches[1], 'ns')

    return start + (end - start) / 2


def _snow_cover(keys, grids, mask=True):
    sc_key = [k for k in keys if 'snow_cover' in k.lower() and 'cgf' not in k.lower()][0]

    if isinstance(grids, xr.Dataset):
        snow_cover = grids[sc_key].values
    else:
        snow_cover = grids[sc_key][:]

    snow_cover[np.isnan(snow_cover)] = 255

    if mask:
        snow_cover[snow_cover > 100] = 255

    return snow_cover.astype(np.uint8)


def _load_data(fn_data):

    parsed = {}

    ext = os.path.splitext(fn_data)[-1]

    if ext == '.hdf':
        ds = xr.open_dataset(fn_data, engine='netcdf4')
        parsed['meta'] = ds.attrs['StructMetadata.0']

        data = {}

        grid_names = list(ds.keys())
        snow_cover = _snow_cover(grid_names, ds)
        data['snow_cover'] = (['time', 'y', 'x'], np.expand_dims(snow_cover, axis=0))

        cgf_snow = ds['CGF_NDSI_Snow_Cover'].values.astype(np.uint8)
        cgf_snow[cgf_snow > 100] = 255
        data['cgf_snow_cover'] = (['time', 'y', 'x'], np.expand_dims(cgf_snow, axis=0))

        meta = ds.attrs['CoreMetadata.0']

        parsed['date'] = parse_date_range(meta)

        cc_re_start = re.compile(r'''OBJECT\s+=\s+QAPERCENTCLOUDCOVER''', re.VERBOSE)
        cc_re_end = re.compile(r'''END_OBJECT\s+=\s+QAPERCENTCLOUDCOVER''', re.VERBOSE)

        cc_txt = meta[cc_re_start.search(meta).start():cc_re_end.search(meta).end()]
        cc_val = re.compile(r'''VALUE\s+=\s+(?P<val>\d+)\n''', re.VERBOSE)

        # data['granule'] = (['time'], np.array([os.path.splitext(os.path.basename(fn_data))[0]]))
        # data['cloud_cover'] = (['time'], np.array([float(cc_val.search(cc_txt).group('val'))]))

        parsed['data'] = data

    elif ext == '.h5':
        with h5py.File(fn_data) as ds:
            parsed['meta'] = ds['HDFEOS INFORMATION/StructMetadata.0'][()].decode('UTF-8')

            data_fields = ds['HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields']
            grid_names = [n for n in data_fields.keys() if 'Projection' not in n]

            data = {}

            snow_cover = _snow_cover(grid_names, data_fields)
            data['snow_cover'] = (['time', 'y', 'x'], np.expand_dims(snow_cover, axis=0))

            cgf_snow = data_fields['CGF_NDSI_Snow_Cover'][:].astype(np.uint8)
            cgf_snow[cgf_snow > 100] = 255
            data['cgf_snow_cover'] = (['time', 'y', 'x'], np.expand_dims(cgf_snow, axis=0))


        # there has to be a way to get this from h5py.File
        ds = xr.open_dataset(fn_data, engine='netcdf4')

        start_time = np.datetime64(ds.attrs['StartTime'], 'ns')
        end_time = np.datetime64(ds.attrs['EndTime'], 'ns')

        parsed['date'] = start_time + (end_time - start_time) / 2

        # data['granule'] = (['time'], np.array([os.path.splitext(os.path.basename(fn_data))[0]]))
        # data['cloud_cover'] = (['time'], np.array([float(ds.attrs['Cloud_Cover_Extent'].strip('%'))]))

        parsed['data'] = data

    else:
        pass

    ul, lr = parse_corners(parsed['meta'])
    radius = parse_radius(parsed['meta'])

    projstr = sin_proj(radius).to_proj4()
    parsed['spatial'] = {'ul': ul, 'lr': lr, 'proj': projstr}

    _, cols, rows = parsed['data']['snow_cover'][1].shape

    xx = np.linspace(parsed['spatial']['ul'][0], parsed['spatial']['lr'][0], cols, endpoint=False)
    yy = np.linspace(parsed['spatial']['ul'][1], parsed['spatial']['lr'][1], rows, endpoint=False)

    out_ds = xr.Dataset(
        data_vars=parsed['data'],
        coords=dict(
            x=xx,
            y=yy,
            time=[parsed['date']]
        )
    )

    out_ds.rio.write_crs(parsed['spatial']['proj'], inplace=True)

    return out_ds


def stack_data(fn_out, dir_name):

    gran_list = sorted(glob('*.hdf', root_dir=dir_name)) + sorted(glob('*.h5', root_dir=dir_name))
    dataset = [os.path.basename(fn).split('.')[0] for fn in gran_list]
    tile_list = [os.path.basename(fn).split('.')[2] for fn in gran_list]

    granules = pd.DataFrame(data={'granule': gran_list, 'tile': tile_list, 'dataset': dataset})
    tile_stacks = []

    for (sens, tile), grans in granules.groupby(['dataset', 'tile']):
        this_stack = [_load_data(Path(dir_name, fn)) for fn in grans['granule']]
        this_ds = xr.concat(this_stack, 'time')
        tile_stacks.append(this_ds)

    final_stack = xr.combine_by_coords(tile_stacks)

    final_stack['cloud_cover'].rio.write_nodata(255, inplace=True)
    final_stack['cgf_cloud_cover'].rio.write_nodata(255, inplace=True)

    final_stack.to_netcdf(fn_out)


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
