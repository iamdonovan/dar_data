import os
import re
import gc
from typing import Any, Union
from glob import glob
from pathlib import Path
import json
import yaml
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


def sin_proj(radius: float = 6371007.181) -> Proj:
    """
    Given a value for Earth radius, return a sinusoidal projection with central longitude and false easting/northing
    of 0.

    :param radius: the value of Earth radius (in m) to use (default: 6371007.181).
    """
    return Proj(f"+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R={radius} +units=m +no_defs")


def parse_corners(meta: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Parse the upper left and lower right (x, y) coordinates from NASA metadata.

    :param meta: the string representation of the metadata object
    :return: (ul, lr) coordinate pairs
    """
    ul_re = re.compile(r'''UpperLeftPointMtrs=\((?P<x>[+-]?\d+\.\d+),(?P<y>[+-]?\d+\.\d+)\)''', re.VERBOSE)
    lr_re = re.compile(r'''LowerRightMtrs=\((?P<x>[+-]?\d+\.\d+),(?P<y>[+-]?\d+\.\d+)\)''', re.VERBOSE)

    ul_match = ul_re.search(meta)
    lr_match = lr_re.search(meta)

    ul = float(ul_match.group('x')), float(ul_match.group('y'))
    lr = float(lr_match.group('x')), float(lr_match.group('y'))

    return ul, lr


def parse_radius(meta: str) -> float:
    """
    Parse the Earth radius value from NASA metadata.

    :param meta: the string representation of the metadata object
    :return: the value of Earth radius
    """
    rad_re = re.compile(r'''ProjParams=\((?P<rad>\d+\.\d+)\,''', re.VERBOSE)
    rad_match = rad_re.search(meta)

    return float(rad_match.group('rad'))


def parse_date_range(meta: str) -> np.datetime64:
    """
    Parse the RANGEDATETIME start and end values from NASA metadata, then return the midpoint of the date range.

    :param meta: the string representation of the metadata object
    :return: the midpoint of the date range
    """
    start_re = re.compile(r'''GROUP\s+=\s+RANGEDATETIME''', re.VERBOSE)
    end_re = re.compile(r'''END_GROUP\s+=\s+RANGEDATETIME''', re.VERBOSE)

    group = meta[start_re.search(meta).start():end_re.search(meta).end()]

    val_re = re.compile(r'''VALUE\s+=\s+"(?P<val>.*?)"\n''', re.VERBOSE)
    matches = val_re.findall(group)

    start = np.datetime64(matches[2] + 'T' + matches[0], 'ns')
    end = np.datetime64(matches[3] + 'T' + matches[1], 'ns')

    return start + (end - start) / 2


def _snow_cover(keys: list, grids: Union[xr.Dataset, dict], mask: bool=True) -> np.ndarray:
    """
    Find the snow cover raster from a dataset.

    :param keys: a list of variable names
    :param grids: an xarray Dataset, or a dict-like representation of a dataset
    :param mask: whether to mask invalid (> 100) snow cover values [default: True]
    :return: an array of snow cover values
    """
    sc_key = [k for k in keys if 'snow_cover' in k.lower() and 'cgf' not in k.lower()][0]

    if isinstance(grids, xr.Dataset):
        snow_cover = grids[sc_key].values
    else:
        snow_cover = grids[sc_key][:]

    snow_cover[np.isnan(snow_cover)] = 255

    if mask:
        snow_cover[snow_cover > 100] = 255

    return snow_cover.astype(np.uint8)


def _load_data(fn_data: Union[str, Path]) -> xr.Dataset:
    """
    Load NASA snow cover data (in either HDF or H5 format) and return a homogenized dataset.

    :param fn_data: the filename of the dataset to open
    :return: an xarray Dataset
    """

    parsed = {}

    ext = os.path.splitext(fn_data)[-1]

    if ext == '.hdf':
        ds = xr.open_dataset(fn_data, engine='netcdf4')
        parsed['meta'] = ds.attrs['StructMetadata.0']

        data = {}

        grid_names = list(ds.keys())
        snow_cover = _snow_cover(grid_names, ds)
        data['snow_cover'] = (['time', 'y', 'x'], np.expand_dims(snow_cover, axis=0))

        cgf_snow = ds['CGF_NDSI_Snow_Cover'].fillna(255).values.astype(np.uint8)
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


def stack_data(fn_out: Union[str, Path],
               dir_name: Union[str, Path],
               globstr: str = None,
               gran_list: Any = None) -> None:
    """
    Given a directory name, load all available NASA snow cover datasets into a single stack, and write the stack to
    disk.

    :param fn_out: the name of the output file to write
    :param dir_name: the name of the directory to search for datasets
    :param globstr: (optional) search string to use to find granules. defaults to *.hdf and *.h5
    :param gran_list: (optional) list of granules to load
    """
    if gran_list is None:
        if globstr is None:
            gran_list = (sorted(glob('*.hdf', root_dir=dir_name)) +
                         sorted(glob('*.h5', root_dir=dir_name)))
        else:
            gran_list = sorted(glob(globstr, root_dir=dir_name))
    print(f"Found {len(gran_list)} tiles to stack.")

    dataset = [os.path.basename(fn).split('.')[0] for fn in gran_list]
    tile_list = [os.path.basename(fn).split('.')[2] for fn in gran_list]

    granules = pd.DataFrame(data={'granule': gran_list, 'tile': tile_list, 'dataset': dataset})
    tile_stacks = []

    for (sens, tile), grans in granules.groupby(['dataset', 'tile']):
        this_stack = [_load_data(Path(dir_name, fn)) for fn in grans['granule']]
        this_ds = xr.concat(this_stack, 'time')

        # reproject the stack to the given CRS
        tile_stacks.append(this_ds)

    print(f"Loaded {len(tile_stacks)} individual tiles. Combining by coordinates.")

    final_stack = xr.combine_by_coords(tile_stacks)

    final_stack['snow_cover'].rio.write_nodata(255, inplace=True)
    final_stack['cgf_snow_cover'].rio.write_nodata(255, inplace=True)

    # save the reprojected stack to a file by compressing the snow cover variables
    final_stack.to_netcdf('tmp.nc', encoding={'snow_cover': {'zlib': True},
                                              'cgf_snow_cover': {'zlib': True}})
    del final_stack, tile_stacks
    gc.collect()

    print("Saving final NetCDF file.")

    # have to do this in two parts, because using encoding somehow breaks writing the CRS variable
    ds = xr.open_dataset('tmp.nc', decode_coords='all')
    ds.rio.write_crs(ds.spatial_ref.crs_wkt, inplace=True)
    ds.to_netcdf(fn_out)

    print("Cleaning up.")
    # clean up the temporary file
    os.remove('tmp.nc')

    print("Finished.")


def reproject_stack(fn_stack: Union[str, Path, xr.Dataset], crs: Any) -> xr.Dataset:
    """
    Reproject an xarray dataset to a given CRS.

    :param fn_stack: the name of the dataset to open, or an xarray Dataset
    :param crs: OGC WKT string or Proj.4 string
    :return: the reprojected Dataset
    """
    if isinstance(fn_stack, (str, Path)):
        ds = xr.open_dataset(fn_stack, decode_coords='all')
        ds = ds.rio.write_crs(ds.spatial_ref.crs_wkt)
    else:
        ds = fn_stack

    return ds.rio.reproject(crs)


def download_basin(name: str, ds_name: str, data_directory: Union[str, Path] = '.', kwargs: dict = {}) -> None:
    """
    Download all data from a given EarthData dataset that intersects a basin of a given name,
    using the file basins.gpkg.

    :param name: the name of the basin to use to search for data
    :param ds_name: the name of the EarthData dataset to search and download
    :param data_directory: the name of the directory to download the files to
    :param kwargs: additional keyword arguments to pass to earthaccess.search_data
    """
    basins = gpd.read_file('../basins.gpkg').set_index('name')
    basin = basins.loc[name, 'geometry']

    download_from_extent(basin, ds_name, Path(data_directory, unidecode(name)), kwargs=kwargs)


def download_from_extent(geom: shapely.geometry.Polygon,
                         ds_name: str,
                         data_directory: Union[str, Path] = '.',
                         kwargs: dict = {}) -> None:
    """
    Given a geometric representation of a search area, download all granules from an EarthData dataset that intersect
    that geometry.

    :param geom: a shapely polygon representing the search area (with latitude/longitude coordinates)
    :param ds_name: the name of the EarthData dataset to search and download
    :param data_directory: the name of the directory to download the files to
    :param kwargs: additional keyword arguments to pass to earthaccess.search_data
    """
    search_area = shapely.geometry.polygon.orient(geom.minimum_rotated_rectangle, sign=1)

    earthaccess.login(strategy='netrc')

    results = earthaccess.search_data(
        short_name=ds_name,
        polygon=search_area.exterior.coords,
        **kwargs
    )

    earthaccess.download(
        results,
        Path(data_directory, ds_name)
    )


def _pct_valid(arr, aoi_mask):
    return 100 * np.count_nonzero(np.isfinite(arr[aoi_mask])) / np.count_nonzero(aoi_mask)


def filter_stack(ds: xr.Dataset,
                 threshold: Union[float, int],
                 aoi_mask: np.typing.ArrayLike = None) -> xr.Dataset:
    """
    Filter a snowcover stack based on the % of valid pixels inside of the AOI.

    :param ds: The stack to filter.
    :param threshold: The threshold to use for filtering. Timestamps with (% coverage <= threshold) will be dropped.
    :param aoi_mask: (optional) AOI mask to use. Defaults to entire (x, y) dimension of raster.
    :return: the filtered dataset
    """
    if aoi_mask is None:
        aoi_mask = np.isfinite(np.ones(ds['snow_cover'][0].shape))

    ds['pct_coverage'] = xr.apply_ufunc(_pct_valid, ds['snow_cover'],
                                        input_core_dims=[['y', 'x']],
                                        vectorize=True,
                                        kwargs={'aoi_mask': aoi_mask})

    return ds.where(ds['pct_coverage'] >= threshold, drop=True)


def landsat_metadata(granule: str, data_dir: str = '.') -> dict:
    """
    Load metadata for a Landsat granule.

    :param granule: The Landsat product ID to load (e.g., LC08_L1TP_...)
    :param data_dir: The directory where the Landsat directory is. Defaults to current directory.
    :return: a dict of the Landsat metadata
    """
    with open(Path(data_dir, granule, granule + '_MTL.json'), 'r') as f:
        return json.load(f)


def parse_ang_file(granule: str, data_dir: str = '.') -> dict:
    """
    Parse a Landsat angle coefficient file (_ANG.txt) into a dict.

    :param granule: The Landsat product ID to load (e.g., LC08_L1TP_...)
    :param data_dir: The directory where the Landsat directory is. Defaults to current directory.
    :return: a dict of the angle coefficient file metadata
    """
    with open(Path(data_dir, granule, granule + '_ANG.txt'), 'r') as f:
        raw_ang = [l.strip() for l in f.readlines()]

    out_dict = dict()
    for l in raw_ang:
        if l == 'END':
            continue
        lsplit = l.split(' = ')
        if lsplit[0] == 'GROUP':
            this_group = lsplit[1]
            this_dict = dict()
        else:
            if lsplit[0] == 'END_GROUP':
                out_dict[this_group] = this_dict
                continue
            if '=' in l:
                this_name = lsplit[0]
                this_dict[this_name] = lsplit[1]
            else:
                this_dict[this_name] = this_dict[this_name] + ' ' + l

    # now clean the results
    for key in out_dict.keys():
        this_dict = out_dict[key]
        for _k in this_dict.keys():
            if ',' in this_dict[_k]:
                out_dict[key][_k] = np.array(
                    [yaml.load(a, Loader=yaml.SafeLoader) for a in this_dict[_k].strip('(').strip(')').split(',')])
            else:
                out_dict[key][_k] = yaml.load(this_dict[_k], Loader=yaml.SafeLoader)

    return out_dict
