from pathlib import Path
import numpy as np
import geoutils as gu
import pandas as pd
from tqdm import tqdm


def prep_df(fn_csv):
    df = pd.read_csv(fn_csv)
    df.dropna(subset='thresh', inplace=True)

    df['date'] = pd.to_datetime(df['granule'].str.split('_', expand=True)[3], format='%Y%m%d')
    df['granule'] = df.groupby('date')['granule'].transform(lambda x: ','.join(x))

    avg_vals = df.groupby('date')[['thresh', 'cloud_scene', 'cloud_land']].mean()
    df.set_index('date', inplace=True)

    df[['thresh', 'cloud_scene', 'cloud_land']] = avg_vals

    return df.drop_duplicates().sort_index().reset_index()


def stack_rasters(df, data_dir='.', res=500):
    rasters = []

    for row in tqdm(df.itertuples(), total=df.shape[0]):
        if ',' in row.granule:
            _rasts = [gu.Raster(Path(data_dir, f"{gran}_albedo_scene_snow.tif")) for gran in row.granule.split(',')]
            rast = gu.raster.merge_rasters(_rasts, merge_algorithm=np.max, progress=False)
        else:
            rast = gu.Raster(Path(data_dir, f"{row.granule}_albedo_scene_snow.tif"))
        rasters.append(rast.reproject(res=res, resampling='mode'))
    return gu.raster.stack_rasters(rasters)


fn_csv = 'snowline_thresholds.csv'
out_name = 'stacked.nc'
out_res = 500

thresholds = prep_df(fn_csv)

stack = stack_rasters(thresholds, res=out_res)
stack.reproject(res=out_res, inplace=True)

# get rid of fill values from reprojecting, re-mask
stack.data.data[stack.data.data == -1] = 0
stack.data.mask = stack.data.data == 0

# convert to xarray dataset
out_ds = stack.to_xarray(name='snow_class').to_dataset()

# add time coordinate, swap with band
out_ds.coords['time'] = ('band', thresholds['date'])
out_ds = out_ds.swap_dims({'band': 'time'})

# add other variables
out_ds['granule'] = ('time', thresholds['granule'])
out_ds['cloud_scene'] = ('time', thresholds['cloud_scene'])
out_ds['cloud_land'] = ('time', thresholds['cloud_land'])

# drop band
out_ds = out_ds.drop_vars('band')

# save to netcdf
out_ds['snow_class'] = out_ds['snow_class'].astype(np.uint8)
out_ds['snow_class'].rio.write_nodata(0, inplace=True)
out_ds.rio.reproject('epsg:4326').to_netcdf(out_name)
