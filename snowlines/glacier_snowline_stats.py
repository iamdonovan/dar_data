import sys
sys.path.append('..')
from pathlib import Path
import pandas as pd
from tools.ssam import get_snowline_stats


def prep_df(fn_csv):
    df = pd.read_csv(fn_csv)
    df.dropna(subset='thresh', inplace=True)

    df['date'] = pd.to_datetime(df['granule'].str.split('_', expand=True)[3], format='%Y%m%d')
    df['granule'] = df.groupby('date')['granule'].transform(lambda x: ','.join(x))

    avg_vals = df.groupby('date')[['thresh', 'cloud_scene', 'cloud_land']].mean()
    df.set_index('date', inplace=True)

    df[['thresh', 'cloud_scene', 'cloud_land']] = avg_vals

    return df.drop_duplicates().sort_index().reset_index()


fn_dem = ''
fn_outlines = ''
fn_csv = 'snowline_thresholds.csv'
data_dir = '.'

thresholds = prep_df(fn_csv)

for granule in thresholds['granule']:
    if ',' in granule:
        if all([Path(data_dir, f"{gran}_albedo_scene_snow.tif").exists() for gran in granule.split(',')]):
            get_snowline_stats(granule, fn_dem, fn_outlines, data_dir=data_dir, bin_size=20)
        else:
            continue
    else:
        if Path(data_dir, f"{granule}_albedo_scene_snow.tif").exists():
            get_snowline_stats(granule, fn_dem, fn_outlines, data_dir=data_dir, bin_size=20)
        else:
            continue
