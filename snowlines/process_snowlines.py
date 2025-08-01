import sys
sys.path.append('')
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from tools import ssam


fn_dem = ''
fn_outlines = ''
data_dir = ''

granules = sorted(glob('*T1', root_dir=data_dir))
threshes = []

for granule in tqdm(granules):
    print(granule)
    try:
        _, thresh = ssam.snow_map(granule,
                                  fn_dem,
                                  fn_outlines,
                                  data_dir=data_dir,
                                  method='albedo',
                                  how='scene',
                                  return_rast=True)
        threshes.append(thresh)
    except ValueError as e:
        print(e)
        threshes.append(np.nan)
        continue


thresh_df = pd.DataFrame(data={'granule': granules, 'thresh': threshes})
thresh_df.to_csv('snowline_thresholds.csv', index=False)
