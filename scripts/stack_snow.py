#!/usr/bin/env python
from tools import tools
from glob import glob


datasets = ['MOD10A1F', 'MYD10A1F', 'VJ110A1F', 'VNP10A1F']

for dataset in datasets:
    gran_list = glob('*.hdf', root_dir=dataset)
    tiles = sorted(list(set([g.split('.')[2] for g in gran_list])))
    years = sorted(list(set([g.split('.')[1].strip('A')[:4] for g in gran_list])))

    for tile in tiles:
        for year in years:
            tools.stack_data(f"{dataset}_{tile}_{year}.nc", dir_name=dataset, globstr=f"*.A{year}*.{tile}.*.hdf")
