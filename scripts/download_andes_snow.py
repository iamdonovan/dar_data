from tools import tools
from shapely.geometry import Polygon


datasets = ['MOD10A1F', 'MYD10A1F', 'VJ110A1F', 'VNP10A1F']

ulx = -82
uly = 5
lrx = -62
lry = -57

search_area = Polygon([(ulx, uly), (lrx, uly), 
                       (lrx, lry), (ulx, lry)])

for dataset in datasets:
    tools.download_from_extent(search_area, dataset)

