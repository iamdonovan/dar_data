from pathlib import Path
import geoutils as gu
import tools


def toa_reflectance(band: int, raster: gu.Raster, metadata: dict) -> gu.Raster:
    """
    Convert a Landsat band to TOA reflectance.

    :param band: the band number to convert
    :param raster: a gu.Raster of the Landsat band
    :param metadata: a dict of the Landsat metadata
    :return: the raster, converted to TOA reflectance
    """
    raster *= float(metadata['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'][f"REFLECTANCE_MULT_BAND_{band}"])
    raster += float(metadata['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'][f"REFLECTANCE_ADD_BAND_{band}"])

    return raster


def ndsi(granule: str, data_dir: str = '.') -> gu.Raster:
    """
    Calculate the normalized difference snow and ice index (NDSI) for a Landsat scene, using the formula:

    NDSI = (Green - SWIR1) / (Green + SWIR1)

    Where Green and SWIR1 are the visible green and shortwave infrared (~1500 nm) bands.

    :param granule: The Landsat product ID to load (e.g., LC08_L1TP_...)
    :param data_dir: The directory where the Landsat directory is. Defaults to current directory.
    :return: the normalized difference snow and ice index
    """

    metadata = tools.landsat_metadata(granule, data_dir)
    sens = granule.split('_')[0]

    if sens in ['LT04', 'LT05', 'LE07']:
        swir_band = 5
        green_band = 2
    else:
        swir_band = 6
        green_band = 3

    swir = gu.Raster(Path(data_dir, granule, '_'.join([granule, f"B{swir_band}.TIF"])))
    green = gu.Raster(Path(data_dir, granule, '_'.join([granule, f"B{green_band}.TIF"])))

    swir = toa_reflectance(swir_band, swir, metadata)
    green = toa_reflectance(green_band, green, metadata)

    return (green - swir) / (green + swir)
