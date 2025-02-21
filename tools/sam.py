from pathlib import Path
import numpy as np
import geoutils as gu
import xdem
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


def ekstrand_corr(granule: str, bandnum: int, fn_dem, data_dir: str ='.') -> gu.Raster:

    metadata = tools.landsat_metadata(granule, data_dir=data_dir)
    mult = float(metadata['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'][f"RADIANCE_MULT_BAND_{bandnum}"])
    add = float(metadata['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'][f"RADIANCE_MULT_BAND_{bandnum}"])

    band = gu.Raster(Path(data_dir, granule, '_'.join([granule, f"B{bandnum}.TIF"])))
    radiance = band * mult + add

    # have to prepare the dem and a hillshade
    # reproject to landsat crs, resolution
    dem = xdem.DEM(fn_dem)
    dem = dem.reproject(band)

    slope = dem.slope()
    aspect = dem.aspect()

    # use landsat metadata for solar zenith, azimuth
    zenith = np.deg2rad(90 - float(metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']))
    azimuth = np.deg2rad(float(metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_AZIMUTH']))

    # calculate the solar incidence angle
    incidence = compute_incidence_angle(metadata, dem)

    # need to get k (minnaert constant)
    kk = minnaert_const(radiance, slope, incidence)

    return radiance * np.power((np.cos(zenith) / np.cos(incidence)), (kk * np.cos(incidence)))


# compute sun x, y, z rasters from rpc coefficients and dem
def compute_incidence_angle(metadata: dict, dem: xdem.DEM) -> gu.Raster:
    """
    Given a DEM and Landsat metadata file, calculate the solar incidence angle for each pixel of the DEM according
    to the equation:

        cos(incidence) = cos(zenith) * cos(slope) + sin(zenith) * sin(slope) * cos(aspect - azimuth),

    where zenith is the solar zenith angle, azimuth is the solar azimuth angle, slope is the DEM slope, and aspect
    is the DEM aspect.

    :param metadata: the landsat metadata, in dict format
    :param dem: the DEM to use to calculate the zenith angle
    :return: the incidence angle for each pixel of the DEM.
    """
    zenith = np.deg2rad(90 - float(metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']))
    azimuth = np.deg2rad(float(metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_AZIMUTH']))

    slope = np.deg2rad(dem.slope())
    aspect = np.deg2rad(dem.aspect())

    incidence = np.arccos(
        np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(aspect - azimuth)
    )

    return incidence


def minnaert_const(radiance: gu.Raster, slope: gu.Raster, incidence: gu.Raster) -> float:
    mask = radiance.get_mask()
    y = np.log(radiance[~mask] * np.cos(slope[~mask]))
    x = np.log(np.cos(incidence[~mask]) * np.cos(slope[~mask]))

    kk, b = np.polyfit(x[x < 0], y[x < 0], 1)

    return kk
