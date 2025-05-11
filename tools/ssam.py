from pathlib import Path
from skimage.filters import threshold_otsu
import numpy as np
import geoutils as gu
import xdem
from . import tools


def to_reflectance(band: int, raster: gu.Raster, metadata: dict, is_sr: bool = False) -> gu.Raster:
    """
    Rescale a Landsat band to TOA reflectance, or to surface reflectance.

    :param band: the band number to convert
    :param raster: a gu.Raster of the Landsat band
    :param metadata: a dict of the Landsat metadata
    :param is_sr: whether the metadata is for the L2 surface reflectance product
    :return: the raster, converted to either TOA reflectance, or surface reflectance
    """
    if not is_sr:
        meta = metadata['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']
    else:
        meta = metadata['LANDSAT_METADATA_FILE']['LEVEL2_SURFACE_REFLECTANCE_PARAMETERS']

    raster *= float(meta[f"REFLECTANCE_MULT_BAND_{band}"])
    raster += float(meta[f"REFLECTANCE_ADD_BAND_{band}"])

    return raster


def to_radiance(band: int, raster: gu.Raster, metadata: dict) -> gu.Raster:
    """
    Rescale a Landsat band to TOA radiance.

    :param band: the band number to convert
    :param raster: a gu.Raster of the Landsat band
    :param metadata: a dict of the Landsat metadata
    :return: the raster, converted to TOA radiance
    """
    raster *= float(metadata['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'][f"RADIANCE_MULT_BAND_{band}"])
    raster += float(metadata['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'][f"RADIANCE_ADD_BAND_{band}"])

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

    swir = to_reflectance(swir_band, swir, metadata)
    green = to_reflectance(green_band, green, metadata)

    return (green - swir) / (green + swir)


def ekstrand_corr(granule: str, bandnum: int, fn_dem: str, data_dir: str ='.') -> gu.Raster:
    """
    Correct TOA Radiance for topography, using the method presented by Ekstrand (1996).

    :param granule: The Landsat product ID to load (e.g., LC08_L1TP_...)
    :param bandnum: The band number of the Landsat scene to correct
    :param fn_dem: the filename of the DEM to use for topographic correction
    :param data_dir: The directory where the Landsat directory is. Defaults to current directory.
    :return: the topographically corrected TOA radiance
    """
    metadata = tools.landsat_metadata(granule, data_dir=data_dir)

    band = gu.Raster(Path(data_dir, granule, '_'.join([granule, f"B{bandnum}.TIF"])))
    radiance = to_radiance(bandnum, band, metadata)

    # have to prepare the dem and a hillshade
    # reproject to landsat crs, resolution
    dem = xdem.DEM(fn_dem)
    dem = dem.reproject(band)

    slope = dem.slope()

    # use landsat metadata for solar zenith
    zenith = np.deg2rad(90 - float(metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']))

    # calculate the solar incidence angle
    incidence = compute_incidence_angle(metadata, dem)

    # need to get k (minnaert constant)
    kk = _minnaert_const(radiance, slope, incidence)

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


def _minnaert_const(radiance: gu.Raster, slope: gu.Raster, incidence: gu.Raster) -> float:
    mask = radiance.get_mask()
    y = np.log(radiance[~mask] * np.cos(slope[~mask]))
    x = np.log(np.cos(incidence[~mask]) * np.cos(slope[~mask]))

    kk, b = np.polyfit(x[x < 0], y[x < 0], 1)

    return kk


# TODO: implement a cloud mask using the QA bands
def snow_map(granule, fn_dem, fn_outlines, data_dir='.', how: str = 'individual') -> None:
    """
    Classify glacier outlines into snow/glacier ice, using the approach by Rastner et al. (2019).

    First, a topographic correction is applied to the NIR TOA Radiance (Ekstrand, 1996). Then, using the supplied
    glacier outlines, a threshold between bright/dark pixels is chosen using the Otsu Threshold (Otsu, 1979).

    :param granule: The Landsat product ID to load (e.g., LC08_L1TP_...)
    :param fn_dem: the filename of the DEM to use for topographic correction
    :param fn_outlines: the filename for the glacier outlines to use.
    :param data_dir: The directory where the Landsat directory is. Defaults to current directory.
    :param how: How to calculate the threshold: on an individual glacier basis, by glacier complex, or by scene. Must
        be one of [individual, complex, scene]; default is individual.
    :return:
    """
    assert how in ['individual', 'complex', 'scene'], "how must be one of: [individual, complex, scene]"

    do_individual = how in ['individual', 'complex']

    sens = granule.split('_')[0]
    if sens in ['LT04', 'LT05', 'LE07']:
        nir_band = 4
    else:
        nir_band = 5

    # nir = gu.Raster(Path(data_dir, granule, '_'.join(granule, f"B{nir_band}.TIF")))
    snow_index = ndsi(granule, data_dir)
    vis_mask = ~snow_index.get_mask()

    # get the cloud mask from the qa band
    is_cloud = cloud_mask(granule, data_dir)

    # apply the ekstrand (1996) correction to the NIR band
    corrected_nir = ekstrand_corr(granule, nir_band, fn_dem, data_dir=data_dir)

    # create a glacier mask and initialize the snow classification raster
    outlines = gu.Vector(fn_outlines)
    if how == 'complex':
        outlines = outlines.dissolve().explode()
        outlines.ds.index = range(len(outlines.ds))

    rasterized = outlines.rasterize(corrected_nir)

    snow_class = rasterized.copy()
    snow_class.data[~vis_mask] = 0

    # get the unique indices from the glacier mask
    masked = np.logical_and(vis_mask, rasterized > 0)
    unique_inds = np.unique(rasterized[masked])

    # TODO: implement some kind of multiprocessing to speed this up?
    # glacier by glacier? dissolve into complexes? treat as a single entity?
    if do_individual:
        for ind in unique_inds:
            glac = rasterized == ind

            if np.count_nonzero(is_cloud[glac]) / np.count_nonzero(glac) > 0.1:
                snow_class[glac] = 0
                continue

            glac_snow_ice = np.logical_and(glac, snow_index > 0.5)
            rad = corrected_nir[glac_snow_ice]
            if rad.size > 0:
                thresh = threshold_otsu(rad)

                snow_class[glac] = 1
                snow_class[glac_snow_ice & (corrected_nir > thresh)] = 2
            else:
                snow_class[glac] = 0
    else:
        glac_snow_ice = np.logical_and(masked, snow_index > 0.6)
        snow_class[masked] = 1

        rad = corrected_nir[glac_snow_ice]
        thresh = threshold_otsu(rad)
        print(f"Chosen radiance threshold: {thresh:.2f}")

        snow_class[glac_snow_ice & (corrected_nir > thresh)] = 2

    snow_class.set_nodata(0)
    snow_class.save(Path(data_dir, granule + '_snow.tif'))


def cloud_mask(granule: str, data_dir: str ='.') -> gu.Raster:
    """
    Apply the QA_PIXEL cloud mask to a Landsat image.

    :param granule: The Landsat product ID to load (e.g., LC08_L1TP_...)
    :param data_dir: The directory where the Landsat directory is. Defaults to current directory.
    :return: a logical mask with True values where any of the QA cloud pixels has been set.
    """
    qa_band = gu.Raster(Path(data_dir, granule, f"{granule}_QA_PIXEL.TIF"))
    cloud_flag = int('1100011111', 2) * np.ones_like(qa_band.data)

    return np.bitwise_and(qa_band, cloud_flag) > 512
