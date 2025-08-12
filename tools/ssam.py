from pathlib import Path
from skimage.filters import threshold_otsu
import numpy as np
import geoutils as gu
import xdem
from tqdm import tqdm
from typing import Union
from . import tools


np.seterr(divide='ignore', invalid='ignore')

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


def ndsi(granule: str, data_dir: str = '.', dark_object: bool = False) -> gu.Raster:
    """
    Calculate the normalized difference snow and ice index (NDSI) for a Landsat scene, using the formula:

    NDSI = (Green - SWIR1) / (Green + SWIR1)

    Where Green and SWIR1 are the visible green and shortwave infrared (~1500 nm) bands.

    :param granule: The Landsat product ID to load (e.g., LC08_L1TP_...)
    :param data_dir: The directory where the Landsat directory is. Defaults to current directory.
    :param dark_object: use dark object subtraction on the resulting image.
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

    swir = gu.Raster(Path(data_dir, granule, '_'.join([granule, f"B{swir_band}.TIF"]))).astype(np.float64)
    green = gu.Raster(Path(data_dir, granule, '_'.join([granule, f"B{green_band}.TIF"]))).astype(np.float64)

    swir = to_reflectance(swir_band, swir, metadata)
    green = to_reflectance(green_band, green, metadata)

    if dark_object:
        swir = _dark_object(swir)
        green = _dark_object(green)

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
    zenith = _solar_zenith(metadata)

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


def _solar_zenith(metadata):
    return np.deg2rad(90 - float(metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']))


def _minnaert_const(radiance: gu.Raster, slope: gu.Raster, incidence: gu.Raster) -> float:
    mask = radiance.get_mask()
    y = np.log(radiance[~mask] * np.cos(slope[~mask]))
    x = np.log(np.cos(incidence[~mask]) * np.cos(slope[~mask]))

    kk, b = np.polyfit(x[x < 0], y[x < 0], 1)

    return kk


def corrected_toa(granule: str, bandnum: int, fn_dem: str, data_dir: str ='.') -> gu.Raster:
    """
    Correct TOA reflectance for topography. First computes the corrected radiance using the method
    presented by Ekstrand (1996), then computes the TOA reflectance using the formula:

        toa_refl = pi * radiance * dist**2 / (irradiance * cos(zenith))

    where dist is the Earth-Sun distance in AU, irradiance is the average solar irradiance for the band, and
    zenith is the solar zenith angle.

    :param granule: The Landsat product ID to load (e.g., LC08_L1TP_...)
    :param bandnum: The band number of the Landsat scene to correct
    :param fn_dem: the filename of the DEM to use for topographic correction
    :param data_dir: The directory where the Landsat directory is. Defaults to current directory.
    :return: the topographically corrected TOA reflectance
    """
    metadata = tools.landsat_metadata(granule, data_dir=data_dir)
    sensor = granule.split('_')[0]

    radiance = ekstrand_corr(granule, bandnum, fn_dem, data_dir)

    zenith = _solar_zenith(metadata)
    dist = float(metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['EARTH_SUN_DISTANCE'])

    irradiance = tools.solar_irradiance(bandnum, sensor)

    return np.pi * radiance * dist**2 / (irradiance * np.cos(zenith))


# TODO: implement a cloud mask using the QA bands
def snow_map(granule, fn_dem, fn_outlines, data_dir='.', method: str = 'nir',
             how: str = 'individual',
             return_rast: bool = False) -> Union[None, tuple[gu.Raster, float]]:
    """
    Classify glacier outlines into snow/glacier ice.

    Using the supplied glacier outlines, a threshold between bright/dark pixels is chosen using the Otsu Threshold
    (Otsu, 1979), using either the topographically-corrected NIR band (method = nir), or the topographically-corrected
    albedo (method = albedo).

    :param granule: The Landsat product ID to load (e.g., LC08_L1TP_...)
    :param fn_dem: the filename of the DEM to use for topographic correction
    :param fn_outlines: the filename for the glacier outlines to use.
    :param data_dir: The directory where the Landsat directory is. Defaults to current directory.
    :param method: the method of choosing the threshold. Must be one of [nir, albedo]; default is albedo.
    :param how: How to calculate the threshold: on an individual glacier basis, by glacier complex, or by scene. Must
        be one of [individual, complex, scene]; default is individual.
    :param return_rast: return the raster after saving it.
    :return: None, or the snow map raster and scene-wide threshold (if return_rast is True)
    """
    assert how in ['individual', 'complex', 'scene'], "how must be one of: [individual, complex, scene]"
    assert method in ['nir', 'albedo'], "method must be one of [nir, albedo]"

    do_individual = how in ['individual', 'complex']

    sens = granule.split('_')[0]
    if sens in ['LT04', 'LT05', 'LE07']:
        nir_band = 4
    else:
        nir_band = 5

    # nir = gu.Raster(Path(data_dir, granule, '_'.join(granule, f"B{nir_band}.TIF")))
    snow_index = ndsi(granule, data_dir, dark_object=True)
    vis_mask = ~snow_index.get_mask()

    # get the cloud mask from the qa band
    # is_cloud = cloud_mask(granule, data_dir)

    # apply the ekstrand (1996) correction to the NIR band
    #corrected_nir = ekstrand_corr(granule, nir_band, fn_dem, data_dir=data_dir)
    if method == 'nir':
        thresh_band = corrected_toa(granule, nir_band, fn_dem, data_dir=data_dir)
    else:
        if Path(data_dir, granule + '_albedo.tif').exists():
            thresh_band = gu.Raster(Path(data_dir, granule + '_albedo.tif'))
        else:
            thresh_band = albedo(granule, fn_dem, data_dir=data_dir)
            thresh_band.save(Path(data_dir, granule + '_albedo.tif'))

        corrected_nir = corrected_toa(granule, nir_band, fn_dem, data_dir=data_dir)

    # create a glacier mask and initialize the snow classification raster
    outlines = gu.Vector(fn_outlines)
    if how == 'complex':
        outlines = outlines.dissolve().explode()
        outlines.ds.index = range(len(outlines.ds))

    rasterized = outlines.rasterize(thresh_band)

    snow_class = rasterized.copy(new_array=np.ones_like(rasterized.data))
    snow_class.set_nodata(0)

    # get the unique indices from the glacier mask
    masked = vis_mask & (rasterized > 0)
    unique_inds = np.unique(rasterized[masked])

    snow_class[~masked] = 0

    # TODO: implement some kind of multiprocessing to speed this up?
    if method == 'nir':
        snow = np.logical_and([(snow_index > 0.5).data, (thresh_band > 0.15).data])
    else:
        snow = np.logical_and([(snow_index > 0.5).data, (corrected_nir > 0.15).data])

    glac_snow_ice = np.logical_and([snow, masked.data])
    snow_class.data[~snow] = 0

    if np.count_nonzero(glac_snow_ice) / np.count_nonzero(masked) < 0.1:
        raise ValueError("Not enough valid on-glacier pixels found.")

    if method == 'nir':
        rad = thresh_band.data[glac_snow_ice]
    else:
        rad = thresh_band.data[np.logical_and.reduce([glac_snow_ice,
                                                      (thresh_band >= 0.25).data,
                                                      (thresh_band <= 0.55).data])]

    glob_thresh = threshold_otsu(rad[~rad.mask])

    print(f"Scene-wide reflectance/albedo threshold: {glob_thresh:.3f}")

    # glacier by glacier? dissolve into complexes? treat as a single entity?
    if do_individual:
        for ind in tqdm(unique_inds, desc='Individual thresholds'):
            glac = rasterized.data == ind
            this_snow_ice = np.logical_and(glac, glac_snow_ice)

            if method == 'nir':
                rad = thresh_band[this_snow_ice]
            else:
                rad = thresh_band.data[np.logical_and.reduce([this_snow_ice,
                                                              (thresh_band >= 0.25).data,
                                                              (thresh_band <= 0.55).data])]

            if rad.size > 0:
                if np.count_nonzero(rad > glob_thresh) / rad.size > 0.1:
                    thresh = threshold_otsu(rad[~rad.mask])
                else:
                    thresh = glob_thresh

                snow_class.data[np.logical_and(this_snow_ice, (thresh_band < thresh).data)] = 1
                snow_class.data[np.logical_and(this_snow_ice, (thresh_band >= thresh).data)] = 2
            else:
                snow_class.data[glac] = 0
    else:
        snow_class.data[np.logical_and(glac_snow_ice, (thresh_band > glob_thresh).data)] = 2

    snow_class.save(Path(data_dir, granule + f"_{method}_{how}_snow.tif"))

    if return_rast:
        return snow_class, glob_thresh
    else:
        return None


def cloud_mask(granule: str, data_dir: str ='.') -> gu.Raster:
    """
    Apply the QA_PIXEL cloud mask to a Landsat image.

    :param granule: The Landsat product ID to load (e.g., LC08_L1TP_...)
    :param data_dir: The directory where the Landsat directory is. Defaults to current directory.
    :return: a logical mask with True values where any of the QA cloud pixels has been set.
    """
    qa_band = gu.Raster(Path(data_dir, granule, f"{granule}_QA_PIXEL.TIF"))
    ## cloud flags are bits 1, 2, 3; confidence flags are 8-11
    cloud_flag = int('1111', 2) * np.ones_like(qa_band.data)

    return np.bitwise_and(qa_band, cloud_flag) > 2


def albedo(granule: str, fn_dem: str, data_dir: str = '.') -> gu.Raster:
    """
    Calculate the terrain-corrected albedo for a given Landsat scene, using the following equation, adapted from
    Liang, S. (2000). Remote Sens. Env. 76, 213-238:

        albedo = 0.356 * blue + 0.130 * red + 0.373 * nir + 0.085 * swir1 + 0.027 * swir2 - 0.0018

    Uses Landsat C2 L1 data, applying a dark-object subtraction to each band to help correct for atmospheric effects.

    :param granule: The Landsat product ID to load (e.g., LC08_L1TP_...)
    :param fn_dem: the filename of the DEM to use for topographic correction
    :param data_dir: The directory where the Landsat directory is. Defaults to current directory.
    :return: the terrain-corrected TOA albedo
    """
    # blue, red, nir, swir1, swir2
    coeffs = [0.356, 0.130, 0.373, 0.085, 0.072]

    sens = granule.split('_')[0]

    if sens in ['LT04', 'LT05', 'LE07']:
        bands = [1, 3, 4, 5, 7]
    else:
        bands = [2, 4, 5, 6, 7]

    corrected = []
    for band in bands:
        corrected.append(_dark_object(corrected_toa(granule, band, fn_dem, data_dir=data_dir)))

    return gu.Raster(sum([coeff * band for coeff, band in zip(coeffs, corrected)]) - 0.0018)


def _dark_object(rast: gu.Raster, p: float = 0.05) -> gu.Raster:
    corrected = rast - np.percentile(rast, p)
    corrected[corrected < 0] = 0.01
    return corrected
