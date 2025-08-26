from pathlib import Path
import warnings
from skimage.filters import threshold_otsu
from scipy import stats
import numpy as np
import geoutils as gu
from shapely.geometry import LineString
import xdem
from tqdm import tqdm
from typing import Union
from collections.abc import Callable
from . import tools


np.seterr(divide='ignore', invalid='ignore')

warnings.filterwarnings('ignore', message="Warning: 'partition' will ignore the 'mask' of the MaskedArray.",
                        category=UserWarning)
warnings.filterwarnings('ignore', message="Unmasked values equal to the nodata value found in data array.",
                        category=UserWarning)

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


def brightness_temp(granule: str, data_dir: Union[str, Path] = '.') -> gu.Raster:
    """
    Rescale a Landsat thermal band to Brightness Temperature.

    :param granule: The Landsat product ID to load (e.g., LC08_L1TP_...)
    :param data_dir: The directory where the Landsat directory is. Defaults to current directory.
    :return: the brightness temperature
    """
    metadata = tools.landsat_metadata(granule, data_dir)
    sens = granule.split('_')[0]

    if sens in ['LT04', 'LT05']:
        thermal_band = 6
    elif sens in ['LE07']:
        thermal_band = '6_VCID_1'
    else:
        thermal_band = 10

    radiance = to_radiance(thermal_band,
                           gu.Raster(Path(data_dir, granule, '_'.join([granule, f"B{thermal_band}.TIF"]))),
                           metadata)

    k1 = float(metadata['LANDSAT_METADATA_FILE']['LEVEL1_THERMAL_CONSTANTS'][f"K1_CONSTANT_BAND_{thermal_band}"])
    k2 = float(metadata['LANDSAT_METADATA_FILE']['LEVEL1_THERMAL_CONSTANTS'][f"K2_CONSTANT_BAND_{thermal_band}"])

    return k2 / np.log((k1 / radiance) + 1)


def ndsi(granule: str, data_dir: str = '.', dark_object: bool = False, is_sr: bool = False) -> gu.Raster:
    """
    Calculate the normalized difference snow and ice index (NDSI) for a Landsat scene, using the formula:

    NDSI = (Green - SWIR1) / (Green + SWIR1)

    Where Green and SWIR1 are the visible green and shortwave infrared (~1500 nm) bands.

    :param granule: The Landsat product ID to load (e.g., LC08_L1TP_...)
    :param data_dir: The directory where the Landsat directory is. Defaults to current directory.
    :param dark_object: use dark object subtraction on the resulting image.
    :param is_sr: whether the metadata is for the L2 surface reflectance product
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

    if is_sr:
        swir = gu.Raster(Path(data_dir, granule, '_'.join([granule, f"SR_B{swir_band}.TIF"]))).astype(np.float64)
        green = gu.Raster(Path(data_dir, granule, '_'.join([granule, f"SR_B{green_band}.TIF"]))).astype(np.float64)
    else:
        swir = gu.Raster(Path(data_dir, granule, '_'.join([granule, f"B{swir_band}.TIF"]))).astype(np.float64)
        green = gu.Raster(Path(data_dir, granule, '_'.join([granule, f"B{green_band}.TIF"]))).astype(np.float64)

    swir = to_reflectance(swir_band, swir, metadata, is_sr=is_sr)
    green = to_reflectance(green_band, green, metadata, is_sr=is_sr)

    if dark_object:
        swir = _dark_object(swir)
        green = _dark_object(green)

    return (green - swir) / (green + swir)


def ndwi(granule: str, data_dir: str = '.', dark_object: bool = False, is_sr: bool = False) -> gu.Raster:
    """
    Calculate the normalized difference water index (NDWI) for a Landsat scene, using the formula:

    NDSI = (Green - NIR) / (Green + NIR)

    Where Green and NIR are the visible green and near infrared bands.

    :param granule: The Landsat product ID to load (e.g., LC08_L1TP_...)
    :param data_dir: The directory where the Landsat directory is. Defaults to current directory.
    :param dark_object: use dark object subtraction on the resulting image.
    :param is_sr: whether the metadata is for the L2 surface reflectance product
    :return: the normalized difference snow and ice index
    """

    metadata = tools.landsat_metadata(granule, data_dir)
    sens = granule.split('_')[0]

    if sens in ['LT04', 'LT05', 'LE07']:
        nir_band = 4
        green_band = 2
    else:
        nir_band = 5
        green_band = 3

    if is_sr:
        nir = gu.Raster(Path(data_dir, granule, '_'.join([granule, f"SR_B{nir_band}.TIF"]))).astype(np.float64)
        green = gu.Raster(Path(data_dir, granule, '_'.join([granule, f"SR_B{green_band}.TIF"]))).astype(np.float64)
    else:
        nir = gu.Raster(Path(data_dir, granule, '_'.join([granule, f"B{nir_band}.TIF"]))).astype(np.float64)
        green = gu.Raster(Path(data_dir, granule, '_'.join([granule, f"B{green_band}.TIF"]))).astype(np.float64)

    nir = to_reflectance(nir_band, nir, metadata, is_sr=is_sr)
    green = to_reflectance(green_band, green, metadata, is_sr=is_sr)

    if dark_object:
        nir = _dark_object(nir)
        green = _dark_object(green)

    return (green - nir) / (green + nir)



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
    cos_inc = np.cos(incidence)
    cos_inc[cos_inc <= 1e-6] = 1e-6

    # need to get k (minnaert constant)
    kk = _minnaert_const(radiance, slope, incidence)

    return radiance * np.power((np.cos(zenith) / cos_inc), (kk * cos_inc))


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
             return_rast: bool = False,
             use_elevation: bool = False) -> Union[None, tuple[gu.Raster, float]]:
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
    :param use_elevation: use the DEM to calculate the albedo threshold based on the maxmimum slope of the
        albedo-elevation profile for on-glacier pixels. Only works if how='albedo'.
    :return: None, or the snow map raster and scene-wide threshold (if return_rast is True)
    """
    assert how in ['individual', 'complex', 'scene'], "how must be one of: [individual, complex, scene]"
    assert method in ['nir', 'albedo'], "method must be one of [nir, albedo]"

    do_individual = how in ['individual', 'complex']

    sens = granule.split('_')[0]
    if sens in ['LT04', 'LT05', 'LE07']:
        nir_band = 4
        thermal_band = 6
    else:
        nir_band = 5
        thermal_band = 10

    # nir = gu.Raster(Path(data_dir, granule, '_'.join(granule, f"B{nir_band}.TIF")))
    if Path(data_dir, granule + '_ndsi.tif').exists():
        snow_index = gu.Raster(Path(data_dir, granule + '_ndsi.tif'))
    else:
        snow_index = ndsi(granule, data_dir, dark_object=True)
        snow_index.save(Path(data_dir, granule + '_ndsi.tif'))

    vis_mask = ~snow_index.get_mask()

    if Path(data_dir, granule + '_ndwi.tif').exists():
        water_index = gu.Raster(Path(data_dir, granule + '_ndwi.tif'))
    else:
        water_index = ndwi(granule, data_dir, dark_object=True)
        water_index.save(Path(data_dir, granule + '_ndwi.tif'))

    # get the cloud mask from the qa band
    # is_cloud = cloud_mask(granule, data_dir)
    if Path(data_dir, granule + '_temp.tif').exists():
        b_temp = gu.Raster(Path(data_dir, granule + '_temp.tif'))
    else:
        b_temp = brightness_temp(granule, data_dir)
        b_temp.save(Path(data_dir, granule + '_temp.tif'))

    if Path(data_dir, granule + '_cloud.tif').exists():
        cloud = gu.Raster(Path(data_dir, granule + '_cloud.tif')) != 0
    else:
        cloud = irish_cloud_filter(granule, data_dir)
        cloud.save(Path(data_dir, granule + '_cloud.tif'))

    if Path(data_dir, granule + '_shadow.tif').exists():
        shadow = gu.Raster(Path(data_dir, granule + '_shadow.tif')) != 0
    else:
        shadow = hillshade_shadow(snow_index, fn_dem, tools.landsat_metadata(granule, data_dir))
        shadow.save(Path(data_dir, granule + '_shadow.tif'))

    # apply the ekstrand (1996) correction to the NIR band
    #corrected_nir = ekstrand_corr(granule, nir_band, fn_dem, data_dir=data_dir)
    if method == 'nir':
        thresh_band = corrected_toa(granule, nir_band, fn_dem, data_dir=data_dir)
        thresh_band = _dark_object(thresh_band)
    else:
        if Path(data_dir, granule + '_albedo.tif').exists():
            thresh_band = gu.Raster(Path(data_dir, granule + '_albedo.tif'))
        else:
            thresh_band = albedo(granule, fn_dem, data_dir=data_dir)
            thresh_band.save(Path(data_dir, granule + '_albedo.tif'))

        if Path(data_dir, granule + '_nir.tif').exists():
            corrected_nir = gu.Raster(Path(data_dir, granule + '_nir.tif'))
        else:
            corrected_nir = corrected_toa(granule, nir_band, fn_dem, data_dir=data_dir)
            corrected_nir = _dark_object(corrected_nir)
            corrected_nir.save(Path(data_dir, granule + '_nir.tif'))

    # create a glacier mask and initialize the snow classification raster
    outlines = gu.Vector(fn_outlines)
    if how == 'complex':
        outlines = outlines.dissolve().explode()
        outlines.ds.index = range(len(outlines.ds))

    rasterized = outlines.rasterize(thresh_band)

    snow_class = rasterized.copy(new_array=2 * np.ones_like(rasterized.data))
    snow_class.set_nodata(0)

    # get the unique indices from the glacier mask
    masked = vis_mask & (rasterized > 0)
    unique_inds = np.unique(rasterized[masked])

    snow_class[~masked] = 0

    # TODO: implement some kind of multiprocessing to speed this up?
    if method == 'nir':
        snow = np.logical_and.reduce([(snow_index > 0.4).data, (water_index < 0.35).data,
                                      (b_temp < 290).data, (~cloud).data])
    else:
        snow = np.logical_and.reduce([(snow_index > 0.4).data, (water_index < 0.35).data,
                                      (b_temp < 290).data, (~cloud).data])

    glac_snow_ice = np.logical_and(snow, masked.data)

    not_shadow = np.logical_and((~shadow).data, glac_snow_ice)
    glac_shadow = np.logical_and(shadow.data, glac_snow_ice)

    snow_class.data[~snow] = 0

    if np.count_nonzero(glac_snow_ice) / np.count_nonzero(masked) < 0.1:
        raise ValueError("Not enough valid on-glacier pixels found.")

    if method == 'nir':
        rad = thresh_band.data[not_shadow]
        glob_thresh = threshold_otsu(rad[~rad.mask])
    else:
        if use_elevation:
            glob_thresh = _albedo_thresh(thresh_band, gu.Raster(fn_dem), not_shadow)
        else:
            rad = thresh_band.data[np.logical_and.reduce([not_shadow,
                                                          (thresh_band >= 0.25).data,
                                                          (thresh_band <= 0.55).data])]
            glob_thresh = threshold_otsu(rad[~rad.mask])

    print(f"Scene-wide reflectance/albedo threshold: {glob_thresh:.3f}")

    # glacier by glacier? dissolve into complexes? treat as a single entity?
    if do_individual:
        for ind in tqdm(unique_inds, desc='Individual thresholds'):
            glac = rasterized.data == ind
            this_snow_ice = np.logical_and.reduce([glac, glac_snow_ice, not_shadow])

            if method == 'nir':
                rad = thresh_band[this_snow_ice]
            else:
                rad = thresh_band.data[np.logical_and.reduce([this_snow_ice,
                                                              (thresh_band >= 0.25).data,
                                                              (thresh_band <= 0.55).data])]

            if rad.size > 0:
                if np.count_nonzero(rad > glob_thresh) / rad.size > 0.1:
                    if method == 'nir' or not use_elevation:
                        thresh = threshold_otsu(rad[~rad.mask])
                    else:
                        thresh = _albedo_thresh(thresh_band, gu.Raster(fn_dem), this_snow_ice)
                else:
                    thresh = glob_thresh

                snow_class.data[np.logical_and.reduce([glac, glac_snow_ice, (thresh_band < thresh).data])] = 1
                snow_class.data[np.logical_and.reduce([glac, glac_snow_ice, (thresh_band >= thresh).data])] = 2

                if use_elevation:
                    snow_class = _elevation_threshold(snow_class, np.logical_and(glac, glac_shadow),
                                                      np.logical_and(glac, not_shadow), gu.Raster(fn_dem))

            else:
                snow_class.data[glac] = 0
    else:
        snow_class.data[np.logical_and(not_shadow, (thresh_band < glob_thresh).data)] = 1

    if use_elevation:
        snow_class = _elevation_threshold(snow_class, glac_shadow, not_shadow, gu.Raster(fn_dem))

    snow_class.save(Path(data_dir, granule + f"_{method}_{how}_snow.tif"))

    if return_rast:
        return snow_class, glob_thresh
    else:
        return None


def _albedo_thresh(albedo: gu.Raster, dem: gu.Raster, glacmask: gu.Mask,
                   bin_size: float = 50., statistic: Union[str, Callable] = 'median') -> float:
    """
    Calculates the albedo threshold value based on the steepest slope in the albedo-elevation profile.

    :param albedo:
    :param dem: a DEM
    :param glacmask: a mask of on-glacier pixels
    :param bin_size: the size of the elevation bins to use
    :param statistic: the statistic to compute, as passed to scipy.stats.binned_statistic
    :return: the computed threshold value
    """

    if not dem.shape == albedo.shape:
        dem = dem.reproject(albedo)

    onglac_albedo = albedo[glacmask]
    onglac_el = dem[glacmask]

    bins = np.unique(bin_size * np.floor(onglac_el / bin_size)).data

    is_masked = np.logical_or(onglac_albedo.mask, onglac_el.mask)

    bin_stat, bins, binned = stats.binned_statistic(onglac_el[~is_masked], onglac_albedo[~is_masked],
                                                    statistic=statistic, bins=bins)

    slope = np.diff(bin_stat, prepend=bin_stat[0])

    smax = np.argmax(slope[np.logical_and(bin_stat > 0.25, bin_stat < 0.55)])

    return bin_stat[np.logical_and(bin_stat > 0.25, bin_stat < 0.55)][smax]


def _elevation_threshold(snow_class, shadow, not_shadow, dem, bin_size=50.):

    if not dem.shape == snow_class.shape:
        dem = dem.reproject(snow_class)

    glac_ice = np.logical_and(not_shadow, (snow_class == 1).data)
    glac_snow = np.logical_and(not_shadow, (snow_class == 2).data)

    snow_el = dem[glac_snow]
    ice_el = dem[glac_ice]

    onglac_el = dem[not_shadow]
    bins = np.unique(bin_size * np.floor(onglac_el / bin_size)).data

    ice_pdf, bin_edges = np.histogram(ice_el, bins=bins, density=True)
    snow_pdf, bin_edges = np.histogram(snow_el, bins=bins, density=True)

    ice_cdf = np.insert(np.cumsum(ice_pdf * np.diff(bin_edges)), 0, 0, axis=0)
    snow_cdf = np.insert(np.cumsum(snow_pdf * np.diff(bin_edges)), 0, 0, axis=0)

    _ice = LineString(zip(bins, 1 - ice_cdf))
    _sno = LineString(zip(bins, snow_cdf))

    el_thresh = _ice.intersection(_sno).x

    snow_class.data[np.logical_and(shadow, (dem >= el_thresh).data)] = 2
    snow_class.data[np.logical_and(shadow, (dem < el_thresh).data)] = 1

    return snow_class


def hillshade_shadow(img: gu.Raster, fn_dem: Union[str, Path], metadata: dict, thresh: int = 70) -> gu.Raster:
    """
    Return a mask of shadows computed from a DEM hillshade.

    :param img: the Landsat image (or raster) to compute the shadow mask for
    :param fn_dem: the filename of the DEM to use for computing the hillshades
    :param metadata: the landsat metadata, in dict format
    :param thresh: the hillshade value to use to determine shadows
    :return: the mask of shadows based on the hillshade
    """

    # load the DEM and reproject it to the image
    dem = xdem.DEM(fn_dem).reproject(img)

    # get the azimuth and elevation angles from the metadata
    azimuth = float(metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_AZIMUTH'])
    elevation = float(metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION'])

    # have to subtract azimuth from 270 (mod 360) to get the same output as gdal for some reason
    hillshade = dem.hillshade(azimuth=(270 - azimuth) % 360, altitude=elevation)

    return hillshade < thresh


def cloud_mask(granule: str, data_dir: str ='.') -> gu.Raster:
    """
    Apply the QA_PIXEL cloud mask to a Landsat image.

    :param granule: The Landsat product ID to load (e.g., LC08_L1TP_...)
    :param data_dir: The directory where the Landsat directory is. Defaults to current directory.
    :return: a logical mask with True values where any of the QA cloud pixels has been set.
    """
    qa_band = gu.Raster(Path(data_dir, granule, f"{granule}_QA_PIXEL.TIF"))
    ## cloud flags are bits 1, 2, 3; confidence flags are 8-11
    cloud_flag = int('11110', 2) * np.ones_like(qa_band.data)

    return np.bitwise_and(qa_band, cloud_flag) > 2


def irish_cloud_filter(granule: str, data_dir: Union[str, Path] = '.') -> gu.Raster:
    """
    An implementation of the cloud masking procedure described by Irish (2000):

        Irish RR (2000) Landsat 7 Automatic Cloud Cover Assessment. SPIE 4049, 348–355. doi:10.1117/12.410358.

    :param granule: The Landsat product ID to load (e.g., LC08_L1TP_...)
    :param data_dir: The directory where the Landsat directory is. Defaults to current directory.
    :return: a logical mask with True values where cloud has been identified
    """
    meta = tools.landsat_metadata(granule, data_dir)

    sens = granule.split('_')[0]
    if sens in ['LT04', 'LT05', 'LE07']:
        green_band = 2
        red_band = 3
        nir_band = 4
        swir_band = 5
        if sens in ['LT04', 'LT05']:
            therm_band = 6
        else:
            therm_band = '6_VCID_1'
    else:
        green_band = 3
        red_band = 4
        nir_band = 5
        swir_band = 6
        therm_band = 10

    therm_rad = to_radiance(therm_band,
                            gu.Raster(Path(data_dir, granule, '_'.join([granule, f"B{therm_band}.TIF"]))), meta)

    b_temp = brightness_temp(granule, data_dir)

    green = to_reflectance(green_band,
                           gu.Raster(Path(data_dir, granule, '_'.join([granule, f"B{green_band}.TIF"]))),
                           meta)

    red = to_reflectance(red_band,
                         gu.Raster(Path(data_dir, granule, '_'.join([granule, f"B{red_band}.TIF"]))),
                         meta)

    nir = to_reflectance(nir_band,
                         gu.Raster(Path(data_dir, granule, '_'.join([granule, f"B{nir_band}.TIF"]))),
                         meta)

    swir = to_reflectance(swir_band,
                          gu.Raster(Path(data_dir, granule, '_'.join([granule, f"B{swir_band}.TIF"]))),
                          meta)

    composite = (1 - swir) * therm_rad
    snow_index = ndsi(granule, data_dir)

    first_mask = np.logical_and.reduce([(red > 0.08).data, (snow_index < 0.7).data, (b_temp < 300).data,
                                        (composite < 7).data, (nir / red < 2).data, (nir / green < 2).data,
                                        (nir / swir > 1).data])

    warm = np.logical_and(first_mask, (composite > 5.5).data)
    cold = np.logical_and(first_mask, (composite <= 5.5).data)

    ambiguous = np.logical_and.reduce([(composite >= 6).data, (nir / red >= 2).data,
                                       (nir / green >= 2).data, (nir / swir <= 1).data])

    snow_pct = 100 * np.count_nonzero(snow_index > 0.7) / np.count_nonzero(~snow_index.get_mask())

    if snow_pct < 1:
        cloud = first_mask
    else:
        cloud = cold
        ambiguous = np.logical_or(warm, ambiguous)

    cc_pct = 100 * np.count_nonzero(cloud) / np.count_nonzero(~swir.get_mask())
    desert_index = np.count_nonzero((nir / swir <= 1) == 1) / np.count_nonzero(~swir.get_mask())

    cloud_temp = b_temp[cloud]

    if any([desert_index > 0.5, cc_pct > 0.4, cloud_temp.mean() < 295]):
        low = np.percentile(cloud_temp, 83.5)
        high = np.percentile(cloud_temp, 97.5)

        skew = stats.skew(cloud_temp)
        if skew > 0:
            shift = max(1, skew * cloud_temp.std())
            high = max(np.percentile(cloud_temp, 98.75), high + shift)
            low = low + shift

        cloud = np.logical_or(cloud, np.logical_and(ambiguous, (b_temp <= low).data))

    return swir.copy(new_array=cloud)


def albedo(granule: str, fn_dem: str, data_dir: str = '.', is_sr: bool = False) -> gu.Raster:
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
        if not is_sr:
            corrected.append(_dark_object(corrected_toa(granule, band, fn_dem, data_dir=data_dir)))
        else:
            rast = gu.Raster(Path(data_dir, granule, '_'.join([granule, f"SR_B{band}.TIF"]))).astype(np.float64)
            corrected.append(to_reflectance(band, rast, tools.landsat_metadata(granule, data_dir), is_sr=True))

    return gu.Raster(sum([coeff * band for coeff, band in zip(coeffs, corrected)]) - 0.0018)


def _dark_object(rast: gu.Raster, p: float = 0.05) -> gu.Raster:
    corrected = rast - np.percentile(rast, p)
    corrected[(corrected <= 0) & ~corrected.get_mask()] = 0.001
    return corrected
