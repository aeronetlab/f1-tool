import geojson
import rasterio
import numpy as np
from typing import List
from geoscore.proc import get_geom, cut_by_area
from shapely.geometry import Polygon, MultiPolygon


def total_area_score(gt_file, pred_file, area=None, filetype='tif', v:bool = False):
    """

    Args:
        gt_file:
        pred_file:
        area:
        score_fn:
        filetype:
        v:

    Returns:

    """
    log = ''
    if filetype == 'geojson':
        try:
            pred = geojson.load(pred_file)
            pred_polygons = get_geom(pred, 'polygon')
            if v:
                log += "Read predicted geojson, contains " + str(len(pred_polygons)) + " objects \n"
        except Exception as e:
            raise Exception(log + 'Failed to read prediction file as geojson\n' + str(e))

        try:
            gt = geojson.load(gt_file)
            # GT is always as polygons, not points
            gt_polygons = get_geom(gt, 'polygon')
            if v:
                log += "Read groundtruth geojson, contains " + str(len(gt_polygons)) + " polygons \n"
        except Exception as e:
            raise Exception(log + 'Failed to read groundtruth file as geojson\n' + str(e))

        if area:
            try:
                gt_polygons = cut_by_area(gt_polygons, area)
                pred_polygons = cut_by_area(pred_polygons, area)
            except Exception as e:
                log += "Intersection cannot be calculated, ignoring area \n" \
                       + str(e) + '\n'

            log += "Cut vector data by specified area:\n" + \
                   str(len(gt_polygons)) + " groundtruth and " + \
                   str(len(pred_polygons)) + " predicted polygons inside\n"

        score, score_log = calc_total_area_vector(gt_polygons, pred_polygons, v)

    else:  # tif or any other (default) value
        try:
            with rasterio.open(gt_file) as src:
                gt_img = src.read(1)
                if v:
                    log += "Read groundtruth image, size = " + str(gt_img.shape) + "\n"
            with rasterio.open(pred_file) as src:
                # reading into the pre-allocated array guarantees equal sizes
                pred_img = np.empty(gt_img.shape, dtype=src.dtypes[0])
                src.read(1, out=pred_img)
                if v:
                    log += "Read predicted image, size = " + str(src.width) + ', ' + str(src.height) \
                           + ', reshaped to size of GT image \n'
        except Exception as e:
            raise Exception(log + 'Failed to read input file as raster\n' + str(e))

        score, score_log = calc_total_area_raster(gt_img, pred_img, v)

    return score, log + score_log

def calc_total_area_raster(groundtruth_array, predicted_array, v: bool = False):
    """
    Calculates f1-score for 2 equal-sized arrays
    :param groundtruth_array:
    :param predicted_array:
    :param v: is_verbose
    :return: float, f1-score and string, log
    """
    log = ''
    assert groundtruth_array.shape == predicted_array.shape, "Images has different sizes"
    groundtruth_array[groundtruth_array > 0] = 1
    predicted_array[predicted_array > 0] = 1

    gt_area = int(groundtruth_array.sum())
    pred_area = int(predicted_array.sum())

    if gt_area == 0:
        score = 0
    else:
        score = 1 - (abs(pred_area-gt_area))/gt_area
    if v:
        log = f'Ground truth area = {gt_area}, predicted area = {pred_area}. \n'
    return score, log


def calc_total_area_vector(gt: List[Polygon], pred: List[Polygon], v: bool = True):
    """
    Measures pixelwise (actually, area-wise) score, but for vector representation instead of raster.
    Args:
        gt: list of shapely Polygons, represents ground truth
        pred: list of shapely Polygons or Points (according to the 'format' param, represents prediction
        score_fn: a function from .common to calculate the score
        v: is_verbose

    Returns:
        float, f1-score and string, log
    """

    log = ''
    gt_mp = MultiPolygon(gt)
    pred_mp = MultiPolygon(pred)

    # try making polygons valid
    gt_mp = gt_mp.buffer(0)
    pred_mp = pred_mp.buffer(0)

    gt_area = gt_mp.area
    pred_area = pred_mp.area

    if gt_area == 0:
        score = 0
    else:
        score = 1 - abs(pred_area-gt_area)/gt_area

    if v:
        log = f'Ground truth area = {gt_area}, predicted area = {pred_area}. \n'

    return score, log
