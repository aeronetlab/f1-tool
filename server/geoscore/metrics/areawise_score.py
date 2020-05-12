import geojson
import rasterio
import numpy as np
from typing import List
from geoscore.proc import get_geom
from geoscore.io import extract_data
from shapely.geometry import MultiPolygon, Polygon


def areawise_score(gt_file, pred_file, score_fn, area=None, filetype='tif', v: bool = False):
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
    try:
        gt_data, pred_data, read_log = extract_data(gt_file, pred_file, filetype, v=v)
        if v:
            log += read_log
    except Exception as e:
        raise Exception(log + 'Failed to read files\n' + str(e))
    if filetype == 'geojson':
        score, score_log = areawise_vector_score(gt_data['all'], pred_data['all'], score_fn, area, v)
    else:  # tif or any other (default) value
        score, score_log = pixelwise_raster_score(gt_data, pred_data, score_fn, v)
    return score, log + score_log


def pixelwise_raster_score(groundtruth_array, predicted_array, score_fn, v: bool = False):
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

    tp = np.logical_and(groundtruth_array, predicted_array).sum()
    fn = int(groundtruth_array.sum() - tp)
    fp = int(predicted_array.sum() - tp)
    tn = groundtruth_array.size - tp - fp - fn

    score = score_fn(tp, fp, tn, fn)
    if v:
        log = 'True Positive = ' + str(tp) + ', False Negative = ' + str(fn) + ', False Positive = ' + str(fp) + '\n'
    return score, log


def areawise_vector_score(gt: List[Polygon], pred: List[Polygon], score_fn, area=None, v: bool = True):
    """
    Measures pixelwise (actually, area-wise) score, but for vector representation instead of raster.
    Args:
        gt: list of shapely Polygons, represents ground truth
        pred: list of shapely Polygons or Points (according to the 'format' param, represents prediction
        score_fn: a function from .common to calculate the score
        area: the area that bounds the `all area` for true negative calculation, hapely Polygon or MultiPolygon
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

    if area is None:
        # if the area is not specified, we get the GT bounding rectangle as the area
        area = gt_mp.union(pred_mp).convex_hull

    tp = gt_mp.intersection(pred_mp).area
    fp = pred_mp.area - tp
    fn = gt_mp.area - tp
    tn = area.area - tp - fp - fn
    score = score_fn(tp, fp, tn, fn)

    if v:
        log += 'True Positive = ' + str(tp) + ', False Negative = ' + str(fn) + ', False Positive = ' + str(fp) + '\n'

    return score, log
