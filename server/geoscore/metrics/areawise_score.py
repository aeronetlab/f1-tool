import geojson
import rasterio
import numpy as np
from typing import List
from geoscore.proc import get_geom, cut_by_area
from shapely.geometry import MultiPolygon, Polygon
import aeronet.dataset as ds

from time import time


def areawise_score(gt, pred, score_fn, area=None, v: bool = False):
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
    #try:
    #pred = geojson.load(pred_file)
    pred_polygons = get_geom(pred, 'polygon')
    if v:
        log += "Read predicted geojson, contains " + str(len(pred_polygons)) + " objects \n"
    #except Exception as e:
    #    raise Exception(log + 'Failed to read prediction file as geojson\n' + str(e))

    try:
        #gt = geojson.load(gt_file)
        # GT is always as polygons, not points
        gt_polygons = get_geom(gt, 'polygon')
        if v:
            log += "Read groundtruth geojson, contains " + str(len(gt_polygons)) + " polygons \n"
    except Exception as e:
        raise Exception(log + 'Failed to read groundtruth file as geojson\n' + str(e))


    # For techinspec we ignore area cutting because the areas already match. But we need the area to calculate
    # accuracy properly, so we cannot skip the area argument
    if area:
        try:
            gt_polygons = cut_by_area(gt_polygons, area, True)
            pred_polygons = cut_by_area(pred_polygons, area, True)
        except Exception as e:
            log += "Intersection cannot be calculated, ignoring area \n" \
                   + str(e) + '\n'

        log += "Cut vector data by specified area:\n" + \
               str(len(gt_polygons)) + " groundtruth and " + \
               str(len(pred_polygons)) + " predicted polygons inside\n"

    score, score_log = areawise_vector_score(gt_polygons, pred_polygons, score_fn, area, v)

    return score, log + score_log


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
    print('begin')
    t = time()

    gt_fc = ds.FeatureCollection([ds.Feature(poly) for poly in gt if poly.buffer(0).is_valid])
    pred_fc = ds.FeatureCollection([ds.Feature(poly) for poly in pred if poly.buffer(0).is_valid])

    print(f'fc: {time() - t}')
    t = time()

    intersection = 0
    for feat in gt_fc:
        pred_inter = pred_fc.intersection(feat)
        for pred_f in pred_inter:
            intersection += pred_f.apply(lambda x: x.intersection(feat.shape)).area

    print(f'intersection: {time() - t}')
    t = time()

    # We assume that features do not intersect. Bold assumption, but will do for now!
    pred_area = np.sum([feat.shape.area for feat in pred_fc])
    gt_area = np.sum([feat.shape.area for feat in gt_fc])
    print(f'mo: {time() - t}')
    t = time()

    tp = intersection
    fp = pred_area - tp
    fn = gt_area - tp

    print(f'area: {time() - t}')
    t = time()

    if area is None:
        # if the area is not specified, we get the GT bounding rectangle as the area
        lon1, lat1, lon2, lat2 = gt_fc.index.bounds
        lon1_pred, lat1_pred, lon2_pred, lat2_pred = pred_fc.index.bounds

        lon1 = min(lon1, lon1_pred)
        lat1 = min(lat1, lat1_pred)
        lon2 = max(lon2, lon2_pred)
        lat2 = max(lat2, lat2_pred)

        area = Polygon([[lon1, lat1], [lon1, lat2], [lon2, lat2], [lon2, lat1], [lon1, lat1]])
        print(area)

    else:
        area = MultiPolygon(area)
    print(f'union: {time() - t}')

    tn = area.area - tp - fp - fn
    score = score_fn(tp, fp, tn, fn)

    if v:
        log += 'True Positive = ' + str(tp) + \
               ', False Negative = ' + str(fn) + \
               ', False Positive = ' + str(fp) + \
               ', True Negative = ' + str(tn) + '\n'

    return score, log
