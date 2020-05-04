import rtree
from typing import List

from shapely.wkb import dumps, loads
from shapely.geometry import Polygon
import geojson

from geoscore.proc import get_geom, cut_by_area


def objectwise_point_score(gt_file, pred_file, area, score_fn, v: bool = True):
    """

    Args:
        gt_file:
        pred_file:
        area:
        v:

    Returns:

    """
    log = ''
    try:
        gt = geojson.load(gt_file)
        # GT is always as polygons, not points
        gt_polygons = get_geom(gt, 'polygon')
        if v:
            log += "Read groundtruth geojson, contains " + str(len(gt_polygons)) + " polygons \n"
    except Exception as e:
        raise Exception(log + 'Failed to read geojson groundtruth file\n' + str(e))

    try:
        pred = geojson.load(pred_file)
        pred_geom = get_geom(pred, 'point')
        if v:
            log += "Read predicted geojson, contains " + str(len(pred_geom)) + " objects \n"
    except Exception as e:
        raise Exception(log + 'Failed to read geojson prediction file\n' + str(e))

    if area:
        try:
            gt_polygons = cut_by_area(gt_polygons, area)
            pred_geom = cut_by_area(pred_geom, area)
        except Exception as e:
            log += "Intersection cannot be calculated, ignoring area \n" \
                   + str(e) + '\n'

        log += "Cut vector data by specified area:\n" + \
               str(len(gt_polygons)) + " groundtruth and " + \
               str(len(pred_geom)) + " predicted points inside\n"

    try:
        score, score_log = calc_f1_point(gt_polygons, pred_geom, score_fn, v=v)
    except Exception as e:
        raise Exception(log + 'Error while calculating objectwise f1-score in point format\n' + str(e))

    return score, log + score_log


def calc_f1_point(gt: List[Polygon], pred, score_fn, v: bool = True):
    """
    Measures objectwise f1-score for two sets of polygons.
    The algorithm description can be found on
    https://medium.com/the-downlinq/the-spacenet-metric-612183cc2ddb

    True Positive counts when the prediction point lies within the polygon of GT

    :param gt: list of shapely Polygons, represents ground truth;
    :param pred: list of shapely Points;
    :param format: 'vector' or 'point', means format of prediction and corresponding variant of algorithm;
    :param v: is_verbose
    :return: float, f1-score and string, log
    """
    log = ''
    groundtruth_rtree_index = rtree.index.Index()

    # for some reason builtin pickling doesn't work
    for i, polygon in enumerate(gt):
        groundtruth_rtree_index.insert(
            i, polygon.bounds, dumps(polygon)
        )

    tp = sum(map(_lies_within_rtree,
                (point for point in pred),
                [groundtruth_rtree_index] * len(pred)))
    fp = len(pred) - tp
    fn = len(gt) - tp

    # to avoid zero-division
    tn = 1  # One object of background class
    score = score_fn(tp, fp, tn, fn)
    if v:
        log += 'True Positive = ' + str(tp) + ', False Negative = ' + str(fn) + ', False Positive = ' + str(fp) + '\n'

    return score, log


def _lies_within_rtree(point, groundtruth_rtree_index):
    """ Searches whether there is an indexed polygon which contains the point
    and deletes the match if it is found

    Args:
        point: point to be matched
        groundtruth_rtree_index: rtree index

    Returns:
        True if match found, False otherwise
    """
    candidate_items = [
        candidate_item
        for candidate_item
        in groundtruth_rtree_index.intersection(
            (point.x, point.y), objects=True
        )
    ]
    for item in candidate_items:
        candidate = loads(item.get_object(loads))
        if candidate.contains(point):
            groundtruth_rtree_index.delete(item.id, candidate.bounds)
            return True
    return False