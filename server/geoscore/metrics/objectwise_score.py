import rtree
from typing import List

from shapely.wkb import dumps, loads
from shapely.geometry import Polygon
import geojson

from geoscore.proc import get_geom, cut_by_area


def objectwise_score(gt_file, pred_file, area, score_fn, iou=0.5, v: bool = True):
    """

    Args:
        gt_file:
        pred_file:
        area:
        score_fn:
        iou:
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
        pred_geom = get_geom(pred, 'polygon')
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
               str(len(pred_geom)) + " predicted polygons inside\n"

    try:
        score, score_log = calc_object_score(gt_polygons, pred_geom, score_fn, iou=iou, v=v)
    except Exception as e:
        raise Exception(log + 'Error while calculating objectwise f1-score \n' + str(e))

    return score, log + score_log


def calc_object_score(gt: List[Polygon], pred, score_fn, iou=0.5, v: bool=True):
    """
    Measures objectwise f1-score for two sets of polygons.
    The algorithm description can be found on
    https://medium.com/the-downlinq/the-spacenet-metric-612183cc2ddb

    Args:
        gt: list of shapely Polygons, represents ground truth
        pred: list of shapely Polygons, represents prediction
        iou:
        v: is_verbose

    Returns:
        float, f1-score and string, log

    """

    log = ''
    groundtruth_rtree_index = rtree.index.Index()

    # for some reason builtin pickling doesn't work
    for i, polygon in enumerate(gt):
        groundtruth_rtree_index.insert(
            i, polygon.bounds, dumps(polygon)
        )

    tp = sum(map(_has_match_rtree,
                 (dumps(polygon) for polygon in pred),
                 [iou]*len(pred),
                 [groundtruth_rtree_index]*len(pred)))

    fp = len(pred) - tp
    fn = len(gt) - tp
    # to avoid zero-division
    tn = 1  # One object of background class
    score = score_fn(tp, fp, tn, fn)
    if v:
        log += 'True Positive = ' + str(tp) + ', False Negative = ' + str(fn) + ', False Positive = ' + str(fp) + '\n'

    return score, log


def _has_match_rtree(polygon_serialized, iou_threshold, groundtruth_rtree_index):
    """ Compares the polygon with the rtree index whether it has a matching indexed polygon,
    and deletes the match if it is found
    :param polygon_serialized: polygon to be matched
    :param iou_threshold: minimum IoU that is required for the polygon to be considered positive example
    :param groundtruth_rtree_index: rtree index
    :return: True if match found, False otherwise
    """
    polygon = loads(polygon_serialized)
    best_iou = 0
    best_item = None
    candidate_items = [
        candidate_item
        for candidate_item
        in groundtruth_rtree_index.intersection(
            polygon.bounds, objects=True
        )
    ]

    for item in candidate_items:
        candidate = loads(item.get_object(loads))
        metric = iou(polygon, candidate)
        if metric > best_iou:
            best_iou = metric
            best_item = item

    if best_iou > iou_threshold and best_item is not None:
        groundtruth_rtree_index.delete(best_item.id,
                                       best_item.bbox)
        return True
    else:
        return False


def iou(polygon1: Polygon, polygon2: Polygon):
    # buffer(0) may be used to “tidy” a polygon
    # works good for self-intersections like zero-width details
    # http://toblerity.org/shapely/shapely.geometry.html
    poly1_fixed = polygon1.buffer(0)
    poly2_fixed = polygon2.buffer(0)
    return poly1_fixed.buffer(0).intersection(poly2_fixed).area / poly1_fixed.union(poly2_fixed).area

