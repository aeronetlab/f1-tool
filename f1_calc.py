import rtree
import numpy as np

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.wkb import dumps, loads

from sklearn.metrics import f1_score
from typing import List
from multiprocessing import Pool


IOU_THRESHOLD = 0.5

def pixelwise_f1_score(groundtruth_array, predicted_array, v: bool=False, echo=print):

    log = {}
    assert groundtruth_array.shape == predicted_array.shape, "Images has different sizes"
    groundtruth_array[groundtruth_array > 0] = 1
    groundtruth_binary = groundtruth_array.flatten()
    predicted_array[predicted_array > 0] = 1
    predicted_binary = predicted_array.flatten()

    if v:
        tp = np.logical_and(groundtruth_array, predicted_array).sum()
        fn = int(groundtruth_array.sum() - tp)
        fp = int(predicted_array.sum() - tp)
        log = {'TP': str(tp), 'FN': str(fn), 'FP': str(fp)}
        print (2*tp/(2*tp + fn + fp))
    return f1_score(groundtruth_binary, predicted_binary), log


def objectwise_f1_score(groundtruth_polygons: List[Polygon],
                        predicted_polygons: List[Polygon],
                        v: bool=True,
                        multiproc: bool=True):
    """
    Measures objectwise f1-score for two sets of polygons.
    The algorithm description can be found on
    https://medium.com/the-downlinq/the-spacenet-metric-612183cc2ddb

    :param groundtruth_polygons: list of shapely Polygons;
    we suppose that these polygons are not intersected
    :param predicted_polygons: list of shapely Polygons with
    the same size as groundtruth_polygons;
    we suppose that these polygons are not intersected
    :param method: 'rtree' or 'basic'
    :param v: is_verbose
    :param echo: function for printing
    :param multiproc: nables/disables multiprocessing
    :return: float, f1-score
    """
    log = {}

    global global_groundtruth_rtree_index
        # for some reason builtin pickling doesn't work
    for i, polygon in enumerate(groundtruth_polygons):
        global_groundtruth_rtree_index.insert(
            i, polygon.bounds, dumps(polygon)
        )
    if multiproc:
        tp = sum(Pool().map(_has_match_rtree, (dumps(polygon) for polygon in predicted_polygons)))
    else:
        tp = sum(map(_has_match_rtree, (dumps(polygon) for polygon in predicted_polygons)))

    # to avoid zero-division
    if tp == 0:
        return 0.
    fp = len(predicted_polygons) - tp
    fn = len(groundtruth_polygons) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if v:
        log = {'TP': str(tp), 'FN': str(fn), 'FP': str(fp)}
    return 2 * (precision * recall) / (precision + recall), log


global_groundtruth_polygons = []
global_groundtruth_rtree_index = rtree.index.Index()


def point_f1_score(gt: List[Polygon],
                   pred: List[Point],
                   v=False):
    """
    Checks the f1 for object detection, true positive is when a detected point is inside a gt polygon
    It does not give precise result if several points are within one objects.
    In our case it is avoided by filtering of the prediction by size

    Also now it does not check if all the predictions are points, and can give incorrect results if there are
    polygons or lines etc.
    :param gt: groundtruth as list of polygons or a multipolygon
    :param pred: prediction as a list of points = centorids of predicted objects
    :return: F1-score
    """
    log = {}
    if len(pred) == 0:
        return 0
    tp = 0
    fp = 0
    gt = MultiPolygon(gt)
    for point in pred:
        if gt.contains(point):
            tp += 1
        else:
            fp += 1
    fn = len(gt) - tp
    if v:
        log = {'TP': str(tp), 'FN': str(fn), 'FP': str(fp)}

    return (2*tp / (2*tp + fp + fn)), log


def _has_match_rtree(polygon_serialized):
    global IOU_THRESHOLD
    global global_groundtruth_rtree_index
    polygon = loads(polygon_serialized)
    best_iou = 0
    candidates = [
        loads(candidate_serialized)
        for candidate_serialized
        in global_groundtruth_rtree_index.intersection(
            polygon.bounds, objects='raw'
        )
    ]
    for candidate in candidates:
        metric = iou(polygon, candidate)
        if metric > best_iou:
            best_iou = metric

    if best_iou > IOU_THRESHOLD:
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


