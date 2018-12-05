import rtree
import geojson
import rasterio
import numpy as np

from shapely.geometry import MultiPolygon, Point
from shapely.wkb import dumps, loads

from typing import List
from multiprocessing import Pool
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, asShape

IOU_THRESHOLD = 0.5
global_groundtruth_rtree_index = rtree.index.Index()


def pixelwise_f1_score(groundtruth_array, predicted_array, v: bool=False):

    log = ''
    assert groundtruth_array.shape == predicted_array.shape, "Images has different sizes"
    groundtruth_array[groundtruth_array > 0] = 1
    predicted_array[predicted_array > 0] = 1

    tp = np.logical_and(groundtruth_array, predicted_array).sum()
    fn = int(groundtruth_array.sum() - tp)
    fp = int(predicted_array.sum() - tp)
    if tp == 0:
        f1 = 0
    else:
        f1 = (2 * tp / (2 * tp + fn + fp))
    if v:
        log = 'True Positive = ' + str(tp) + ', False Negative = ' + str(fn) + ', False Positive = ' + str(fp) + '\n'
    return f1, log


def pixelwise_file_score(gt_file, pred_file, v: bool=False):
    log = ''
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
    score, score_log = pixelwise_f1_score(gt_img, pred_img, v)
    return score, log + score_log


def objectwise_f1_score(groundtruth_polygons: List[Polygon],
                        predicted_polygons: List[Polygon],
                        iou=0.5,
                        v: bool=True,
                        multiproc: bool=True):
    """
    Measures objectwise f1-score for two sets of polygons.
    The algorithm description can be found on
    https://medium.com/the-downlinq/the-spacenet-metric-612183cc2ddb

    It is implemented not perfectly fair as here we do not remove groundtruth polygpons from index
    after the match is found. But if IoU threshold is higher than 0.5, and the features in prediction do not intersect,
    there can be only one match, and we presume that it is so

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
    log = ''
    global IOU_THRESHOLD
    IOU_THRESHOLD = iou
    global global_groundtruth_rtree_index
    global_groundtruth_rtree_index = rtree.index.Index()

    # for some reason builtin pickling doesn't work
    for i, polygon in enumerate(groundtruth_polygons):
        global_groundtruth_rtree_index.insert(
            i, polygon.bounds, dumps(polygon)
        )
    if multiproc:
        tp = sum(Pool().map(_has_match_rtree, (dumps(polygon) for polygon in predicted_polygons)))
    else:
        tp = sum(map(_has_match_rtree, (dumps(polygon) for polygon in predicted_polygons)))

    fp = len(predicted_polygons) - tp
    fn = len(groundtruth_polygons) - tp
    # to avoid zero-division
    if tp == 0:
        f1 = 0.
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    if v:
        log += 'True Positive = ' + str(tp) + ', False Negative = ' + str(fn) + ', False Positive = ' + str(fp) + '\n'

    return f1, log


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
    :param v: bool - verbose output
    :return: F1-score
    """
    log = ''
    if len(pred) == 0:
        return 0
    tp = 0
    fp = 0
    gt_fixed = MultiPolygon(gt).buffer(0)
    for point in pred:
        if gt_fixed.contains(point):
            tp += 1
        else:
            fp += 1
    fn = len(gt) - tp
    if tp == 0:
        f1 = 0.
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    if v:
        log = 'True Positive = ' + str(tp) + ', False Negative = ' + str(fn) + ', False Positive = ' + str(fp) + '\n'

    return f1, log


def vector_file_score(gt_file, pred_file, area, format, v: bool=True, iou=0.5):

    log = ''
    try:
        gt = geojson.load(gt_file)
        gt_polygons = get_polygons(gt)
        if v:
            log += "Read groundtruth geojson, contains " + str(len(gt_polygons)) + " polygons \n"
    except Exception as e:
        raise Exception(log + 'Failed to read geojson groundtruth file\n' + str(e))

    try:
        pred = geojson.load(pred_file)
        pred_polygons = get_polygons(pred)
        if v:
            log += "Read predicted geojson, contains " + str(len(pred_polygons)) + " polygons \n"
    except Exception as e:
        raise Exception(log + 'Failed to read geojson prediction file\n' + str(e))

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

    try:
        if format == 'vector':
            score, score_log = objectwise_f1_score(gt_polygons, pred_polygons, iou=iou, v=v)
        else:  # point
            pred_points = [poly.centroid for poly in pred_polygons]
            score, score_log = point_f1_score(gt_polygons, pred_points, v)

    except Exception as e:
        raise Exception(log + 'Error while calculating objectwise f1-score in ' + format + ' format\n' + str(e))

    return score, log + score_log

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


def get_polygons(json) -> List[Polygon]:
    res = []  # type: List[Polygon]
    if isinstance(json['crs'], str):
        src_crs = json['crs']
    else:
        src_crs = json['crs']['properties']['name']
    dst_crs = 'EPSG:4326'
    for f in json.features:
        if isinstance(f.geometry, geojson.MultiPolygon):
            new_geom = transform_geom(src_crs=src_crs,
                                      dst_crs=dst_crs,
                                      geom=f.geometry)
            if new_geom['type'] == 'Polygon':
                res += [asShape(new_geom)]
            else:
                res += [asShape(geojson.Polygon(c)) for c in new_geom['coordinates']]
        elif isinstance(f.geometry, geojson.Polygon):
            new_geom = transform_geom(src_crs=src_crs,
                                      dst_crs=dst_crs,
                                      geom=f.geometry)
            res += [asShape(new_geom)]
        else:
            raise Exception("Unexpected FeatureType:\n" + f.geometry['type'] + "\nExpected Polygon or MultiPolygon")
    return res

def get_area(bbox: List[float]) -> List[Polygon]:
    poly = Polygon([(bbox[0], bbox[3]), (bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])
    assert poly.is_valid, "Bounding box polygon " + str(poly) +" is invalid \n"
    return [poly]

def cut_by_area(polygons, area):
    if area:
        area = MultiPolygon(area)
        polygons = [poly for poly in polygons if poly.intersects(area)]
    return polygons
