import rtree
import geojson
import rasterio
import numpy as np

from typing import List
from shapely.wkb import dumps, loads
from rasterio.warp import transform_geom
from rasterio import RasterioIOError
from shapely.geometry import MultiPolygon, Point, Polygon, asShape


EPS = 0.00000001


# ================= PIXELWISE F1 ================================

def pixelwise_file_score(gt_file,
                         pred_file,
                         v: bool = False,
                         filetype='tif'):
    """

    :param gt_file:
    :param pred_file:
    :param v:
    :return:
    """
    log = ''
    if filetype == 'geojson':
        try:
            pred = geojson.load(pred_file)
            pred_polygons = get_geom(pred, 'vector')
            if v:
                log += "Read predicted geojson, contains " + str(len(pred_polygons)) + " objects \n"
        except Exception as e:
            raise Exception(log + 'Failed to read prediction file as geojson\n' + str(e))

        try:
            gt = geojson.load(gt_file)
            # GT is always as polygons, not points
            gt_polygons = get_geom(gt, 'vector')
            if v:
                log += "Read groundtruth geojson, contains " + str(len(gt_polygons)) + " polygons \n"
        except Exception as e:
            raise Exception(log + 'Failed to read groundtruth file as geojson\n' + str(e))
        score, score_log = pixelwise_vector_f1(gt_polygons, pred_polygons, v)

    else: # tif or any other (default) value
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
            score, score_log = pixelwise_raster_f1(gt_img, pred_img, v)
        except Exception as e:
            raise Exception(log + 'Failed to read input file as raster\n' + str(e))
    return score, log + score_log

def pixelwise_raster_f1(groundtruth_array, predicted_array, v: bool=False):
    """
    Calculates f1-score for 2 equal-sized arrays
    :param groundtruth_array:
    :param predicted_array:
    :param v: is_verbose
    :return:
    """
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


def pixelwise_vector_f1(gt: List[Polygon],
                        pred: List[Polygon],
                        v: bool=True):
    """
    Measures pixelwise f1-score, but for vector representation instead of raster.

    :param gt: list of shapely Polygons, represents ground truth;
    :param pred: list of shapely Polygons or Points (according to the 'format' param, represents prediction;
    :param format: 'vector' or 'point', means format of prediction and corresponding variant of algorithm;
    :param v: is_verbose
    :return: float, f1-score and string, log
    """
    log = ''
    gt_mp = MultiPolygon(gt)
    pred_mp = MultiPolygon(pred)

    # try making polygons valid
    gt_mp = gt_mp.buffer(0)
    pred_mp = pred_mp.buffer(0)

    tp = gt_mp.intersection(pred_mp).area
    fp = pred_mp.area - tp
    fn = gt_mp.area - tp

    if tp == 0:
        f1 = 0.
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    if v:
        log += 'True Positive = ' + str(tp) + ', False Negative = ' + str(fn) + ', False Positive = ' + str(fp) + '\n'

    return f1, log


# ==================================== OBJECTWISE F1 ============================================


def objectwise_file_score(gt_file, pred_file, area, format, v: bool=True, iou=0.5):
    '''
    All the work with vector data, either in object or in point score
    :param gt_file:
    :param pred_file:
    :param area:
    :param format:
    :param v:
    :param iou:
    :return:
    '''
    log = ''
    try:
        gt = geojson.load(gt_file)
        # GT is always as polygons, not points
        gt_polygons = get_geom(gt, 'vector')
        if v:
            log += "Read groundtruth geojson, contains " + str(len(gt_polygons)) + " polygons \n"
    except Exception as e:
        raise Exception(log + 'Failed to read geojson groundtruth file\n' + str(e))

    try:
        pred = geojson.load(pred_file)
        pred_geom = get_geom(pred, format)
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
        score, score_log = objectwise_f1_score(gt_polygons, pred_geom, format, iou=iou, v=v)
    except Exception as e:
        raise Exception(log + 'Error while calculating objectwise f1-score in ' + format + ' format\n' + str(e))

    return score, log + score_log


def objectwise_f1_score(gt: List[Polygon],
                        pred,
                        format,
                        iou=0.5,
                        v: bool=True):
    """
    Measures objectwise f1-score for two sets of polygons.
    The algorithm description can be found on
    https://medium.com/the-downlinq/the-spacenet-metric-612183cc2ddb

    If the format = 'point' True Positive counts when the prediction point lies within the polygon of GT

    :param gt: list of shapely Polygons, represents ground truth;
    :param pred: list of shapely Polygons or Points (according to the 'format' param, represents prediction;
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

    if format == 'vector':
        tp = sum(map(_has_match_rtree,
                     (dumps(polygon) for polygon in pred),
                     [iou]*len(pred),
                     [groundtruth_rtree_index]*len(pred)))
    else:  # format = 'point'
        tp = sum(map(_lies_within_rtree,
                     (point for point in pred),
                     [groundtruth_rtree_index]*len(pred)))
    fp = len(pred) - tp
    fn = len(gt) - tp
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


def _has_match_rtree(polygon_serialized, iou_threshold, groundtruth_rtree_index):

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


def _lies_within_rtree(point, groundtruth_rtree_index):

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


def iou(polygon1: Polygon, polygon2: Polygon):
    # buffer(0) may be used to “tidy” a polygon
    # works good for self-intersections like zero-width details
    # http://toblerity.org/shapely/shapely.geometry.html
    poly1_fixed = polygon1.buffer(0)
    poly2_fixed = polygon2.buffer(0)
    return poly1_fixed.buffer(0).intersection(poly2_fixed).area / poly1_fixed.union(poly2_fixed).area


def get_geom(json, format):
    polys = []  # type: List[Polygon]
    points = [] # type: List[Point]

    # the crs may be specified by the geojson standard or as 'crs':'EPSG:____', we should accept both
    if isinstance(json['crs'], str):
        src_crs = json['crs']
    else:
        src_crs = json['crs']['properties']['name']

    # then we will reproject all to lat-lon (EPSG:4326)
    dst_crs = 'EPSG:4326'

    for f in json.features:
        if isinstance(f.geometry, geojson.MultiPolygon):
            try:
                new_geom = transform_geom(src_crs=src_crs,
                                          dst_crs=dst_crs,
                                          geom=f.geometry)
            except ValueError:
                # we ignore the invalid geometries
                continue
            # transform_geom outputs an instance of Polygon, if the input is a MultiPolygon with one contour
            if new_geom['type'] == 'Polygon':
                polys += [asShape(new_geom)]
            else:
                polys += [asShape(geojson.Polygon(c)) for c in new_geom['coordinates']]
        elif isinstance(f.geometry, geojson.Polygon):
            try:
                new_geom = transform_geom(src_crs=src_crs,
                                          dst_crs=dst_crs,
                                          geom=f.geometry)
            except ValueError:
                # we ignore the invalid geometries
                continue
            polys += [asShape(new_geom)]
        elif isinstance(f.geometry, geojson.Point):
            try:
                new_geom = transform_geom(src_crs=src_crs,
                                          dst_crs=dst_crs,
                                          geom=f.geometry)
            except ValueError:
                # we ignore the invalid geometries
                continue
            points += [asShape(new_geom)]
        else:
            pass # raise Exception("Unexpected FeatureType:\n" + f.geometry['type'] + "\nExpected Polygon or MultiPolygon")

    if format == 'vector':
        return polys
    else:  # format == 'point':
        points += [poly.centroid for poly in polys]
        return points

def get_area(bbox: List[float]) -> List[Polygon]:
    poly = Polygon([(bbox[0], bbox[3]), (bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])
    assert poly.is_valid, "Bounding box polygon " + str(poly) +" is invalid \n"
    return [poly]

def cut_by_area(polygons, area):
    if area:
        area = MultiPolygon(area)
        polygons = [poly for poly in polygons if poly.intersects(area)]
    return polygons
