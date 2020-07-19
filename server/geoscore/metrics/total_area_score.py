import geojson
import rasterio
import numpy as np
from typing import List
from geoscore.proc import get_geom, cut_by_area
from shapely.geometry import Polygon, MultiPolygon


def total_area_score(gt, pred, area=None, v: bool = False):
    """
    Specific area score for all the classes together
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
    properties = []
    gt_polygons = []
    pred_polygons = []
    print('Begin calc total area score')
    try:
        #gt = geojson.load(gt_file)
        # GT is always as polygons, not points

        # Find all classes
        for feat in gt.features:
            if 'class_id' in feat['properties'].keys():
                if feat['properties']['class_id'] not in properties:
                    properties.append(feat['properties']['class_id'])
        print(f'List of classes: properties')
        for prop in properties:
            gt_polygons.append(get_geom(gt, 'polygon', classes={'class_id': prop}))

        if v:
            log += "Read groundtruth geojson, contains " + str(len(gt_polygons)) + " polygons \n"
    except Exception as e:
        raise Exception(log + 'Failed to read groundtruth file as geojson\n' + str(e))

    try:
        #pred = geojson.load(pred_file)
        for prop in properties:
            pred_polygons.append(get_geom(pred, 'polygon', classes={'class_id': prop}))
        if v:
            log += "Read predicted geojson, contains " + str(len(pred_polygons)) + " objects \n"
    except Exception as e:
        raise Exception(log + 'Failed to read prediction file as geojson\n' + str(e))

    all_class_scores = []
    areas = []
    for prop, gt_class, pred_class in zip(properties, gt_polygons, pred_polygons):
        print(f'Class {prop}')

        if area:
            try:
                gt_class = cut_by_area(gt_class, area, True)
                pred_class = cut_by_area(pred_class, area, True)
            except Exception as e:
                log += "Intersection cannot be calculated, ignoring area \n" \
                       + str(e) + '\n'

            log += "Class " + str(prop) + "\n" + \
                   "Cut vector data by specified area:\n" + \
                   str(len(gt_class)) + " groundtruth and " + \
                   str(len(pred_class)) + " predicted polygons inside\n"

        score, score_log = calc_total_area_vector(gt_class, pred_class, v)
        all_class_scores.append(score)
        areas.append(MultiPolygon(gt_class).area)
        log += score_log

        print(areas[-1])
        print(score)


    # Calculate weighted average
    avg_score = np.sum([area*score for area,score in zip(areas, all_class_scores)])/np.sum(areas)
    return avg_score, log


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
        score = 1 - abs(pred_area-gt_area)/(gt_area + pred_area)

    if v:
        log = f'Ground truth area = {gt_area}, predicted area = {pred_area}. \n'

    return score, log
