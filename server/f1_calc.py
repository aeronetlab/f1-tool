import geojson
import rasterio
import numpy as np

from .raster import pixelwise_raster_f1
from .vector import pixelwise_vector_f1, objectwise_f1_score
from .proc import get_geom, cut_by_area


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

