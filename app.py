import os
import flask
import logging
import geojson
import rasterio

import numpy as np
from flask import Flask, jsonify
from zipfile import ZipFile

from f1_calc import objectwise_f1_score, pixelwise_f1_score, point_f1_score, get_polygons

app = Flask(__name__)
INTERNAL_DIR = '/data'
debug = os.environ.get('ENVIRONMENT') != 'production'


@app.before_first_request
def setup_logging():
    print("setting up logging")
    if not debug:
        # In production mode, add log handler to sys.stderr.
        app.logger.addHandler(logging.StreamHandler())

    app.logger.setLevel(logging.INFO)


@app.route("/heartbeat", methods=["GET"])
def heartbeat():
    app.logger.info("heartbeat request")
    return "OK"


@app.route("/f1", methods=["POST"])
def evaluate():

    # task={'iou':'0.5'}
    # files={}

    try:
        iou = float(flask.request.args.get('iou'))
    except Exception as e:
        return jsonify({'score': 0.0,
                        'log': "Invalid request: iou is expected to be valid float\n" + str(e)}), \
               400
    v = flask.request.args.get('v') in ['True', 'true', 'yes', 'Yes', 'y', 'Y']

    # zip file must contain files named gt.[tif|geojson] and pred.[tif|geojson]
    if 'gt' not in flask.request.files.keys() or 'pred' not in flask.request.files.keys():
        return jsonify({'score': 0.0, 'log': 'Invalid request. Expected: gt and pred files'}), 400

    gt_file = flask.request.files['gt']
    pred_file = flask.request.files['pred']

    if (gt_file.filename[-4:].lower() == '.tif' or gt_file.filename[-5:].lower() == '.tiff') and \
        (pred_file.filename[-4:].lower() == '.tif' or pred_file.filename[-5:].lower() == '.tiff'):
        format = 'raster'
    elif gt_file.filename[-8:].lower() == '.geojson' and pred_file.filename[-8:].lower() == '.geojson':
        format = 'vector'
    else:
        return jsonify({'score': 0.0,
                        'log': 'Invalid request. gt and pred files must be both tiff or both geojson'}), \
               400

    if 'area' in flask.request.files.keys():
        area_file = flask.request.files['area']
    else:
        area_file = None

    log = ''
    if format == 'raster':
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
                    log += "Read predicted image, size = " + str(src.width) + ', ' + str(src.height)\
                           + ', reshaped to size of GT image \n'
            score, score_log = pixelwise_f1_score(gt_img, pred_img, v)
        except Exception as e:
            return jsonify({'score': 0.0, 'log': log + str(e)}), 500

    elif format == 'vector':
        try:
            gt = geojson.load(gt_file)
            gt_polygons = get_polygons(gt)
            if v:
                log += "Read groundtruth geojson, contains " + str(len(gt_polygons)) + " polygons \n"
        except Exception as e:
            return jsonify({'score': 0.0,
                            'log': log + 'Failed to read geojson prediction file\n' + str(e)}), \
                   400
        try:
            pred = geojson.load(pred_file)
            pred_polygons = get_polygons(pred)
            if v:
                log += "Read predicted geojson, contains " + str(len(pred_polygons)) + " polygons \n"
        except Exception as e:
            return jsonify({'score': 0.0,
                            'log': log + 'Failed to read geojson prediction file\n' + str(e)}), \
                   400
        if area_file:
            try:
                area = geojson.load(area_file)
                area = get_polygons(area)
            except Exception as e:
                return jsonify({'score': 0.0,
                                'log': log + 'Failed to read area file\n' + str(e)}), \
                        400
        else:
            area = None
        try:
            score, score_log = objectwise_f1_score(gt_polygons, pred_polygons, iou=iou, v=v, area=area)
        except Exception as e:
            return jsonify({'score': 0.0,
                            'log': log + 'Error while calculating objectwise f1-score\n' + str(e)}), \
                   500


    elif format == 'point':
        try:
            gt = geojson.load(gt_file)
            pred = geojson.load(pred_file)

            gt_polygons = get_polygons(gt)
            if v:
                log += "Read groundtruth geojson, contains " + str(len(gt_polygons)) + " polygons \n"
            pred_polygons = get_polygons(pred)
            if v:
                log += "Read predicted geojson, contains " + str(len(pred_polygons)) + " polygons \n"
            pred_points = [poly.centroid for poly in pred_polygons]
            if v:
                log += "Extracted points as centrods of the predicted polygons \n"
            score, score_log = point_f1_score(gt_polygons, pred_points, v)
        except Exception as e:
            return jsonify({'score': 0.0, 'log': str(e)}), 500

    else:
        return jsonify({'score': 0.0, 'log': 'Invalid format. Expected: raster/vector/point'}), 400

    log += score_log
    if v:
        return jsonify({'score': score, 'log': log})
    else:
        return jsonify({'score': score})
    # return the data dictionary as a JSON response

def extract_files(zip, format, dir):
    """
    Parses the data from an inference request
    """
    zf = ZipFile(zip)
    names = zf.namelist()
    if len(names) < 2 or len(names) > 3:
        raise ValueError('There must be 2 or 3 files in archive')

    if 'area.geojson' not in names:
        area_name = None
    else:
        area_name = os.path.join(dir, 'area.geojson')

    if format == 'raster':
        if 'gt.tif' not in names or 'pred.tif' not in names:
            raise ValueError('There must be gt.tif and pred.tif files in archive')
        gt_name = 'gt.tif'
        pred_name = 'pred.tif'

    else: # vector or point
        if 'gt.geojson' not in names or 'pred.geojson' not in names:
            raise ValueError('There must be gt.geojson and pred.geojson files in archive')
        gt_name = 'gt.geojson'
        pred_name = 'pred.geojson'

    zf.extractall(path=dir)

    return os.path.join(dir, gt_name), os.path.join(dir, pred_name), area_name


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=debug)
