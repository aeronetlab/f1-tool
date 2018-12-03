import os
import flask
import logging
from flask import Flask
from typing import List
from io import BytesIO
from zipfile import ZipFile
from flask import jsonify

import geojson
import rasterio
from shapely.geometry import Polygon, asShape, MultiPolygon, Point

from f1_calc import objectwise_f1_score, pixelwise_f1_score, point_f1_score

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

    #task={'format':'raster|vector|point', 'iou':'0.5', 'timestamp':yyyymmddhhmmss}

    #task = eval(flask.request.args.get('task'))
    format = flask.request.args.get('format')
    timestamp = flask.request.args.get('timestamp')
    v = flask.request.args.get('v')
    print(format, timestamp)

    # request body??
    # zip file must contain files named gt.[tif|geojson] and pred.[tif|geojson]
    zip = flask.request.files['file']
    #zip.save('./tmp2.zip')
    gt_path, pred_path, area_path = extract_files(zip,
                                                  format,
                                                  os.path.join(INTERNAL_DIR, timestamp))

    if format == 'raster':
        with rasterio.open(gt_path) as src:
            gt_img = src.read(1)
        with rasterio.open(pred_path) as src:
            pred_img = src.read(1)
        score, log = pixelwise_f1_score(gt_img, pred_img, v)

    elif format == 'vector':
        with open(gt_path) as src:
            gt = geojson.load(src)
        with open(pred_path) as src:
            pred = geojson.load(src)

        gt_polygons = get_polygons(gt)
        pred_polygons = get_polygons(pred)
        score, log = objectwise_f1_score(gt_polygons, pred_polygons, v)

    elif format == 'point':
        with open(gt_path) as src:
            gt = geojson.load(src)
        with open(pred_path) as src:
            pred = geojson.load(src)

        gt_polygons = get_polygons(gt)
        pred_polygons = get_polygons(pred)
        pred_points = [poly.centroid for poly in pred_polygons]
        score, log = point_f1_score(gt_polygons, pred_points, v)
    else:
        raise ValueError('Invalid format. Expected: raster/vector/point')
    print(score)
    if v:
        return jsonify({'score': score, 'log': log})
    else:
        return jsonify({'score': score})
    # return the data dictionary as a JSON response

def extract_files(zip, format, dir):
    """
    Parses the data from an inference request from backend
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
            raise ValueError('There must be gt and pred files in archive')
        gt_name = 'gt.tif'
        pred_name = 'pred.tif'

    else: #vector or point
        if 'gt.geojson' not in names or 'pred.geojson' not in names:
            raise ValueError('There must be gt and pred files in archive')
        gt_name = 'gt.geojson'
        pred_name = 'pred.geojson'

    zf.extractall(path=os.path.join(dir))

    return os.path.join(dir, gt_name), os.path.join(dir, pred_name), area_name

def get_polygons(json) -> List[Polygon]:
    res = []  # type: List[Polygon]
    for f in json.features:
        if isinstance(f.geometry, geojson.MultiPolygon):
            res += [asShape(geojson.Polygon(c)) for c in f.geometry.coordinates]
        elif isinstance(f.geometry, geojson.Polygon):
            res += [asShape(f.geometry)]
        else:
            raise Exception("Unexpected FeatureType:\n" + f.geometry['type'] + "\nExpected Polygon or MultiPolygon")
    return res

def get_points(json) -> List[Point]:
    res = []  # type: List[Point]
    for f in json.features:
        if isinstance(f.geometry, geojson.MultiPoint):
            res += [asShape(geojson.Point(c)) for c in f.geometry.coordinates]
        elif isinstance(f.geometry, geojson.Point):
            res += [asShape(f.geometry)]
        else:
            raise Exception("Unexpected FeatureType:\n" + f.geometry['type'] + "\nExpected Point")
    return res


if __name__ == '__main__':

    # development environment
    if debug:
        # in dev environment we have hot-code-reload which reloads app immediately after start
        # (looks like starting twice),
        # because of this, model is loaded before first request (see setup_dev_environment)
        app.run(host='0.0.0.0', debug=True)
    # production environment
    else:
        # in production we set up model on start
        # instead of "before first request" to have no "cold start problem"
        # setup_model()
        app.run(host='0.0.0.0')
