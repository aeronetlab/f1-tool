import os
import flask
import logging
import geojson
import rasterio

import numpy as np
from flask import Flask, jsonify
from typing import List
from zipfile import ZipFile
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, asShape

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

    # task={'format':'raster|vector|point', 'iou':'0.5', 'timestamp':yyyymmddhhmmss}

    format = flask.request.args.get('format')
    timestamp = flask.request.args.get('timestamp')
    v = flask.request.args.get('v')
    print(format, timestamp)

    # zip file must contain files named gt.[tif|geojson] and pred.[tif|geojson]
    zip = flask.request.files['file']
    gt_path, pred_path, area_path = extract_files(zip,
                                                  format,
                                                  os.path.join(INTERNAL_DIR, timestamp))

    if format == 'raster':
        with rasterio.open(gt_path) as src:
            gt_img = src.read(1)
        # reading into the pre-allocated array guarantees equal sizes
        pred_img = np.empty(gt_img.shape)
        with rasterio.open(pred_path) as src:
            src.read(1, out=pred_img)
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
        os.removedirs(os.path.join(INTERNAL_DIR, timestamp))
        raise ValueError('Invalid format. Expected: raster/vector/point')

    os.removedirs(os.path.join(INTERNAL_DIR, timestamp))

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
            raise ValueError('There must be gt and pred files in archive')
        gt_name = 'gt.tif'
        pred_name = 'pred.tif'

    else: # vector or point
        if 'gt.geojson' not in names or 'pred.geojson' not in names:
            raise ValueError('There must be gt and pred files in archive')
        gt_name = 'gt.geojson'
        pred_name = 'pred.geojson'

    zf.extractall(path=dir)

    return os.path.join(dir, gt_name), os.path.join(dir, pred_name), area_name


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
                res += [asShape(geojson.Polygon(c)) for c in new_geom.coordinates]
        elif isinstance(f.geometry, geojson.Polygon):
            new_geom = transform_geom(src_crs=src_crs,
                                      dst_crs=dst_crs,
                                      geom=f.geometry)
            res += [asShape(new_geom)]
        else:
            raise Exception("Unexpected FeatureType:\n" + f.geometry['type'] + "\nExpected Polygon or MultiPolygon")
    return res


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=debug)
