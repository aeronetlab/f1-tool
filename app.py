import os
import flask
import logging
import geojson
import time
from flask import Flask, jsonify
from flask_cors import CORS

from f1_calc import pixelwise_file_score, objectwise_file_score, get_geom, get_area

app = Flask(__name__)
INTERNAL_DIR = '/data'
debug = os.environ.get('ENVIRONMENT') != 'production'

# if env variable CORS_ALLOWED is presented, use it as list of cors
use_cors = os.environ.get('CORS_ALLOWED')
if use_cors:
    origins = use_cors.split(',')
    CORS(app, origins=origins)
    print("Use cors: {}".format(use_cors))


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
    start_time = time.time()
    log = ''
    try:
        format, v, gt_file, pred_file, log_, area, bbox, iou, filetype = parse_request(
            flask.request)
    except Exception as e:
        return jsonify({'score': 0.0,
                        'log': log + 'Invalid request:\n' + str(e)}),
        400
    log += log_
    '''
    if (gt_file.filename[-4:].lower() == '.tif' or gt_file.filename[-5:].lower() == '.tiff') and \
        (pred_file.filename[-4:].lower() == '.tif' or pred_file.filename[-5:].lower() == '.tiff'):
        format = 'raster'
    elif gt_file.filename[-8:].lower() == '.geojson' and pred_file.filename[-8:].lower() == '.geojson':
        format = 'vector'
    else:
        return jsonify({'score': 0.0,
                        'log': 'Invalid request. gt and pred files must be both tiff or both geojson'}), \
               400
    '''

    if format == 'raster':
        try:
            score, score_log = pixelwise_file_score(gt_file, pred_file, v, filetype)
        except Exception as e:
            return jsonify({'score': 0.0, 'log': log + str(e)}), 500

    elif format in ['vector', 'point']:
        try:
            score, score_log = objectwise_file_score(
                gt_file, pred_file, area, format, v, iou=iou)
        except Exception as e:
            return jsonify({'score': 0.0, 'log': log + str(e)}), 500

    else:
        return jsonify({'score': 0.0, 'log': 'Invalid format. Expected: raster/vector/point'}), 400

    log += score_log
    log += 'Execution time: ' + str(time.time() - start_time)
    if v:
        return jsonify({'score': score, 'log': log})
    else:
        return jsonify({'score': score})
    # return the data dictionary as a JSON response


def parse_request(request):

    log = ''

    format = request.args.get('format')
    filetype = request.args.get('filetype', default='tif')

    if format == 'vector':
        try:
            iou = float(request.args.get('iou'))
            assert iou < 1.0 and iou > 0.0, "IoU must be from 0 to 1"
        except Exception:
            log += "Iou is not specified correctly, using default value 0.5\n"
            iou = 0.5
            # raise Exception("Invalid request: iou is expected to be valid float\n" + str(e))
    else:
        iou = None
    v = request.args.get('v') in ['True', 'true', 'yes', 'Yes', 'y', 'Y']

    # area is preferred over bbox, so if both are specified, area overrides bbox
    area = None
    bbox = None
    if request.args.get('bbox'):
        try:
            bbox = [float(s) for s in request.args.get('bbox').split(',')]
            assert len(bbox) == 4, "Length of bbox must be 4"
            area = get_area(bbox)
        except Exception as e:
            log += "Specified bbox is invalid, ignoring it \n"\
                   "Correct format is \'min_lon, min_lat, max_lon, max_lat\' \n" \
                   + str(e) + '\n'

    if 'area' in request.files.keys():
        area_file = request.files['area']
        try:
            area_gj = geojson.load(area_file)
            area = get_geom(area_gj, format='vector')
        except Exception as e:
            log += "Specified area is invalid, ignoring it \n" \
                   "Correct format is geojson containining Polygons or MultiPolygon" \
                   + str(e) + '\n'

    if 'gt' not in request.files.keys() or 'pred' not in request.files.keys():
        raise Exception('Invalid request. Expected: gt and pred files')

    gt_file = request.files['gt']
    pred_file = request.files['pred']

    return format, v, gt_file, pred_file, log, area, bbox, iou, filetype


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=debug)
