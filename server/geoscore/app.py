import os
import flask
import logging
import geojson
import time
from flask import Flask, jsonify
from flask_cors import CORS

from metrics import objectwise_score, objectwise_point_score, areawise_score, get_scoring_function, total_area_score
from proc import get_area, get_geom

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

    start_time = time.time()
    try:
        gt_file, pred_file, score_fn, area, v, log = parse_request(
            flask.request)
    except Exception as e:
        res = jsonify({'score': 0.0, 'log': 'Invalid request:' + str(e)}), 400
        return res  # jsonify({'score': 0.0, 'log': 'Invalid request:\n' + str(e)}), 400

    '''
    if (gt_file.filename[-4:].lower() == '.tif' or gt_file.filename[-5:].lower() == '.tiff') and \
        (pred_file.filename[-4:].lower() == '.tif' or pred_file.filename[-5:].lower() == '.tiff'):
        pass
    Maybe derive the filetype from the extension? Makes sense, if the filename is preserved during the transition
    '''

    #try:

    pred = geojson.load(pred_file)
    gt = geojson.load(gt_file)

    localization_score, localization_score_log = areawise_score(gt, pred, score_fn, area, v)
    print("LOC SCORE = " + str(localization_score))

    classification_score, classification_score_log = total_area_score(gt, pred, area, v)
    print("CLASS_SCORE = " + str(classification_score))
    #except Exception as e:
    #    print(e)
    #    return jsonify({'score': 0.0, 'log': log + str(e)}), 500

    log += localization_score_log + classification_score_log
    log += 'Execution time: ' + str(time.time() - start_time)
    if v:
        return jsonify({'localization_score': localization_score,
                        'classification_score': classification_score,
                        'log': log})
    else:
        return jsonify({'localization_score': localization_score,
                        'classification_score': classification_score})
    # return the data dictionary as a JSON response


def parse_request(request):
    """

    :param request: an HTTP-request to be parsed.

    It must contain:
    - method: the name of the method to use. Allowed: f1-pixel, f1-object, f1-point
    - filetype: file extension, tif or geojson. Actually, other raster file formats are also supported
    - gt_file: the file with Ground Truth data
    - pred_file: the file with Predicted data, whisch we want to compare to Ground Truth data using the method

    Optional parameters are:
    - area: a geojson file containing the area boundaries; the pred and gt data are cut by the area and all objects out
            of the area are ignored.
    - v: verbose, default False, if True the log is returned
    - bbox: bounding box ('min_lon, min_lat, max_lon, max_lat') which acts the same as the area.
            If both area and bbox are specified, the area overrides the bbox
    - score_fn: function that is used to calculate the final metrics from tp,tn,fp and fn values. Defaults to f1-score

    :return:
    """
    log = ''

    v = request.args.get('v') in ['True', 'true', 'yes', 'Yes', 'y', 'Y']

    # if function is not specified, f1-score is used. It may be not necessary for the method, so it is not required
    score_fn = request.args.get('score_fn')
    if not score_fn:
        score_fn = 'f1_score'
    score_fn = get_scoring_function(score_fn)
    # area is preferred over bbox, so if both are specified, area overrides bbox
    area = None
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
            area = get_geom(area_gj, geom_type='polygon')
        except Exception as e:
            log += "Specified area is invalid, ignoring it \n" \
                   "Correct format is geojson containining Polygons or MultiPolygon" \
                   + str(e) + '\n'

    # Fetching files
    if 'gt' not in request.files.keys() or 'pred' not in request.files.keys():
        raise Exception('Invalid request. Expected: gt and pred files')

    gt_file = request.files['gt']
    pred_file = request.files['pred']

    return gt_file, pred_file, score_fn, area, v, log


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=debug)
