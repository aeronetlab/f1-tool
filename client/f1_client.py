import click
import requests

URL = 'http://0.0.0.0:5000/f1'


@click.command()
@click.argument("groundtruth_path", type=click.Path(exists=True))
@click.argument("predicted_path", type=click.Path(exists=True))
@click.option("--format", required=True, type=click.Choice(['area', 'object', 'point']))
@click.option("--score", required=True, type=str, default='f1_score')
@click.option("--iou", type=str, default=0.5, help="Intersection-over-union threshold (default: 0.5)")
@click.option("-v", is_flag=True, help='Enables verbose output')
@click.option("--url", default=URL, help='Specifies url instead of default')
#@click.option("--logfile", required=False, help='Log file for saving all the application output')
@click.option("--area", default=None, help='Specifies area')
@click.option("--bbox", default=None, help='Specifies area')

def command(groundtruth_path,
            predicted_path,
            format='area',
            score_fn='f1_score',
            v=False, iou=0.5,
            url=URL, area=None,
            bbox=None):

    if v:
        print('Calculating f1-score for files:\n ground truth %s \n prediction %s' %(groundtruth_path, predicted_path))

    if groundtruth_path[-8:] == '.geojson':
        filetype = 'geojson'
    else:
        filetype = 'tif'
    params = {'v': v, 'iou': iou, 'method': format, 'score_fn': score_fn, 'filetype': filetype}
    if bbox:
        params['bbox'] = bbox
    files = {'gt': open(groundtruth_path, 'rb'),
             'pred':open(predicted_path, 'rb')}
    if area:
        files['area'] = open(area, 'rb')
    response = requests.post(url, files=files, params=params)

    if v:
        print(response.json()['log'])
    if response.status_code != 200:
        print("Error " + str(response.status_code))
    else:
        print("F1 score = %.3f" % response.json()['score'])


if __name__ == '__main__':
    command()
