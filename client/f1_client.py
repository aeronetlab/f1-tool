import click
import requests

URL = 'http://0.0.0.0:5000/f1'


@click.command()
@click.argument("groundtruth_path", type=click.Path(exists=True))
@click.argument("predicted_path", type=click.Path(exists=True))
@click.option("--format", required=True, type=click.Choice(['area', 'object', 'point', 'total_area']))
@click.option("--score_fn", required=True, type=str, default='')
@click.option("--iou", type=str, default=0.5, help="Intersection-over-union threshold (default: 0.5)")
@click.option("-v", is_flag=True, help='Enables verbose output')
@click.option("--url", default=URL, help='Specifies url instead of default')
#@click.option("--logfile", required=False, help='Log file for saving all the application output')
@click.option("--area", default=None, help='Specifies area')
@click.option("--bbox", default=None, help='Specifies area')
def command(groundtruth_path,
            predicted_path,
            score_fn='',
            v=False, iou=0.5,
            url=URL, area=None,
            bbox=None):

    if v:
        print(f'Calculating {format} {score_fn} for files:\n ground truth %s \n prediction %s' %(groundtruth_path, predicted_path))

    params = {'v': v, 'iou': iou, 'score_fn': score_fn}
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
        print(f'Localization score = %.3f' % response.json()['localization_score'])
        print(f'Classification score = %.3f' % response.json()['classification_score'])


if __name__ == '__main__':
    command()
