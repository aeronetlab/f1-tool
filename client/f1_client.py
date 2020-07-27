import click
import requests

URL = 'http://0.0.0.0:5000/f1'


@click.command()
@click.argument("groundtruth_path", type=click.Path(exists=True))
@click.argument("predicted_path", type=click.Path(exists=True))
@click.option("--score_fn", required=True, type=str, default='')
@click.option("-v", is_flag=True, help='Enables verbose output')
@click.option("-l", is_flag=True, help='Calculates localization score')
@click.option("-c", is_flag=True, help='Calculates classification score')
@click.option("--url", default=URL, help='Specifies url instead of default')
#@click.option("--logfile", required=False, help='Log file for saving all the application output')
@click.option("--area", default=None, help='Specifies area')
@click.option("--bbox", default=None, help='Specifies area')
def command(groundtruth_path,
            predicted_path,
            score_fn='',
            v=False,
            l=False,
            c=False,
            url=URL, area=None,
            bbox=None):
    if not c and not l:
        c=True
        l=True
    if v:
        print(f'Calculating {score_fn} for files:\n ground truth %s \n prediction %s' %(groundtruth_path, predicted_path))

    params = {'v': v, 'score_fn': score_fn}
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
        if l:
            print(f'Localization score = %.3f' % response.json()['localization_score'])
        if c:
            print(f'Classification score = %.3f' % response.json()['classification_score'])


if __name__ == '__main__':
    command()
