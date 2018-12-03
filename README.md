# F1-score utility

Geographic data F1-score metric calculation utility

## Features

* Object- and pixelwise scores
* Supported formats:
  * GeoTIFF
  * GeoJSON
* (Upcoming) Generating difference masks


## Metric

Based on the metric used in SpaceNet Challenge
https://medium.com/the-downlinq/the-spacenet-metric-612183cc2ddb

## Pre-requisites

### Any OS

* Python 3

### Windows

* Download latest Python 3 from https://www.python.org/downloads/windows/. As of today the latest is Python 3.7, installer for x64 architecture is https://www.python.org/ftp/python/3.7.0/python-3.7.0-amd64.exe
* Run command prompt as Administrator and execute `python -m pip install --upgrade pip`
* Run `pip install wheel`
* Download Windows binaries from https://www.lfd.uci.edu/~gohlke/pythonlibs/ for libraries Shapely and Rtree, for python 3.7 and x64 architecture they are Shapely‑1.6.4.post1‑cp37‑cp37m‑win_amd64.whl and Rtree‑0.8.3‑cp37‑cp37m‑win_amd64.whl, respectively.
* `pip install Shapely‑1.6.4.post1‑cp37‑cp37m‑win_amd64.whl`
* `pip install Rtree‑0.8.3‑cp37‑cp37m‑win_amd64.whl`


## Setup
Create new virtual environment with Python 3 or use system one.
* `pip install -r requirements.txt`

## Usage
Usage example:
```bash
python f1_utility.py \
    tests/data/ventura/ventura_class_801.geojson \
    tests/data/ventura/ventura_class_801_pred.geojson \
    --format=vector -v
```

Available formats are
* raster: compares two `*.tiff` and measures pixelwise f1 score
* vector: compares two `*.geojson` and measures objectwise f1 score

Consult help for actual arguments:
```bash
python f1_utility.py --help
```

```bash
Usage: f1_utility.py [OPTIONS] GROUNDTRUTH_PATH PREDICTED_PATH

Options:
  --format [raster|vector]      [required]
  --multiproc / --no-multiproc  Enables/disables multiprocessing for
                                objectwise score. Enabled by default.
                                Multiprocessing doesn't work in Windows.
  --method [basic|rtree]        The method to match groundtruth and predicted
                                polygons (objectwise score)
  --iou FLOAT                   Intersection-over-union threshold (default:
                                0.5)
  -v                            Enables verbose output
  --help                        Show this message and exit.
```

## Test

In command line from this directory:
```bash
python -m unittest tests/test_f1_score.py
```

## Known bugs and limitations

Multiprocessing isn't supported on Windows.