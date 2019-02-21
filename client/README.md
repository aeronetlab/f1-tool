# F1-score utility

Simple http client for f1_server

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
* point: works as 'vector', but extracts centroid from prediction polygons

Consult help for actual arguments:
```bash
python f1_utility.py --help
```

```bash
Usage: f1_client.py [OPTIONS] GROUNDTRUTH_PATH PREDICTED_PATH

Options:
  --format [raster|vector|point][required]
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

