# About F1-score utility

Geographic data F1-score metric calculation utility.

You can test / use it on our server. [https://f1.aeronetlab.space](https://f1.aeronetlab.space) (the limit for input files is 15000x15000 pixels)

## Features

- Object- and pixelwise scores, and point-to-object score
- Supported formats:
  - GeoTIFF
  - GeoJSON
- (Upcoming) Generating difference masks

## Metric

Objectwise metric is based on the metric used in SpaceNet Challenge
https://medium.com/the-downlinq/the-spacenet-metric-612183cc2ddb

Pixelwise metric is plain F1 score:
F1 = 2\*precision\*recall/(precision + recall)

Point metric is calculated similar to objectwise, but instead of IoU > Threshold the
condition for positive detection is when centroid of the detected object lies within the polygon
of a ground truth object

## Pre-requisites

This is a server app that is hosted in the docker (dockerfile provided)

### Any OS

- docker 17+

## Setup

- build docker image:

```bash
docker build -t f1_server .
```

- run docker containter:

```bash
docker run -d -p <outer_port>:5000 f1_server
```

- for CORS support pass list of url's:

```bash
docker run -d -p <outer_port>:5000 -e CORS_ALLOWED=https://aeronetlab.space,http://osd.aeronetlab.space f1_server
```

## Usage

The server accepts HTTP POST requests in the following form:

- parameters:
  - format:raster|vector|point,
  - iou: float from 0 to 1, default 0.5
  - timestamp: for request identification, POSIX timestamp
  - v: boolean True/False for verbose output
- request body: \* files = {'file': [zip file]}
  where zip file is an archive containing groundtruth and prediction files:
  <b>gt.tif</b> and <b>pred.tif</b> in case of 'raster' format,
  and <b>gt.geojson</b> and <b>pred.geojson</b> in case of 'vector' or 'point' format

Request example:

```bash
curl -X POST url:port/f1?format=vector&iou=0.5&timestamp=1543932176.827446?v=True \
    tests/data/ventura/ventura_class_801.geojson \
    tests/data/ventura/ventura_class_801_pred.geojson \
    --format=vector -v
```

Available formats are

- raster: compares two `*.tif` and measures pixelwise f1 score
- vector: compares two `*.geojson` and measures objectwise f1 score

## Test calculation functions

In command line from this directory:

```bash
python -m unittest tests/test_f1_calc.py
```
