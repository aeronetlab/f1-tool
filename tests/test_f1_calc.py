from time import time

import numpy as np

import unittest as unittest

import geojson
from PIL import Image

from f1_calc import objectwise_f1_score, get_polygons, pixelwise_f1_score, point_f1_score


class TestF1Score(unittest.TestCase):

    def test_get_polygons_multi_polygon(self):
        with open('tests/data/ventura/ventura_class_801.geojson') as src:
            gt = geojson.load(src)
        gt_polygons = get_polygons(gt)
        self.assertEqual(len(gt_polygons), 321)

    def test_get_polygons(self):
        with open('tests/data/ventura/ventura_class_801_pred.geojson') as src:
            pred = geojson.load(src)
        polygons = get_polygons(pred)
        self.assertEqual(len(polygons), 307)

    def test_objectwise_f1_score_rtree(self):
        with open('tests/data/ventura/ventura_class_801.geojson') as src:
            gt = geojson.load(src)
        with open('tests/data/ventura/ventura_class_801_pred.geojson') as src:
            pred = geojson.load(src)

        gt_polygons = get_polygons(gt)
        pred_polygons = get_polygons(pred)
        t = time()
        self.assertAlmostEqual(
            objectwise_f1_score(gt_polygons, pred_polygons, v=True, iou=0.5),
            0.79,
            places=2
        )
        print(time() - t)

    def test_point_f1_score(self):
        with open('tests/data/ventura/ventura_class_801.geojson') as src:
            gt = geojson.load(src)
        with open('tests/data/ventura/ventura_class_801_pred.geojson') as src:
            pred = geojson.load(src)

        gt_polygons = get_polygons(gt)
        pred_polygons = get_polygons(pred)
        pred_points = [poly.centroid for poly in pred_polygons]

        t = time()
        self.assertAlmostEqual(
            point_f1_score(gt_polygons, pred_points, v=True),
            0.87,
            places=2
        )
        print(time() - t)

    def test_pixelwise_f1_score(self):
        groundtruth_array = np.array(Image.open('tests/data/ventura/ventura_class_801.tif'))
        predicted_array = np.array(Image.open('tests/data/ventura/ventura_class_801_pred.tif'))
        self.assertAlmostEqual(
            pixelwise_f1_score(groundtruth_array, predicted_array, v=True, echo=print),
            0.73,
            places=2
        )
