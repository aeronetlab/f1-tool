import numpy as np
import geojson
import rasterio
from typing import List
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, Point, asShape


def extract_data(gt_file,
                 pred_file,
                 filetype,
                 gt_feature_type='polygon',
                 pred_feature_type='polygon',
                 vector_properties=None,
                 v=False):
    if filetype == 'geojson':
        try:
            if vector_properties is not None:
                gt_data, pred_data, log = read_multiclass_vector_data(gt_file, pred_file,
                                                                      gt_type=gt_feature_type,
                                                                      pred_type=pred_feature_type,
                                                                      properties=vector_properties, v=v)
            else:
                gt_data, pred_data, log = read_vector_data(gt_file, pred_file,
                                                           gt_type=gt_feature_type, pred_type=pred_feature_type, v=v)
        except Exception as e:
            log = 'Error while reading vector data: ' + str(e)
            raise Exception(log + str(e))
    else:  # any other supported filetype is raster
        try:
            gt_data, pred_data, log = read_raster_data(gt_file, pred_file, v=v)
        except Exception as e:
            log = 'Error while reading raster data: ' + str(e)
            raise Exception(log + str(e))

    return gt_data, pred_data, log


def read_raster_data(gt_file, pred_file, v=False):
    log = ''
    with rasterio.open(gt_file) as src:
        gt_img = src.read(1)
        if v:
            log += "Read groundtruth image, size = " + str(gt_img.shape) + "\n"
    with rasterio.open(pred_file) as src:
        # reading into the pre-allocated array guarantees equal sizes
        pred_img = np.empty(gt_img.shape, dtype=src.dtypes[0])
        src.read(1, out=pred_img)
        if v:
            log += "Read predicted image, size = " + str(src.width) + ', ' + str(src.height) \
                   + ', reshaped to size of GT image \n'
    return gt_img, pred_img, log


def read_vector_data(gt_file, pred_file,
                     gt_type='polygon', pred_type='polygon', v=False):
    log = ''
    try:
        pred = geojson.load(pred_file)
        pred_polygons = {'all': get_geom(pred, pred_type)}
        if v:
            log += 'Read predicted geojson, contains {} {} objects \n'.format(len(pred_polygons), pred_type)
    except Exception as e:
        raise Exception('Failed to read prediction file as geojson\n' + str(e))

    try:
        gt = geojson.load(gt_file)
        gt_polygons = {'all': get_geom(gt, gt_type)}
        if v:
            log += 'Read groundtruth geojson, contains {} {} objects \n'.format(len(gt_polygons), gt_type)
    except Exception as e:
        raise Exception(log + 'Failed to read groundtruth file as geojson\n' + str(e))
    return gt_polygons, pred_polygons, log


def read_multiclass_vector_data(gt_file, pred_file,
                                gt_type='polygon', pred_type='polygon',
                                properties=None, v=False):
    log = ''
    pred_polygons = {}
    gt_polygons = {}
    for prop in properties.items():
        prop_key = '{}:{}'.format(prop[0], prop[1])
        try:
            pred = geojson.load(pred_file)
            pred_polygons[prop_key] = get_geom(pred, pred_type, prop)
            if v:
                log += 'Read predicted geojson, contains {} {} features of {}\n'.format(len(pred_polygons[prop_key]),
                                                                                        pred_type, prop_key)
        except Exception as e:
            raise Exception(log + 'Failed to read prediction file as geojson\n' + str(e))

        try:
            gt = geojson.load(gt_file)
            gt_polygons['{}:{}'.format(prop[0], prop[1])] = get_geom(gt, gt_type, prop)
            if v:
                log += 'Read groundtruth geojson, contains {} {} features of {}\n'.format(len(gt_polygons[prop_key]),
                                                                                          gt_type, prop_key)
        except Exception as e:
            raise Exception(log + 'Failed to read groundtruth file as geojson\n' + str(e))

    return gt_polygons, pred_polygons, log


def get_geom(json, geom_type='polygon',
             prop=None):
    """

    Extracts all the polygons from the geojson object and reproject them to lat-lon crs
    The lines are ignored, while multipolygons are divided into individual polygons and concatenated
    with polygons list.
    If format == 'point', all the points are extracted, and for every polygon its centroid is returned

    Args:
        json: Input json structure
        geom_type: 'polygon' or 'point', represents return data type
        prop: a single property as a tuple (property name, property value) to extract only the matching features
    Returns:
        list of geometries of the selected type

    """

    if geom_type.lower() not in ['polygon', 'point']:
        raise ValueError('Param geom_type must be either `polygon` or `point`, got {} instead'.format(geom_type))

    polys = []  # type: List[Polygon]
    points = []  # type: List[Point]

    # the crs may be specified by the geojson standard or as 'crs':'EPSG:____', we should accept both
    if isinstance(json['crs'], str):
        src_crs = json['crs']
    else:
        src_crs = json['crs']['properties']['name']

    # then we will reproject all to lat-lon (EPSG:4326)
    dst_crs = 'EPSG:4326'

    for f in json.features:
        # We do not check the feature properties if the preperty of interest is None
        if prop is None \
                or (f.properties is not None
                    and prop[0] in f.properties.keys()
                    and f.properties[prop[0]] == prop[1]):

            if isinstance(f.geometry, geojson.MultiPolygon):
                try:
                    new_geom = transform_geom(src_crs=src_crs,
                                              dst_crs=dst_crs,
                                              geom=f.geometry)
                except ValueError:
                    # we ignore the invalid geometries
                    continue
                # transform_geom outputs an instance of Polygon, if the input is a MultiPolygon with one contour
                if new_geom['type'] == 'Polygon':
                    polys += [asShape(new_geom)]
                else:
                    polys += [asShape(geojson.Polygon(c)) for c in new_geom['coordinates']]
            elif isinstance(f.geometry, geojson.Polygon):
                try:
                    new_geom = transform_geom(src_crs=src_crs,
                                              dst_crs=dst_crs,
                                              geom=f.geometry)
                except ValueError:
                    # we ignore the invalid geometries
                    continue
                polys += [asShape(new_geom)]
            elif isinstance(f.geometry, geojson.Point):
                try:
                    new_geom = transform_geom(src_crs=src_crs,
                                              dst_crs=dst_crs,
                                              geom=f.geometry)
                except ValueError:
                    # we ignore the invalid geometries
                    continue
                points += [asShape(new_geom)]

    if geom_type == 'polygon':
        return polys
    else:  # geom_type == 'point':
        points += [poly.centroid for poly in polys]
        return points
