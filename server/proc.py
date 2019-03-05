import geojson
from typing import List
from rasterio.warp import transform_geom
from shapely.geometry import MultiPolygon, Polygon, Point,  asShape

# Vector preprocessing functions

def get_geom(json, format):
    """ Extracts all the polygons from the geojson object and reproject them to lat-lon crs
    The lines are ignored, while multipolygons are divided into individual polygons and concatenated
    with polygons list.
    If format == 'point', al the points are extracted, and for every polygon its centroid is returned

    :param json: Input json structure
    :param format: 'vector' or 'point', represents return data type
    # TODO: refactor - change this param name
    :return: list of geometries
    """
    polys = []  # type: List[Polygon]
    points = [] # type: List[Point]

    # the crs may be specified by the geojson standard or as 'crs':'EPSG:____', we should accept both
    if isinstance(json['crs'], str):
        src_crs = json['crs']
    else:
        src_crs = json['crs']['properties']['name']

    # then we will reproject all to lat-lon (EPSG:4326)
    dst_crs = 'EPSG:4326'

    for f in json.features:
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
        else:
            pass # raise Exception("Unexpected FeatureType:\n" + f.geometry['type'] + "\nExpected Polygon or MultiPolygon")

    if format == 'vector':
        return polys
    else:  # format == 'point':
        points += [poly.centroid for poly in polys]
        return points


def cut_by_area(polygons, area):
    """ Cuts away all the polygons that do not intersect area

    :param polygons: geometry, list of shapely(?) polygons
    :param area: Area of interest
    :return: new list of polygons without features beyond AOI
    """
    if area:
        area = MultiPolygon(area).buffer(0)
        polygons = [poly for poly in polygons if poly.intersects(area)]
    return polygons

def get_area(bbox: List[float]) -> List[Polygon]:
    poly = Polygon([(bbox[0], bbox[3]), (bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])
    assert poly.is_valid, "Bounding box polygon " + str(poly) +" is invalid \n"
    return [poly]

def find_intersection(gt_file, pred_file):
    # return gt_bbox, pred_bbox
    raise NotImplementedError

def geo_crop (gt_file, pred_file):
    # gt_bbox, pred_bbox = find_intersection(gt_file, pred_file)
    # gt_crop = read(gt_file)[gt_bbox]
    # pred_crop = read(pred_file)[pred_bbox].reshape(gt_crop.shape)
    # return gt_crop, pred_crop
    raise NotImplementedError