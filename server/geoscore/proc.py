import geojson
from typing import List
from rasterio.warp import transform_geom
from rasterio.crs import CRS, CRSError
from shapely.geometry import MultiPolygon, Polygon, Point,  asShape

# Vector preprocessing functions


def get_geom(json, geom_type='polygon', classes=None):
    """

    Extracts all the polygons from the geojson object and reproject them to lat-lon crs
    The lines are ignored, while multipolygons are divided into individual polygons and concatenated
    with polygons list.
    If format == 'point', all the points are extracted, and for every polygon its centroid is returned

    Args:
        json: Input json structure
        geom_type: 'polygon' or 'point', represents return data type

    Returns:
        list of geometries of the selected type

    """

    if geom_type.lower() not in ['polygon', 'point']:
        raise ValueError(f'Param geom_type must be either `polygon` or `point`, got {geom_type} instead')

    polys = []  # type: List[Polygon]
    points = [] # type: List[Point]

    # the crs may be specified by the geojson standard or as 'crs':'EPSG:____', we should accept both
    try:
        src_crs = CRS.from_user_input(json['crs'])
    except CRSError as e:
        print('Unknown CRS ' + str(e))
        src_crs = json['crs']['properties']['name']

    # then we will reproject all to lat-lon (EPSG:4326)
    dst_crs = 'EPSG:4326'

    if classes is not None:
        features = [feat for feat in json.features if all([item in feat['properties'].items() for item in classes.items()])]
    else:
        features = json.features
    for f in features:
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

    if geom_type == 'polygon':
        return polys
    else:  # geom_type == 'point':
        points += [poly.centroid for poly in polys]
        return points


def cut_by_area(polygons, area, cut_features=False):
    """ Cuts away all the polygons that do not intersect area

    :param polygons: geometry, list of shapely(?) polygons
    :param area: Area of interest
    :return: new list of polygons without features beyond AOI
    """

    area = MultiPolygon(area).buffer(0)
    if not cut_features:
        # We leave all the features intersecting the area whole
        new_polygons = [poly.buffer(0) for poly in polygons
                        if poly.buffer(0).intersects(area)]
    else:
        new_polygons = []
        for poly in polygons:
            new_poly = poly.buffer(0).intersection(area)
            if not new_poly.is_empty:
                if isinstance(new_poly, Polygon):
                    new_polygons.append(new_poly)
                elif isinstance(new_poly, MultiPolygon):
                    new_polygons += [p for p in new_poly]

    return new_polygons


def get_area(bbox: List[float]) -> List[Polygon]:
    poly = Polygon([(bbox[0], bbox[3]), (bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])
    assert poly.is_valid, "Bounding box polygon " + str(poly) +" is invalid \n"
    return [poly]