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

    '''
    Calculates min and max pixel coordinates for the image intersection in each image
    :param gt_file: file object to open with rasterio
    :param pred_file: file object to open with rasterio
    :return: [[y0, y1], [x0,x1]], [[y0, y1], [x0,x1]] - start and end pixel coordinates
    '''
    # return gt_bbox, pred_bbox
    # raise NotImplementedError

    # data_gt = rasterio.open(gt_file)
    # data_pred = rasterio.open(pred_file)

    with rasterio.open(gt_file) as dat:
        data_gt = dat

    with rasterio.open(pred_file) as data:
        data_pred = data

    geo1_gt = list(data_gt.transform * (0, 0))  # Верхний левый угол
    geo2_gt = list(data_gt.transform * (data_gt.width, data_gt.height))  # Правый нижний угол
    geo3_gt = [(geo1_gt[0] + (geo2_gt[0] - geo1_gt[0])), geo1_gt[1]]  # Верхний правый угол
    geo4_gt = [(geo2_gt[0] - (geo2_gt[0] - geo1_gt[0])), geo2_gt[1]]  # Левый нижний угол

    geo1_pred = list(data_pred.transform * (0, 0))  # Верхний левый угол
    geo2_pred = list(data_pred.transform * (data_pred.width, data_pred.height))  # Правый нижний угол
    geo3_pred = [(geo1_pred[0] + (geo2_pred[0] - geo1_pred[0])), geo1_pred[1]]  # Верхний правый угол
    geo4_pred = [(geo2_pred[0] - (geo2_pred[0] - geo1_pred[0])), geo2_pred[1]]  # Левый нижний угол

    im_1 = Polygon([geo1_gt, geo3_gt, geo2_gt, geo4_gt])
    im_2 = Polygon([geo1_pred, geo3_pred, geo2_pred, geo4_pred])
    section = im_1.intersection(im_2)

    if section.area > 0:

        s = section.boundary.coords
        p_s = s[0]
        p_s1 = s[1]
        p_s2 = s[2]
        p_s3 = s[3]

        x, y = p_s
        row_gt, col_gt = data_gt.index(x, y)
        row_pred, col_pred = data_pred.index(x, y)

        x1, y1 = p_s1
        row1_gt, col1_gt = data_gt.index(x1, y1)
        row1_pred, col1_pred = data_pred.index(x1, y1)

        x2, y2 = p_s2
        row2_gt, col2_gt = data_gt.index(x2, y2)
        row2_pred, col2_pred = data_pred.index(x2, y2)

        x3, y3 = p_s3
        row3_gt, col3_gt = data_gt.index(x3, y3)
        row3_pred, col3_pred = data_pred.index(x3, y3)

        band2 = data_test_post_class.read(1)
        a_ = band2[row_:row3_ + 1, col_:col1_]

        return [[row_gt, col_gt], [row3_gt, col1_gt]], [[row_pred, col_pred],
                                                        [row3_pred, col1_pred]]  ## start and end pixel coordinates

    else:

        return None


def geo_crop (gt_file, pred_file):
    """
    Crops the intersecting region of 2 images and returns both raster ready ready for comparison
    (guaranteed equal size and the same georeference)
    :param gt_file:
    :param pred_file:
    :return: 2 not geore ferenced arrays for comparison
    """
    # gt_bbox, pred_bbox = find_intersection(gt_file, pred_file)
    # gt_crop = read(gt_file)[gt_bbox]
    # pred_crop = read(pred_file)[pred_bbox].reshape(gt_crop.shape)
    # return gt_crop, pred_crop
    # raise NotImplementedError

    # with rasterio.open(gt_file) as dat1:
    #   data_gt = dat1

    # with rasterio.open(pred_file) as data1:
    #   data_pred = data1

    data_gt = rasterio.open(gt_file)
    data_pred = rasterio.open(pred_file)

    gt_bbox, pred_bbox = find_intersection(gt_file, pred_file)

    gt = data_gt.read(1)
    gt_crop = gt[gt_bbox[0][0]: gt_bbox[1][0], gt_bbox[0][1]: gt_bbox[1][1]]

    pred = data_pred.read(1)
    pred_crop = pred[pred_bbox[0][0]: pred_bbox[1][0], gt_bbox[0][1]: gt_bbox[1][1]]

    return gt_crop, pred_crop


