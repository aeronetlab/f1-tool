

class Metric:
    def __init__(self):
        pass

    def __call__(self, gt_data, pred_data, filetype, **kwargs):
        if filetype == 'geojson':
            return self.vector_metric(gt_data, pred_data, **kwargs)
        else:
            return self.raster_metric(gt_data, pred_data, **kwargs)

    def raster_metric(self, gt_data, pred_data, **kwargs):
        raise NotImplementedError('Raster data is not accepted by this metric')

    def vector_metric(self, gt_data, pred_data, **kwargs):
        raise NotImplementedError('Vector data is not accepted by this metric')