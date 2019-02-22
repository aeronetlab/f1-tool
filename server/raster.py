import numpy as np

def pixelwise_raster_f1(groundtruth_array, predicted_array, v: bool=False):
    """
    Calculates f1-score for 2 equal-sized arrays
    :param groundtruth_array:
    :param predicted_array:
    :param v: is_verbose
    :return: float, f1-score and string, log
    """
    log = ''
    assert groundtruth_array.shape == predicted_array.shape, "Images has different sizes"
    groundtruth_array[groundtruth_array > 0] = 1
    predicted_array[predicted_array > 0] = 1

    tp = np.logical_and(groundtruth_array, predicted_array).sum()
    fn = int(groundtruth_array.sum() - tp)
    fp = int(predicted_array.sum() - tp)
    if tp == 0:
        f1 = 0
    else:
        f1 = (2 * tp / (2 * tp + fn + fp))
    if v:
        log = 'True Positive = ' + str(tp) + ', False Negative = ' + str(fn) + ', False Positive = ' + str(fp) + '\n'
    return f1, log
