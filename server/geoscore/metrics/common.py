import warnings
EPS = 0.00000001
"""
Here is a set of scoring functions which can be derived from true positive, 
false positive, true negative, false negative values. 
"""


def get_scoring_function(name, **kwargs):

    if name == 'f_score':
        beta_ = kwargs.get('beta')
        try:
            beta = float(beta_)
            assert beta>0
        except Exception:
            warnings.warn(f'Beta must be a positive number, got {beta_}. Using 1 instead')
            beta = 1
        return get_f_score(beta)

    scoring_functions = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'jaccard': jaccard
    }

    if name not in scoring_functions.keys():
        raise KeyError(f'Unknown scoring function {name}. \n Allowed functions are: {list(scoring_functions.keys())}')

    return scoring_functions[name]

def check_input(tp, fp, tn, fn):
    if tp < 0 or fp < 0 or tn < 0 or fn < 0:
        raise ValueError(f'All the values must be non-negative, got TP={tp}, TN={tn}, FP={fp}, FN={fn}')


def precision(tp, fp, tn, fn):
    check_input(tp, fp, tn, fn)
    if tp == 0:
        return 0
    return tp / (tp + fp)


def recall(tp, fp, tn, fn):
    check_input(tp, fp, tn, fn)
    if tp == 0:
        return 0
    return  tp / (tp + fn)


def get_f_score(beta):
    def f_beta_score(tp, fp, tn, fn):
        check_input(tp, fp, tn, fn)
        if tp == 0:
            return 0.
        pr = precision(tp, fp, tn, fn)
        rec = recall(tp, fp, tn, fn)
        return (1+beta**2)*pr*rec/(beta**2*pr + rec)

    return f_beta_score


def f1_score(tp, fp, tn, fn):
    return get_f_score(1)(tp, fp, tn, fn)


def accuracy(tp, fp, tn, fn):
    check_input(tp, fp, tn, fn)
    if tp == 0:
        return 0
    return (tp+tn)/(tp+tn+fp+fn)


def jaccard(tp, fp, tn, fn):
    check_input(tp, fp, tn, fn)
    if tp == 0:
        return 0
    return tp/(tp+fn+fp)

