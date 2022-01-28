def accuracy(vp, vn, fp, fn):
    return (vp+vn) / (vp + vn + fp + fn)


def precision(vp, fp):
    return vp / (vp + fp)


def recall(vp, fn):
    return vp / (vp + fn)


def f1score(precision_val, recall_val):
    return (2 * precision_val * recall_val) / (precision_val + recall_val)

