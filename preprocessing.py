import numpy as np


def min_max(value):
    values=np.array(value)

    if values.shape==0:
        raise ValueError("size is zero")
    return ((values-values.min())/(values.max()-values.min()))