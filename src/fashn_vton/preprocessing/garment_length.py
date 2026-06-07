import numpy as np


def is_long_garment(seg_pred, labels_ids_dict):
    """
    Detect whether garment is vertically long.

    Works for:
    - kurtis
    - oversized shirts
    - tunics
    - midi dresses
    - long dresses
    """

    garment_mask = np.isin(
        seg_pred,
        [
            labels_ids_dict.get("top", -1),
            labels_ids_dict.get("dress", -1),
            labels_ids_dict.get("outerwear", -1),
        ],
    )

    ys, xs = np.where(garment_mask)

    if len(ys) == 0:
        return False

    garment_height = ys.max() - ys.min()
    garment_width = xs.max() - xs.min()

    if garment_width <= 0:
        return False

    ratio = garment_height / garment_width

    return ratio > 1.15