import numpy as np


def detect_sleeve_type(seg_pred, labels_ids_dict):
    """
    Detect sleeve type from garment segmentation.

    Returns:
        - sleeveless
        - short
        - long
    """

    arms_id = labels_ids_dict.get("arms", -1)

    arms_mask = seg_pred == arms_id

    ys, xs = np.where(arms_mask)

    if len(ys) == 0:
        return "sleeveless"

    sleeve_height = ys.max() - ys.min()
    sleeve_width = xs.max() - xs.min()

    if sleeve_width <= 0:
        return "short"

    ratio = sleeve_height / sleeve_width

    # heuristic thresholds
    if ratio < 0.6:
        return "sleeveless"

    elif ratio < 1.4:
        return "short"

    return "long"