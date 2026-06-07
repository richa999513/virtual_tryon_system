"""Clothing-agnostic image creation."""

import logging
from typing import List, Optional

import cv2
import numpy as np
from fashn_human_parser import BODY_COVERAGE_TO_LABELS, IDENTITY_LABELS, LABELS_TO_IDS

from ..utils import setup_logger
from .masks import asymmetric_dilate_mask, create_bounded_mask, create_contour_following_mask, dilate_mask

# Re-export constants from fashn_human_parser for convenience
FASHN_LABELS_TO_IDS = LABELS_TO_IDS
BODY_COVERAGE_TO_FASHN_LABELS = BODY_COVERAGE_TO_LABELS
IDENTITY_FASHN_LABELS = tuple(IDENTITY_LABELS)


def _default(val, default_val):
    """Return val if not None, else default_val (or call it if callable)."""
    if val is not None:
        return val
    return default_val() if callable(default_val) else default_val


def _create_hybrid_contour_bounded_mask(
    contour_mask: np.ndarray,
    bounded_mask: np.ndarray,
    min_distance_threshold: float = 100.0,
    logger: Optional[logging.Logger] = None,
    baseline_height: float = 864.0,
) -> np.ndarray:
    """
    Create hybrid mask by removing over-aggressive bounded expansions.

    Combines contour-following and bounding-box masks, removing pixels from
    the bounded mask that are too far from the contour mask.

    Args:
        contour_mask: Precise contour-following mask
        bounded_mask: More aggressive bounded box mask
        min_distance_threshold: Max distance from contour for bounded pixels (at baseline height)
        logger: Optional logger instance
        baseline_height: Reference height for scaling threshold

    Returns:
        Hybrid mask with over-aggressive bounded pixels removed
    """
    import cv2

    logger = _default(logger, lambda: setup_logger("hybrid_mask"))

    # Scale threshold based on image height
    height_scale = contour_mask.shape[0] / baseline_height
    scaled_threshold = min_distance_threshold * height_scale

    if scaled_threshold <= 0:
        logger.debug("scaled_threshold<=0, returning pure contour mask")
        return contour_mask

    hybrid_mask = bounded_mask.copy()

    # Find pixels in bounded but not in contour (potential over-expansion)
    bounded_extra = bounded_mask & ~contour_mask

    if not np.any(bounded_extra):
        logger.debug("No extra pixels in bounded mask, returning bounded mask")
        return bounded_mask

    # Compute distance from bounded extra pixels to nearest contour mask pixel
    distance_from_contour = cv2.distanceTransform(
        (~contour_mask).astype(np.uint8), cv2.DIST_L2, 5
    )

    # Remove pixels too far from contour
    bounded_extra_coords = np.where(bounded_extra)
    extra_distances = distance_from_contour[bounded_extra]
    remove_mask = extra_distances > scaled_threshold
    remove_coords = (bounded_extra_coords[0][remove_mask], bounded_extra_coords[1][remove_mask])

    hybrid_mask[remove_coords] = False

    return hybrid_mask


def create_garment_image(
    img_np: np.ndarray,
    seg_pred: np.ndarray,
    labels_to_segment_indices: List[int],
    mask_value: int = 127,
    disable_masking: bool = False,
    min_seg_area_ratio: float = 0.01,
) -> np.ndarray:
    """
    Create garment image with optional masking.

    Masks out regions not belonging to the specified garment labels.

    Args:
        img_np: Input image array (will be modified in-place)
        seg_pred: Segmentation prediction array
        labels_to_segment_indices: List of label indices to keep
        mask_value: Value to fill masked regions (default: 127 gray)
        disable_masking: If True, return image unchanged

    Returns:
        Processed garment image array
    """
    if disable_masking:
        return _neutralize_flat_lay_background(img_np, mask_value=mask_value)

    selected_labels_mask = np.isin(seg_pred, labels_to_segment_indices)

    if selected_labels_mask.mean() < min_seg_area_ratio:
        selected_labels_mask = _estimate_foreground_mask(img_np)

    if np.any(selected_labels_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        selected_labels_mask = cv2.morphologyEx(
            selected_labels_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
        selected_labels_mask = dilate_mask(selected_labels_mask, kernel=(5, 5))
        img_np[~selected_labels_mask] = mask_value

    return img_np


def _neutralize_flat_lay_background(img_np: np.ndarray, mask_value: int = 127) -> np.ndarray:
    """Replace obvious non-garment background in product shots with neutral gray."""
    foreground_mask = _estimate_foreground_mask(img_np)
    if foreground_mask.mean() < 0.01 or foreground_mask.mean() > 0.95:
        return img_np

    result = img_np.copy()
    result[~foreground_mask] = mask_value
    return result


def _estimate_foreground_mask(img_np: np.ndarray) -> np.ndarray:
    """
    Conservative foreground estimate for flat-lay/non-white-background garments.

    This is intentionally simple and dependency-free. It uses border color as
    the likely background, then keeps the largest central foreground components.
    """
    h, w = img_np.shape[:2]
    border = np.concatenate(
        [
            img_np[: max(1, h // 24), :, :].reshape(-1, 3),
            img_np[-max(1, h // 24) :, :, :].reshape(-1, 3),
            img_np[:, : max(1, w // 24), :].reshape(-1, 3),
            img_np[:, -max(1, w // 24) :, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    bg_color = np.median(border, axis=0)
    color_dist = np.linalg.norm(img_np.astype(np.float32) - bg_color, axis=2)

    threshold = max(18.0, float(np.percentile(color_dist, 68)))
    foreground = color_dist > threshold

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    foreground = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    if num_labels <= 1:
        return foreground.astype(bool)

    image_center = np.array([w / 2.0, h / 2.0])
    keep = np.zeros(num_labels, dtype=bool)
    min_area = max(64, int(h * w * 0.0025))

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        center_dist = np.linalg.norm(np.array(centroids[label]) - image_center)
        if center_dist < max(h, w) * 0.42 or area > h * w * 0.03:
            keep[label] = True

    return keep[labels]


# def create_clothing_agnostic_image(
#     img_np: np.ndarray,
#     seg_pred: np.ndarray,
#     labels_to_segment_indices: List[int],
#     body_coverage: str,
#     mask_value: int = 127,
#     disable_masking: bool = False,
#     min_distance_threshold: float = 100.0,
#     baseline_height: float = 864.0,
#     mask_limbs: bool = True,
#     logger: Optional[logging.Logger] = None,
# ) -> np.ndarray:
#     """
#     Create clothing-agnostic image.

#     Masks garments and body parts based on the target category.

#     Args:
#         img_np: Input image array (will be modified in-place)
#         seg_pred: Segmentation prediction array
#         labels_to_segment_indices: List of label indices to mask
#         body_coverage: Coverage type ("full", "upper", or "lower")
#         mask_value: Value to fill masked regions (default: 127 gray)
#         disable_masking: If True, return image unchanged
#         min_distance_threshold: Distance threshold for hybrid mask (at baseline height)
#         baseline_height: Reference height for parameter scaling
#         mask_limbs: If True, also mask arms/legs based on body_coverage
#         logger: Optional logger instance

#     Returns:
#         Clothing-agnostic image array
#     """
#     logger = _default(logger, lambda: setup_logger("clothing_agnostic"))

#     if disable_masking:
#         return img_np

#     # Scale parameters based on image height
#     height_scale = seg_pred.shape[0] / baseline_height
#     logger.debug(f"Height scale factor: {height_scale:.3f} (height: {seg_pred.shape[0]})")

#     # Add body parts to mask based on body coverage
#     labels_ids_dict = FASHN_LABELS_TO_IDS.copy()
#     if mask_limbs:
#         if body_coverage in ("full", "upper"):
#             labels_to_segment_indices += [labels_ids_dict["arms"], labels_ids_dict["torso"]]
#         if body_coverage in ("full", "lower"):
#             labels_to_segment_indices += [labels_ids_dict["legs"]]

#     # Create base mask
#     mask = np.isin(seg_pred, labels_to_segment_indices)

#     # Buffer mask to avoid leaks
#     scaled_buffer_kernel = max(1, int(4 * height_scale))
#     buffer_mask = dilate_mask(mask, kernel=(scaled_buffer_kernel, scaled_buffer_kernel))

#     # Create bounded mask
#     bounded_mask = create_bounded_mask(mask)

#     # Create contour following mask
#     scaled_brush_radius = max(1, int(18 * height_scale))
#     contour_mask = create_contour_following_mask(mask, brush_radius=scaled_brush_radius)

#     # Create hybrid mask
#     ca_mask = _create_hybrid_contour_bounded_mask(
#         contour_mask, bounded_mask, logger=logger, min_distance_threshold=min_distance_threshold
#     )

#     # Apply asymmetric dilation for inpainting workspace
#     scaled_right = int(33 * height_scale)
#     scaled_left = int(33 * height_scale)
#     scaled_up = int(16 * height_scale)
#     scaled_down = int(16 * height_scale)
#     ca_mask = asymmetric_dilate_mask(ca_mask, right=scaled_right, left=scaled_left, up=scaled_up, down=scaled_down)

#     # Create exclusion mask (regions to preserve)
#     identity_ids = [labels_ids_dict[label] for label in IDENTITY_FASHN_LABELS]

#     # Conditional identity based on coverage
#     if body_coverage == "upper":
#         identity_ids.append(labels_ids_dict["legs"])
#     elif body_coverage == "lower":
#         identity_ids.append(labels_ids_dict["arms"])

#     exclusion_mask = np.isin(seg_pred, identity_ids)

#     # Handle hands and feet
#     if body_coverage in ("full", "upper"):
#         hands_mask = seg_pred == labels_ids_dict["hands"]
#         exclusion_mask = exclusion_mask | hands_mask

#     if body_coverage in ("full", "lower"):
#         feet_mask = seg_pred == labels_ids_dict["feet"]
#         exclusion_mask = exclusion_mask | feet_mask

#     final_mask = buffer_mask | (ca_mask & ~exclusion_mask)
#     img_np[final_mask] = mask_value

#     return img_np


def create_clothing_agnostic_image(
    img_np: np.ndarray,
    seg_pred: np.ndarray,
    labels_to_segment_indices: List[int],
    body_coverage: str,
    mask_value: int = 127,
    disable_masking: bool = False,
    min_distance_threshold: float = 100.0,
    baseline_height: float = 864.0,
    mask_limbs: bool = True,
    logger: Optional[logging.Logger] = None,
    long_garment: bool = False,
    sleeve_type: str = "long"
) -> np.ndarray:
    """
    Create clothing-agnostic image.

    Updated version with support for:
    - long tops
    - kurtis
    - oversized shirts
    - dresses
    - midi garments
    """

    logger = _default(logger, lambda: setup_logger("clothing_agnostic"))

    if disable_masking:
        return img_np

    labels_ids_dict = FASHN_LABELS_TO_IDS.copy()

    height, width = seg_pred.shape[:2]
    height_scale = height / baseline_height

    logger.debug(f"Height scale factor: {height_scale:.3f}")
    logger.debug(f"Long garment mode: {long_garment}")

    base_mask = np.isin(seg_pred, labels_to_segment_indices)
    extra_mask = np.zeros_like(base_mask)

    # =========================================================
    # EXPANDED BODY REGION MASKING
    # =========================================================

    if mask_limbs:

        if body_coverage in ("full", "upper"):

            labels_to_segment_indices += [
                labels_ids_dict["torso"],
            ]

            # =====================================================
            # DYNAMIC ARM MASKING
            # =====================================================

            if sleeve_type == "long":
                labels_to_segment_indices += [
                    labels_ids_dict["arms"]
                ]

            elif sleeve_type == "short":

                arms_mask = seg_pred == labels_ids_dict["arms"]

                ys, xs = np.where(arms_mask)

                if len(ys) > 0:

                    cutoff = ys.min() + int(
                        (ys.max() - ys.min()) * 0.45
                    )

                    short_arm_mask = np.zeros_like(arms_mask)
                    short_arm_mask[ys.min():cutoff, :] = arms_mask[ys.min():cutoff, :]
                    extra_mask |= short_arm_mask

            # sleeveless:
            # do not mask arms heavily

        if body_coverage in ("full", "lower"):
            labels_to_segment_indices += [
                labels_ids_dict["legs"]
            ]

    # =========================================================
    # BASE MASK
    # =========================================================

    mask = np.isin(seg_pred, labels_to_segment_indices) | extra_mask

    # =========================================================
    # IMPORTANT FIX:
    # EXTEND LONG GARMENTS DOWNWARD
    # =========================================================

    if long_garment and body_coverage == "upper":
        torso_mask = (seg_pred == labels_ids_dict["torso"]) | base_mask

        ys, xs = np.where(torso_mask)

        if len(ys) > 0:
            bottom_y = ys.max()
            left_x = xs.min()
            right_x = xs.max()

            extension = int(height * 0.28)

            extended_bottom = min(height, bottom_y + extension)

            extended_region = np.zeros_like(mask)

            extended_region[
                bottom_y:extended_bottom,
                left_x:right_x,
            ] = True

            mask = mask | extended_region

    if body_coverage == "full":
        torso_or_cloth_mask = (
            base_mask
            | (seg_pred == labels_ids_dict["torso"])
            | (seg_pred == labels_ids_dict["legs"])
        )
        ys, xs = np.where(torso_or_cloth_mask)
        if len(ys) > 0:
            left_x = xs.min()
            right_x = xs.max()
            top_y = ys.min()
            bottom_y = min(height, ys.max() + int(height * 0.08))
            full_region = np.zeros_like(mask)
            full_region[top_y:bottom_y, left_x:right_x] = True
            mask |= full_region

    # =========================================================
    # BUFFER MASK
    # =========================================================

    scaled_buffer_kernel = max(3, int(8 * height_scale))

    buffer_mask = dilate_mask(
        mask,
        kernel=(scaled_buffer_kernel, scaled_buffer_kernel),
    )

    # =========================================================
    # BOUNDED MASK
    # =========================================================

    bounded_mask = create_bounded_mask(mask)

    # =========================================================
    # CONTOUR MASK
    # =========================================================

    scaled_brush_radius = max(8, int(28 * height_scale))

    contour_mask = create_contour_following_mask(
        mask,
        brush_radius=scaled_brush_radius,
    )

    # =========================================================
    # HYBRID MASK
    # =========================================================

    ca_mask = _create_hybrid_contour_bounded_mask(
        contour_mask,
        bounded_mask,
        logger=logger,
        min_distance_threshold=min_distance_threshold,
    )

    # =========================================================
    # IMPORTANT FIX:
    # STRONGER DOWNWARD DILATION
    # =========================================================

    scaled_right = int(42 * height_scale)
    scaled_left = int(42 * height_scale)
    scaled_up = int(22 * height_scale)

    # CRITICAL FIX
    # =========================================================
    # LOWER BODY SPECIAL HANDLING
    # =========================================================

    if body_coverage == "lower":

        ys, xs = np.where(mask)

        if len(xs) > 0:
            garment_width = xs.max() - xs.min()
            garment_height = ys.max() - ys.min()

            width_ratio = garment_width / max(garment_height, 1)

            # Wide lower garments:
            # skirts, palazzos, flared bottoms
            if width_ratio > 0.55:
                scaled_right = int(65 * height_scale)
                scaled_left = int(65 * height_scale)

    # =========================================================
    # DOWNWARD DILATION
    # =========================================================

    if long_garment:
        scaled_down = int(90 * height_scale)

    elif body_coverage == "full":
        scaled_down = int(70 * height_scale)

    elif body_coverage == "lower":
        scaled_down = int(55 * height_scale)

    else:
        scaled_down = int(40 * height_scale)

    ca_mask = asymmetric_dilate_mask(
        ca_mask,
        right=scaled_right,
        left=scaled_left,
        up=scaled_up,
        down=scaled_down,
    )

    # =========================================================
    # EXCLUSION MASK
    # =========================================================

    identity_ids = [
        labels_ids_dict[label]
        for label in IDENTITY_FASHN_LABELS
    ]

    if body_coverage == "upper":
        identity_ids.append(labels_ids_dict["legs"])

    elif body_coverage == "lower":
        identity_ids.append(labels_ids_dict["arms"])

    exclusion_mask = np.isin(seg_pred, identity_ids)

    # Keep hands
    if body_coverage in ("full", "upper"):
        hands_mask = seg_pred == labels_ids_dict["hands"]
        exclusion_mask = exclusion_mask | hands_mask

    # Keep feet
    if body_coverage in ("full", "lower"):
        feet_mask = seg_pred == labels_ids_dict["feet"]
        exclusion_mask = exclusion_mask | feet_mask

    # =========================================================
    # FINAL MASK
    # =========================================================

    final_mask = buffer_mask | (ca_mask & ~exclusion_mask)

    img_np[final_mask] = mask_value

    return img_np
