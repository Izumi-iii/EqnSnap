#!/usr/bin/env python3
"""Portable Otsu and grid-perimeter contract shared with the Swift client."""

from typing import Any

import numpy as np


def otsu_threshold(pixels: np.ndarray) -> int:
    if pixels.ndim != 2 or pixels.size == 0:
        raise ValueError("Expected a non-empty 2D grayscale image")
    if pixels.dtype != np.uint8:
        raise ValueError("Expected uint8 grayscale pixels")

    histogram = np.bincount(pixels.ravel(), minlength=256).astype(np.int64)
    levels = np.arange(256, dtype=np.int64)
    total_count = int(pixels.size)
    total_sum = int(np.dot(levels, histogram))
    left_count = 0
    left_sum = 0
    best_score = -1.0
    best_threshold = 0

    for threshold in range(256):
        count = int(histogram[threshold])
        left_count += count
        left_sum += threshold * count
        right_count = total_count - left_count
        if left_count == 0 or right_count == 0:
            continue

        difference = total_sum * left_count - left_sum * total_count
        score = float(difference) ** 2 / (left_count * right_count)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold


def grid_perimeter(mask: np.ndarray) -> int:
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("Expected a non-empty 2D foreground mask")

    foreground = mask.astype(bool, copy=False)
    perimeter = int(
        foreground[0, :].sum()
        + foreground[-1, :].sum()
        + foreground[:, 0].sum()
        + foreground[:, -1].sum()
        + np.logical_xor(foreground[1:, :], foreground[:-1, :]).sum()
        + np.logical_xor(foreground[:, 1:], foreground[:, :-1]).sum()
    )
    return perimeter


def stroke_profile_metrics(pixels: np.ndarray) -> dict[str, Any]:
    threshold = otsu_threshold(pixels)
    foreground = pixels <= threshold
    area = int(foreground.sum())
    perimeter = grid_perimeter(foreground)
    if area == 0 or perimeter == 0:
        raise ValueError("Otsu segmentation produced no measurable foreground")

    stroke_width = 2.0 * area / perimeter
    return {
        "otsu_threshold": threshold,
        "foreground_area": area,
        "foreground_perimeter": perimeter,
        "stroke_width_area_perimeter": stroke_width,
        "relative_stroke_width": stroke_width / pixels.shape[0],
    }
