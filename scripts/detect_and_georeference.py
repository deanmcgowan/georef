#!/usr/bin/env python3
"""
detect_and_georeference.py — End-to-end georeferencing pipeline for scanned
map images with printed coordinate crosses.

Detects coordinate crosses via template matching, assigns map coordinates
based on user-supplied parameters, optionally transforms to a target CRS,
and generates a VRT file plus an HTML quality report.

All detection parameters are dynamically derived from image properties
(DPI, dimensions) and user-supplied grid information.  No map series or
sheet-numbering knowledge is hard-coded.

Usage:
    python scripts/detect_and_georeference.py <image_file> \\
        --source-crs EPSG:3152 --target-crs EPSG:3006 \\
        --grid-origin 99300,77400 --grid-spacing 200,100 \\
        [--grid-size 4x4] [--dpi 300]

Outputs:
    - <review_dir>/<name>_<target_crs_tag>.vrt  — VRT with GCPs (in review/ next to image)
    - <reports_dir>/<name>_report.html           — quality report
"""

import argparse
import base64
import io
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import pyproj


# ---------------------------------------------------------------------------
# Default detection parameters — can be overridden via CLI or derived from DPI
# ---------------------------------------------------------------------------

DEFAULT_DPI = 300

# These reference values are for 300 DPI; the code scales them when the
# actual DPI is different.
REF_TEMPLATE_ARM_PX = 20
REF_TEMPLATE_THICKNESS_PX = 3
REF_NMS_WINDOW_PX = 41
REF_DEDUP_DISTANCE_PX = 50

MIN_TEMPLATE_CORR = 0.70       # Minimum normalised cross-correlation
MIN_CROSS_CONTRAST = 25        # Minimum corner-minus-arm brightness difference
MAX_CENTER_INTENSITY = 160     # Maximum brightness at the cross centre
MAX_ARM_INTENSITY = 170        # Maximum mean arm brightness

SPACING_TOLERANCE = 0.12       # Fractional tolerance for row spacing consistency

GAP_FILL_MIN_CORR = 0.65       # Minimum correlation for gap-filled crosses
GAP_FILL_MAX_DIST_PX = 15      # Maximum distance from affine prediction (pixels)

OUTLIER_THRESHOLD_FACTOR = 2.0  # Outlier threshold = factor × RMS residual


def _scale_factor(dpi: int) -> float:
    """Return a multiplicative scale factor relative to 300 DPI."""
    return dpi / DEFAULT_DPI


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GCP:
    gcp_id: str
    pixel: float
    line: float
    x_src: float    # source CRS easting / X
    y_src: float    # source CRS northing / Y
    x_out: float = 0.0   # output CRS easting / X
    y_out: float = 0.0   # output CRS northing / Y
    corr_score: float = 0.0
    residual_x: float = 0.0
    residual_y: float = 0.0
    residual_total: float = 0.0
    col_idx: int = 0
    row_idx: int = 0


@dataclass
class GridParams:
    """User-supplied (or inferred) grid description."""
    origin_x: float       # map coordinate of the first (top-left) cross
    origin_y: float       # map coordinate of the first (top-left) cross
    spacing_x: float      # easting increment per column (positive → right)
    spacing_y: float      # northing increment per row   (negative → down)
    expected_cols: int = 0   # 0 = auto-detect
    expected_rows: int = 0   # 0 = auto-detect


@dataclass
class DetectionResult:
    image_path: str
    image_w: int
    image_h: int
    source_crs: str
    target_crs: str
    grid: GridParams
    gcps: list = field(default_factory=list)
    affine_coeffs: np.ndarray = None
    rms_residual: float = 0.0
    max_residual: float = 0.0
    outlier_threshold: float = 0.0
    outlier_ids: list = field(default_factory=list)
    transform_pipeline: str = ""
    n_cols: int = 0
    n_rows: int = 0
    col_spacing_px: float = 0.0
    row_spacing_px: float = 0.0
    scale_x: float = 0.0  # pixels per map unit (easting)
    scale_y: float = 0.0  # pixels per map unit (northing)
    dpi: int = DEFAULT_DPI


# ---------------------------------------------------------------------------
# Cross detection
# ---------------------------------------------------------------------------

def _make_cross_template(arm_length: int, thickness: int) -> np.ndarray:
    """Create a synthetic '+' cross template (white bg, black cross)."""
    size = 2 * arm_length + 1
    tpl = np.ones((size, size), dtype=np.uint8) * 255
    c = arm_length
    ht = thickness // 2
    tpl[c - ht:c + ht + 1, c - arm_length:c + arm_length + 1] = 0
    tpl[c - arm_length:c + arm_length + 1, c - ht:c + ht + 1] = 0
    return tpl


def _verify_cross(img: np.ndarray, ix: int, iy: int, arm: int) -> bool:
    """Verify that a candidate point looks like a cross in the image.

    The *arm* parameter controls how far along each axis we sample,
    keeping the check proportional to the template size.
    """
    margin = arm + 10
    if ix < margin or ix >= img.shape[1] - margin:
        return False
    if iy < margin or iy >= img.shape[0] - margin:
        return False

    h_mean = float(img[iy, ix - arm:ix + arm + 1].astype(np.float64).mean())
    v_mean = float(img[iy - arm:iy + arm + 1, ix].astype(np.float64).mean())
    arm_mean = (h_mean + v_mean) / 2.0

    diag_offset = max(arm // 2, 5)
    patch_half = max(arm // 6, 2)
    corners = []
    for dx, dy in [(-diag_offset, -diag_offset), (diag_offset, -diag_offset),
                    (-diag_offset, diag_offset), (diag_offset, diag_offset)]:
        patch = img[iy + dy - patch_half:iy + dy + patch_half + 1,
                    ix + dx - patch_half:ix + dx + patch_half + 1]
        if patch.size == 0:
            return False
        corners.append(float(patch.mean()))
    corner_mean = float(np.mean(corners))

    contrast = corner_mean - arm_mean
    center_val = int(img[iy, ix])

    return (contrast > MIN_CROSS_CONTRAST
            and center_val < MAX_CENTER_INTENSITY
            and arm_mean < MAX_ARM_INTENSITY)


def detect_crosses(img: np.ndarray, dpi: int = DEFAULT_DPI,
                   min_corr: float = MIN_TEMPLATE_CORR) -> list:
    """
    Detect coordinate crosses using template matching.

    Template dimensions are scaled proportionally to the scan DPI.
    Returns list of (x, y, correlation_score) tuples.
    """
    sf = _scale_factor(dpi)
    arm = max(3, int(round(REF_TEMPLATE_ARM_PX * sf)))
    thick = max(1, int(round(REF_TEMPLATE_THICKNESS_PX * sf)))
    nms_win = max(3, int(round(REF_NMS_WINDOW_PX * sf)))
    nms_win = nms_win if nms_win % 2 == 1 else nms_win + 1  # ensure odd
    dedup_dist = max(5, int(round(REF_DEDUP_DISTANCE_PX * sf)))

    tpl = _make_cross_template(arm, thick)
    result = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    offset = arm

    # Non-maximum suppression
    nms_kernel = np.ones((nms_win, nms_win), np.float32)
    dilated = cv2.dilate(result, nms_kernel)
    local_max = (result == dilated) & (result >= min_corr)
    ys, xs = np.where(local_max)
    scores = result[local_max]

    candidates = sorted(zip(xs.astype(float) + offset,
                            ys.astype(float) + offset,
                            scores.astype(float)),
                        key=lambda t: t[2], reverse=True)

    # Verify each candidate
    verified = []
    for cx, cy, corr in candidates:
        if _verify_cross(img, int(round(cx)), int(round(cy)), arm):
            verified.append((cx, cy, corr))

    # De-duplicate (keep highest correlation within dedup_dist)
    unique = []
    for cx, cy, corr in verified:
        if not any(abs(cx - ux) < dedup_dist
                   and abs(cy - uy) < dedup_dist
                   for ux, uy, _ in unique):
            unique.append((cx, cy, corr))

    # Sub-pixel refinement via parabolic interpolation on correlation surface
    refined = []
    for cx, cy, corr in unique:
        rx = int(round(cx - offset))
        ry = int(round(cy - offset))
        if 1 <= rx < result.shape[1] - 1 and 1 <= ry < result.shape[0] - 1:
            fx = [result[ry, rx - 1], result[ry, rx], result[ry, rx + 1]]
            denom = fx[0] - 2 * fx[1] + fx[2]
            dx = 0.5 * (fx[0] - fx[2]) / denom if abs(denom) > 1e-10 else 0.0

            fy = [result[ry - 1, rx], result[ry, rx], result[ry + 1, rx]]
            denom = fy[0] - 2 * fy[1] + fy[2]
            dy = 0.5 * (fy[0] - fy[2]) / denom if abs(denom) > 1e-10 else 0.0

            refined.append((cx + dx, cy + dy, corr))
        else:
            refined.append((cx, cy, corr))

    return refined


def _cluster_1d(values: list, min_gap: float) -> list:
    """Cluster sorted 1D values into groups separated by at least min_gap."""
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] < min_gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return clusters


def _auto_cluster_gap(values: list) -> float:
    """Heuristically determine a reasonable cluster gap.

    Uses half the median pairwise spacing among sorted values, with a
    floor of 20 px to avoid splitting clusters that are very close.
    """
    if len(values) < 2:
        return 50.0
    sorted_v = sorted(values)
    diffs = [sorted_v[i + 1] - sorted_v[i] for i in range(len(sorted_v) - 1)]
    median_diff = float(np.median(diffs))
    return max(20.0, median_diff * 0.5)


def _find_interior_indices(centers: list, image_size: int,
                           expected_count: int = None) -> list:
    """
    Find the consistently-spaced interior elements from a list of cluster centers.

    Returns a list of indices into *centers* that form the best chain of
    evenly-spaced positions.  If *expected_count* is given, excess elements
    nearest to the image edges are trimmed.
    """
    if len(centers) < 3:
        return list(range(len(centers)))

    all_sps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    sp_arr = np.array(all_sps)
    big_sps = sp_arr[sp_arr > sp_arr.max() * 0.5]
    dominant_sp = float(np.median(big_sps)) if len(big_sps) > 0 else float(np.median(sp_arr))

    best_chain: list = []
    for start in range(len(centers)):
        chain = [start]
        for j in range(start + 1, len(centers)):
            sp = centers[j] - centers[chain[-1]]
            n = round(sp / dominant_sp)
            if n >= 1 and abs(sp - n * dominant_sp) < SPACING_TOLERANCE * dominant_sp:
                chain.append(j)
        if len(chain) > len(best_chain):
            best_chain = chain

    interior = best_chain

    if expected_count is not None and len(interior) > expected_count:
        while len(interior) > expected_count:
            near_start = centers[interior[0]]
            near_end = image_size - centers[interior[-1]]
            if near_start < near_end:
                interior = interior[1:]
            else:
                interior = interior[:-1]

    return interior


def organise_grid(crosses: list, image_w: int, image_h: int,
                  expected_rows: int = None,
                  expected_cols: int = None) -> tuple:
    """
    Organise detected crosses into a regular grid.

    The cluster gap is determined automatically from the spacing of the
    detected crosses (half the median neighbour distance).  This makes
    the function independent of any specific map scale or DPI.

    Returns (grid_dict, col_centers, row_centers, interior_rows).
    """
    if len(crosses) < 4:
        raise RuntimeError("Too few crosses detected")

    # Cluster x and y coordinates — gap is derived dynamically
    x_vals = [c[0] for c in crosses]
    y_vals = [c[1] for c in crosses]
    x_gap = _auto_cluster_gap(x_vals)
    y_gap = _auto_cluster_gap(y_vals)

    x_clusters = _cluster_1d(x_vals, x_gap)
    y_clusters = _cluster_1d(y_vals, y_gap)
    all_col_centers = [float(np.median(c)) for c in x_clusters]
    all_row_centers = [float(np.median(c)) for c in y_clusters]

    # Assign each cross to the nearest grid cell using all detected centers
    raw_grid = {}
    for cx, cy, corr in crosses:
        col = int(np.argmin([abs(cx - cc) for cc in all_col_centers]))
        row = int(np.argmin([abs(cy - rc) for rc in all_row_centers]))
        key = (col, row)
        if key not in raw_grid or corr > raw_grid[key][2]:
            raw_grid[key] = (cx, cy, corr)

    # Find interior columns and rows (consistently spaced, not edge artefacts)
    interior_col_idxs = _find_interior_indices(all_col_centers, image_w, expected_cols)
    interior_row_idxs = _find_interior_indices(all_row_centers, image_h, expected_rows)

    # Rebuild col_centers and row_centers as the trimmed interior-only lists
    col_centers = [all_col_centers[i] for i in interior_col_idxs]
    row_centers = [all_row_centers[i] for i in interior_row_idxs]

    # Remap grid keys from original indices to new sequential indices
    col_remap = {old: new for new, old in enumerate(interior_col_idxs)}
    row_remap = {old: new for new, old in enumerate(interior_row_idxs)}
    grid = {}
    for (old_col, old_row), val in raw_grid.items():
        if old_col in col_remap and old_row in row_remap:
            grid[(col_remap[old_col], row_remap[old_row])] = val

    interior_rows = list(range(len(row_centers)))

    return grid, col_centers, row_centers, interior_rows


def refine_grid_positions(img: np.ndarray, grid: dict, col_centers: list,
                          row_centers: list, interior_rows: list,
                          dpi: int = DEFAULT_DPI) -> dict:
    """
    Refine detected positions using a two-pass approach:
    1. Use high-confidence detections to build an affine model
    2. Refine existing detections in tight windows
    3. Gap-fill missing cells only if the found position is close to the
       affine-predicted position and has sufficient correlation
    """
    sf = _scale_factor(dpi)
    arm = max(3, int(round(REF_TEMPLATE_ARM_PX * sf)))
    thick = max(1, int(round(REF_TEMPLATE_THICKNESS_PX * sf)))
    tpl = _make_cross_template(arm, thick)
    offset = arm

    # Re-estimate column/row centers from direct detections only
    for col in range(len(col_centers)):
        xs = [grid[(col, r)][0] for r in interior_rows if (col, r) in grid]
        if xs:
            col_centers[col] = float(np.median(xs))
    for row in interior_rows:
        ys = [grid[(c, row)][1] for c in range(len(col_centers))
              if (c, row) in grid]
        if ys:
            row_centers[row] = float(np.median(ys))

    # Build affine model from direct detections to predict gap positions
    direct_pts = []
    for col in range(len(col_centers)):
        for ri, row in enumerate(interior_rows):
            if (col, row) in grid:
                cx, cy, _ = grid[(col, row)]
                direct_pts.append((col, ri, cx, cy))

    # Fit col_idx, row_idx → pixel, line
    if len(direct_pts) >= 3:
        A_fit = np.array([[col, ri, 1.0] for col, ri, _, _ in direct_pts])
        px_fit = np.array([px for _, _, px, _ in direct_pts])
        py_fit = np.array([py for _, _, _, py in direct_pts])
        cpx, _, _, _ = np.linalg.lstsq(A_fit, px_fit, rcond=None)
        cpy, _, _, _ = np.linalg.lstsq(A_fit, py_fit, rcond=None)
        has_model = True
    else:
        has_model = False

    refined = {}
    for col in range(len(col_centers)):
        for ri, row in enumerate(interior_rows):
            is_direct = (col, row) in grid

            if is_direct:
                cx, cy, corr = grid[(col, row)]
                hw = 15  # tight search window for existing detections
            else:
                if has_model:
                    # Use affine model to predict expected position
                    cx = float(cpx[0] * col + cpx[1] * ri + cpx[2])
                    cy = float(cpy[0] * col + cpy[1] * ri + cpy[2])
                else:
                    cx, cy = col_centers[col], row_centers[row]
                corr = 0.0
                hw = 40  # wider search for gap-filling

            ix, iy = int(round(cx)), int(round(cy))
            x0 = max(0, ix - hw - offset)
            y0 = max(0, iy - hw - offset)
            x1 = min(img.shape[1], ix + hw + offset + 1)
            y1 = min(img.shape[0], iy + hw + offset + 1)
            patch = img[y0:y1, x0:x1]

            if patch.shape[0] <= tpl.shape[0] or patch.shape[1] <= tpl.shape[1]:
                if is_direct:
                    refined[(col, row)] = grid[(col, row)]
                continue

            res = cv2.matchTemplate(patch, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            new_x = float(x0 + max_loc[0] + offset)
            new_y = float(y0 + max_loc[1] + offset)

            if is_direct:
                # Always keep direct detections, use refined position
                refined[(col, row)] = (new_x, new_y, float(max_val))
            else:
                # Gap-fill: require high correlation AND proximity to prediction
                if has_model:
                    pred_x = float(cpx[0] * col + cpx[1] * ri + cpx[2])
                    pred_y = float(cpy[0] * col + cpy[1] * ri + cpy[2])
                    dist = math.sqrt((new_x - pred_x) ** 2 +
                                     (new_y - pred_y) ** 2)
                else:
                    dist = 0.0

                if (max_val >= GAP_FILL_MIN_CORR and dist < GAP_FILL_MAX_DIST_PX and
                        _verify_cross(img, int(round(new_x)),
                                      int(round(new_y)), arm)):
                    refined[(col, row)] = (new_x, new_y, float(max_val))

    return refined


# ---------------------------------------------------------------------------
# Coordinate assignment
# ---------------------------------------------------------------------------

def assign_coordinates(grid: dict, col_centers: list, row_centers: list,
                       interior_rows: list, gparams: GridParams) -> list:
    """
    Assign map coordinates to detected crosses using the user-supplied
    grid origin and spacing.

    origin_x / origin_y  = coordinate of the first (top-left) cross
    spacing_x            = easting increment per column  (usually positive)
    spacing_y            = northing increment per row     (usually negative)
    """
    gcps = []
    gcp_id = 1
    for ri, row in enumerate(interior_rows):
        map_y = gparams.origin_y + ri * gparams.spacing_y
        for col in range(len(col_centers)):
            map_x = gparams.origin_x + col * gparams.spacing_x
            if (col, row) in grid:
                px, py, corr = grid[(col, row)]
                gcps.append(GCP(
                    gcp_id=str(gcp_id), pixel=px, line=py,
                    x_src=float(map_x), y_src=float(map_y),
                    corr_score=corr, col_idx=col, row_idx=ri,
                ))
                gcp_id += 1

    return gcps


# ---------------------------------------------------------------------------
# Inter-consistency analysis
# ---------------------------------------------------------------------------

def fit_affine_and_residuals(gcps: list) -> tuple:
    """
    Fit a 6-parameter affine from pixel/line → map X, Y.
    Returns (coeffs, rms, max_res, threshold, outlier_ids).
    """
    n = len(gcps)
    A = np.zeros((n, 3))
    bx = np.zeros(n)
    by = np.zeros(n)
    for i, g in enumerate(gcps):
        A[i] = [g.pixel, g.line, 1.0]
        bx[i] = g.x_src
        by[i] = g.y_src

    cx, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    cy, _, _, _ = np.linalg.lstsq(A, by, rcond=None)
    coeffs = np.concatenate([cx, cy])

    residuals = []
    for g in gcps:
        pred_x = cx[0] * g.pixel + cx[1] * g.line + cx[2]
        pred_y = cy[0] * g.pixel + cy[1] * g.line + cy[2]
        g.residual_x = g.x_src - pred_x
        g.residual_y = g.y_src - pred_y
        g.residual_total = math.sqrt(g.residual_x ** 2 + g.residual_y ** 2)
        residuals.append(g.residual_total)

    rms = math.sqrt(sum(r ** 2 for r in residuals) / n)
    max_res = max(residuals)
    threshold = OUTLIER_THRESHOLD_FACTOR * rms
    outlier_ids = [g.gcp_id for g in gcps if g.residual_total > threshold]

    return coeffs, rms, max_res, threshold, outlier_ids


# ---------------------------------------------------------------------------
# Coordinate transformation
# ---------------------------------------------------------------------------

def transform_coordinates(gcps: list, source_crs: str,
                          target_crs: str) -> str:
    """Transform GCPs from source CRS to target CRS.  Returns pipeline description."""
    if source_crs == target_crs:
        for g in gcps:
            g.x_out, g.y_out = g.x_src, g.y_src
        return "(identity — source and target CRS are the same)"
    transformer = pyproj.Transformer.from_crs(
        source_crs, target_crs, always_xy=True
    )
    for g in gcps:
        g.x_out, g.y_out = transformer.transform(g.x_src, g.y_src)
    return str(transformer)


# ---------------------------------------------------------------------------
# VRT output
# ---------------------------------------------------------------------------

def write_vrt(gcps: list, image_path: str, image_w: int, image_h: int,
              output_path: str, crs: str = "EPSG:3006"):
    """Write a GDAL VRT file with GCPs in the target CRS.

    The SourceFilename is written relative to the VRT location so that
    the VRT and image can sit in the same directory.
    """
    root = ET.Element("VRTDataset", attrib={
        "rasterXSize": str(image_w),
        "rasterYSize": str(image_h),
    })

    gcp_list = ET.SubElement(root, "GCPList", attrib={"Projection": crs})
    for g in gcps:
        ET.SubElement(gcp_list, "GCP", attrib={
            "Id": g.gcp_id,
            "Pixel": f"{g.pixel:.1f}",
            "Line": f"{g.line:.1f}",
            "X": f"{g.x_out:.3f}",
            "Y": f"{g.y_out:.3f}",
            "Z": "0",
        })

    out_dir = os.path.dirname(os.path.abspath(output_path))
    rel_image = os.path.relpath(os.path.abspath(image_path), out_dir)

    band = ET.SubElement(root, "VRTRasterBand", attrib={
        "dataType": "Byte", "band": "1",
    })
    ET.SubElement(band, "ColorInterp").text = "Gray"
    src = ET.SubElement(band, "SimpleSource")
    ET.SubElement(src, "SourceFilename", attrib={
        "relativeToVRT": "1"
    }).text = rel_image
    ET.SubElement(src, "SourceBand").text = "1"
    ET.SubElement(src, "SrcRect", attrib={
        "xOff": "0", "yOff": "0",
        "xSize": str(image_w), "ySize": str(image_h),
    })
    ET.SubElement(src, "DstRect", attrib={
        "xOff": "0", "yOff": "0",
        "xSize": str(image_w), "ySize": str(image_h),
    })

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=False)
    with open(output_path, "a") as f:
        f.write("\n")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _generate_overlay_image(r: DetectionResult) -> str:
    """Render the GCP overlay plot and return it as a base64-encoded PNG."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    img = Image.open(r.image_path)
    ax.imshow(img, cmap="gray", aspect="equal")

    for g in r.gcps:
        is_out = g.gcp_id in r.outlier_ids
        colour = "red" if is_out else "#00cc44"
        marker = "x" if is_out else "+"
        ax.plot(g.pixel, g.line, marker, color=colour,
                markersize=12, markeredgewidth=2, zorder=5)
        ax.annotate(f" {g.gcp_id}", (g.pixel, g.line),
                    fontsize=7, fontweight="bold", color=colour,
                    ha="left", va="bottom", zorder=6)

    ax.set_title(f"GCP Locations — {os.path.basename(r.image_path)}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Line")

    legend = [
        Line2D([0], [0], marker="+", color="#00cc44", linestyle="None",
               markersize=10, markeredgewidth=2, label="GCP (OK)"),
        Line2D([0], [0], marker="x", color="red", linestyle="None",
               markersize=10, markeredgewidth=2, label="GCP (Outlier)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def generate_html_report(result: DetectionResult, output_path: str):
    """Generate an HTML quality report."""
    r = result
    overlay_b64 = _generate_overlay_image(r)

    quality_rating = ("GOOD" if r.rms_residual < 2.0 else
                      "ACCEPTABLE" if r.rms_residual < 5.0 else "POOR")
    quality_colour = ("#27ae60" if quality_rating == "GOOD" else
                      "#e67e22" if quality_rating == "ACCEPTABLE" else "#c0392b")

    n_outliers = len(r.outlier_ids)
    total_possible = r.n_cols * r.n_rows
    n_detected = len(r.gcps)
    n_missing = total_possible - n_detected
    detection_pct = (n_detected / total_possible * 100) if total_possible else 0

    src_label = html_escape(r.source_crs)
    tgt_label = html_escape(r.target_crs)

    # Build residual table rows
    residual_rows = ""
    for g in r.gcps:
        is_outlier = g.gcp_id in r.outlier_ids
        row_class = ' class="outlier"' if is_outlier else ""
        flag = "Outlier" if is_outlier else "OK"
        residual_rows += (
            f"<tr{row_class}>"
            f"<td>{html_escape(g.gcp_id)}</td>"
            f"<td>{g.pixel:.1f}</td><td>{g.line:.1f}</td>"
            f"<td>{g.x_src:.0f}</td><td>{g.y_src:.0f}</td>"
            f"<td>{g.residual_x:+.3f}</td><td>{g.residual_y:+.3f}</td>"
            f"<td>{g.residual_total:.3f}</td>"
            f"<td>{g.corr_score:.3f}</td>"
            f"<td>{flag}</td>"
            f"</tr>\n"
        )

    # Build transformation table rows
    transform_rows = ""
    for g in r.gcps:
        transform_rows += (
            f"<tr>"
            f"<td>{html_escape(g.gcp_id)}</td>"
            f"<td>{g.x_src:.0f}</td><td>{g.y_src:.0f}</td>"
            f"<td>{g.x_out:.3f}</td><td>{g.y_out:.3f}</td>"
            f"</tr>\n"
        )

    c = r.affine_coeffs
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    gp = r.grid
    spacing_desc = f"{abs(gp.spacing_x)}&nbsp;×&nbsp;{abs(gp.spacing_y)}"

    sf = _scale_factor(r.dpi)
    tpl_arm_px = max(3, int(round(REF_TEMPLATE_ARM_PX * sf)))
    tpl_thick_px = max(1, int(round(REF_TEMPLATE_THICKNESS_PX * sf)))

    html = f"""<!DOCTYPE html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<title>Georeferencing Report — {html_escape(os.path.basename(r.image_path))}</title>
<style>
  :root {{ --accent: #2c3e50; --good: #27ae60; --warn: #e67e22; --bad: #c0392b; --bg: #f8f9fa; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         font-size: 14px; line-height: 1.6; color: #333; background: #fff;
         max-width: 1100px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 22px; color: var(--accent); border-bottom: 2px solid var(--accent);
       padding-bottom: 8px; margin-bottom: 16px; }}
  h2 {{ font-size: 17px; color: var(--accent); margin: 28px 0 10px 0;
       border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  h3 {{ font-size: 15px; color: #555; margin: 18px 0 8px 0; }}
  p, li {{ margin-bottom: 6px; }}
  .summary-box {{ background: var(--bg); border: 1px solid #ddd; border-radius: 6px;
                  padding: 18px 22px; margin: 12px 0; }}
  .quality-badge {{ display: inline-block; padding: 4px 14px; border-radius: 4px;
                    font-weight: bold; font-size: 15px; color: #fff; }}
  .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px 24px; }}
  .meta-grid dt {{ font-weight: 600; color: #555; }}
  .meta-grid dd {{ margin: 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }}
  th {{ background: #34495e; color: #fff; padding: 8px 10px; text-align: center;
       font-weight: 600; }}
  td {{ padding: 6px 10px; text-align: center; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  tr.outlier {{ background: #fce4e4; }}
  .note {{ font-size: 13px; color: #555; margin: 8px 0 16px 0; font-style: italic; }}
  .overlay-img {{ width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 8px 0; }}
  code {{ background: #eef; padding: 1px 5px; border-radius: 3px; font-size: 13px; }}
  .tech-section {{ background: #f4f6f8; border-left: 3px solid var(--accent);
                   padding: 14px 18px; margin: 10px 0; font-size: 13px; }}
  .footer {{ margin-top: 30px; padding-top: 12px; border-top: 1px solid #ddd;
             font-size: 12px; color: #888; text-align: center; }}
  @media print {{ body {{ max-width: 100%; }} }}
</style>
</head>
<body>

<h1>Georeferencing Report</h1>
<p><strong>{html_escape(os.path.basename(r.image_path))}</strong></p>

<!-- ──────────────── EXECUTIVE SUMMARY ──────────────── -->

<h2>Executive Summary</h2>
<div class="summary-box">
  <p>
    <span class="quality-badge" style="background:{quality_colour}">{quality_rating}</span>
    &nbsp; The georeferencing of this image is rated <strong>{quality_rating.lower()}</strong>.
  </p>
  <p>
    {n_detected} ground control points were identified across an
    {r.n_cols}&times;{r.n_rows} grid on the scanned image.{f" {n_missing} grid positions could not be matched, typically where map content obscured the printed crosses." if n_missing else " All expected grid positions were successfully matched."}
    {"One control point was flagged as a statistical outlier and excluded from the quality assessment." if n_outliers == 1 else f"{n_outliers} control points were flagged as statistical outliers." if n_outliers else "No outliers were detected."}
  </p>
  <p>
    The root-mean-square positional error across all control points is
    <strong>{r.rms_residual:.2f}&nbsp;m</strong>.
    The georeferenced output is in
    <strong>{tgt_label}</strong> and is ready for use in GIS.
  </p>
</div>

<dl class="meta-grid">
  <dt>Image file</dt><dd>{html_escape(os.path.basename(r.image_path))}</dd>
  <dt>Image dimensions</dt><dd>{r.image_w} &times; {r.image_h} pixels</dd>
  <dt>Scan DPI</dt><dd>{r.dpi}</dd>
  <dt>Grid spacing (map units)</dt><dd>{spacing_desc}</dd>
  <dt>Source coordinate system</dt><dd>{src_label}</dd>
  <dt>Output coordinate system</dt><dd>{tgt_label}</dd>
  <dt>Control points used</dt><dd>{n_detected} of {total_possible} ({detection_pct:.0f}%)</dd>
  <dt>RMS residual</dt><dd>{r.rms_residual:.3f} m</dd>
  <dt>Maximum residual</dt><dd>{r.max_residual:.3f} m</dd>
</dl>

<!-- ──────────────── GCP OVERLAY ──────────────── -->

<h2>Control Point Overlay</h2>
<p>
  The image below shows each detected control point plotted on the scanned map.
  Green markers indicate accepted points; red markers indicate statistical outliers.
</p>
<img class="overlay-img" src="data:image/png;base64,{overlay_b64}"
     alt="GCP overlay on scanned map">
<p class="note">
  Each control point was identified by template-matching a synthetic cross pattern
  against the scanned image. Positions are refined to sub-pixel accuracy using
  parabolic interpolation on the correlation surface.
</p>

<!-- ──────────────── RESIDUAL ANALYSIS ──────────────── -->

<h2>Residual Analysis</h2>
<p>
  A six-parameter affine transformation was fitted from pixel coordinates to
  source map coordinates. The residuals below show how well each control point
  fits this model. Points exceeding twice the RMS
  ({r.outlier_threshold:.3f}&nbsp;m) are flagged as outliers.
</p>

<table>
  <thead>
    <tr>
      <th>GCP</th><th>Pixel</th><th>Line</th>
      <th>Src&nbsp;X</th><th>Src&nbsp;Y</th>
      <th>Res&nbsp;X&nbsp;(m)</th><th>Res&nbsp;Y&nbsp;(m)</th>
      <th>Res&nbsp;Total&nbsp;(m)</th>
      <th>Correlation</th><th>Status</th>
    </tr>
  </thead>
  <tbody>
{residual_rows}  </tbody>
</table>

<div class="tech-section">
  <h3>Affine model</h3>
  <p>
    X = {c[0]:.6f} &times; pixel + {c[1]:.6f} &times; line + {c[2]:.2f}<br>
    Y = {c[3]:.6f} &times; pixel + {c[4]:.6f} &times; line + {c[5]:.2f}
  </p>
  <p>
    RMS: {r.rms_residual:.3f}&nbsp;m &ensp;|&ensp;
    Max: {r.max_residual:.3f}&nbsp;m &ensp;|&ensp;
    Outlier threshold: {r.outlier_threshold:.3f}&nbsp;m (2&times;RMS)
  </p>
</div>

<!-- ──────────────── COORDINATE TRANSFORMATION ──────────────── -->

<h2>Coordinate Transformation</h2>
<p>
  Each control point was transformed from the source coordinate system
  ({src_label}) to the output system ({tgt_label}).
</p>

<table>
  <thead>
    <tr>
      <th>GCP</th>
      <th>Source&nbsp;X</th><th>Source&nbsp;Y</th>
      <th>Output&nbsp;E</th><th>Output&nbsp;N</th>
    </tr>
  </thead>
  <tbody>
{transform_rows}  </tbody>
</table>

<div class="tech-section">
  <p><strong>PROJ pipeline:</strong><br>
  <code>{html_escape(r.transform_pipeline)}</code></p>
</div>

<!-- ──────────────── METHODOLOGY ──────────────── -->

<h2>Processing Methodology</h2>

<h3>1. Cross detection</h3>
<p>
  A synthetic cross template (arm&nbsp;length&nbsp;{tpl_arm_px}&nbsp;px,
  line&nbsp;thickness&nbsp;{tpl_thick_px}&nbsp;px, scaled for {r.dpi}&nbsp;DPI) was matched against
  the greyscale image using normalised cross-correlation
  (<code>cv2.matchTemplate</code> with <code>TM_CCOEFF_NORMED</code>).
  Peaks above a correlation threshold of {MIN_TEMPLATE_CORR} were extracted
  via non-maximum suppression.
</p>
<p>
  Each candidate was verified by analysing the pixel intensity profile: the
  horizontal and vertical arms must be dark, the centre must be dark, and
  the diagonal corners must be significantly brighter than the arms.
  Duplicate detections were merged, keeping the highest-correlation candidate.
  Positions were refined to sub-pixel accuracy using parabolic interpolation.
</p>

<h3>2. Grid organisation</h3>
<p>
  Detected crosses were clustered into columns and rows using 1D gap-based
  clustering with an automatically determined gap threshold. The interior grid
  was identified by finding the longest chain of consistently spaced
  positions (tolerance&nbsp;{SPACING_TOLERANCE * 100:.0f}%).
</p>

<h3>3. Affine-validated gap filling</h3>
<p>
  A least-squares affine model was fitted from the directly detected crosses.
  For each empty grid cell, a local template search was performed around the
  affine-predicted position. A gap-filled point was accepted only if
  its correlation exceeded {GAP_FILL_MIN_CORR} <em>and</em> its position was
  within {GAP_FILL_MAX_DIST_PX}&nbsp;px of the prediction.
</p>

<h3>4. Coordinate assignment</h3>
<p>
  Map coordinates were assigned based on each point&rsquo;s grid position
  using the supplied grid origin ({gp.origin_x},&nbsp;{gp.origin_y}) and
  spacing ({gp.spacing_x},&nbsp;{gp.spacing_y}).
</p>

<h3>5. Transformation</h3>
<p>
  Coordinates were transformed from {src_label} to {tgt_label}
  using <code>pyproj.Transformer</code>.
</p>

<h3>6. Output</h3>
<p>
  The final VRT file contains the transformed GCPs and can be opened
  directly in QGIS or processed with GDAL tools.
</p>

<h3>Tools and libraries</h3>
<ul>
  <li><strong>OpenCV</strong> (<code>cv2</code>) &mdash; template matching, non-maximum suppression</li>
  <li><strong>NumPy</strong> &mdash; array operations, least-squares fitting</li>
  <li><strong>Pillow</strong> (<code>PIL</code>) &mdash; image loading</li>
  <li><strong>pyproj</strong> &mdash; coordinate transformation (PROJ wrapper)</li>
  <li><strong>Matplotlib</strong> &mdash; overlay plot generation</li>
</ul>

<div class="footer">
  Report generated {timestamp} by <code>detect_and_georeference.py</code>.
</div>

</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)



# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _get_image_dpi(img_pil: Image.Image) -> int:
    """Attempt to read DPI from image metadata; fall back to DEFAULT_DPI."""
    dpi_info = img_pil.info.get("dpi")
    if dpi_info and isinstance(dpi_info, (tuple, list)) and dpi_info[0] > 0:
        return int(round(dpi_info[0]))
    return DEFAULT_DPI


def _crs_tag(crs: str) -> str:
    """Turn an EPSG code into a filename-safe tag, e.g. 'EPSG:3006' → 'EPSG3006'."""
    return crs.replace(":", "")


def run_pipeline(image_path: str, source_crs: str, target_crs: str,
                 grid_origin: tuple, grid_spacing: tuple,
                 grid_size: tuple = None,
                 dpi_override: int = None,
                 review_dir: str = None,
                 reports_dir: str = None):
    """Run the full detection → georeferencing → reporting pipeline.

    Parameters
    ----------
    image_path : str
        Path to the scanned map image.
    source_crs : str
        EPSG code or PROJ string for the source CRS (e.g. "EPSG:3152").
    target_crs : str
        EPSG code or PROJ string for the output CRS (e.g. "EPSG:3006").
    grid_origin : (float, float)
        Map coordinates of the first (top-left) cross: (x, y).
    grid_spacing : (float, float)
        (dx per column, dy per row).  dy is typically negative (north → south).
    grid_size : (int, int) or None
        Expected (columns, rows).  None = auto-detect.
    dpi_override : int or None
        Override the DPI read from the image metadata.
    review_dir : str or None
        Directory for the VRT output (default: review/ at repo root).
    reports_dir : str or None
        Directory for the HTML report (default: reports/ at repo root).
    """
    image_path = os.path.abspath(image_path)
    if not os.path.isfile(image_path):
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    basename = os.path.basename(image_path)
    name_no_ext = os.path.splitext(basename)[0]
    safe_name = name_no_ext.replace(" ", "_")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if review_dir is None:
        review_dir = os.path.join(repo_root, "review")
    if reports_dir is None:
        reports_dir = os.path.join(repo_root, "reports")
    os.makedirs(review_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    gparams = GridParams(
        origin_x=grid_origin[0], origin_y=grid_origin[1],
        spacing_x=grid_spacing[0], spacing_y=grid_spacing[1],
        expected_cols=grid_size[0] if grid_size else 0,
        expected_rows=grid_size[1] if grid_size else 0,
    )

    # ── 1. Load image ──
    print(f"Image: {basename}")
    img_pil = Image.open(image_path)
    img = np.array(img_pil)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape[:2]

    dpi = dpi_override if dpi_override else _get_image_dpi(img_pil)
    print(f"  Size: {w}×{h} pixels  DPI: {dpi}")

    # ── 2. Detect crosses ──
    print("Detecting coordinate crosses...")
    crosses = detect_crosses(img, dpi=dpi)
    print(f"  Raw detections: {len(crosses)}")

    # ── 3. Organise grid ──
    exp_cols = gparams.expected_cols if gparams.expected_cols > 0 else None
    exp_rows = gparams.expected_rows if gparams.expected_rows > 0 else None
    grid, col_centers, row_centers, interior_rows = organise_grid(
        crosses, w, h,
        expected_rows=exp_rows,
        expected_cols=exp_cols)
    print(f"  Grid: {len(col_centers)} cols × {len(interior_rows)} rows")

    # ── 4. Refine positions ──
    print("Refining cross positions...")
    refined = refine_grid_positions(img, grid, col_centers,
                                    row_centers, interior_rows, dpi=dpi)
    print(f"  Refined cells: {len(refined)}")

    # ── 5. Assign coordinates ──
    print("Assigning map coordinates...")
    gcps = assign_coordinates(refined, col_centers, row_centers,
                              interior_rows, gparams)
    print(f"  GCPs: {len(gcps)}")

    # ── 6. Inter-consistency ──
    print("Inter-consistency analysis...")
    coeffs, rms, max_res, threshold, outliers = fit_affine_and_residuals(gcps)
    print(f"  RMS: {rms:.3f} m  Max: {max_res:.3f} m")
    if outliers:
        print(f"  Outliers: {outliers}")

    # ── 7. Transform to target CRS ──
    print(f"Transforming {source_crs} → {target_crs}...")
    pipeline = transform_coordinates(gcps, source_crs, target_crs)

    # ── 8. Compute scale stats ──
    col_sps = [col_centers[i + 1] - col_centers[i]
               for i in range(len(col_centers) - 1)]
    row_sps = [row_centers[interior_rows[i + 1]] - row_centers[interior_rows[i]]
               for i in range(len(interior_rows) - 1)]
    abs_spacing_x = abs(gparams.spacing_x) if gparams.spacing_x != 0 else 1
    abs_spacing_y = abs(gparams.spacing_y) if gparams.spacing_y != 0 else 1
    scale_x = float(np.mean(col_sps) / abs_spacing_x) if col_sps else 0.0
    scale_y = float(np.mean(row_sps) / abs_spacing_y) if row_sps else 0.0

    result = DetectionResult(
        image_path=image_path, image_w=w, image_h=h,
        source_crs=source_crs, target_crs=target_crs,
        grid=gparams,
        gcps=gcps, affine_coeffs=coeffs,
        rms_residual=rms, max_residual=max_res,
        outlier_threshold=threshold, outlier_ids=outliers,
        transform_pipeline=pipeline,
        n_cols=len(col_centers), n_rows=len(interior_rows),
        col_spacing_px=float(np.mean(col_sps)) if col_sps else 0.0,
        row_spacing_px=float(np.mean(row_sps)) if row_sps else 0.0,
        scale_x=scale_x, scale_y=scale_y,
        dpi=dpi,
    )

    # ── 9. Write outputs ──
    crs_tag = _crs_tag(target_crs)
    vrt_path = os.path.join(review_dir, f"{safe_name}_{crs_tag}.vrt")
    print(f"Writing VRT: {vrt_path}")
    write_vrt(gcps, image_path, w, h, vrt_path, crs=target_crs)

    report_path = os.path.join(reports_dir, f"{safe_name}_report.html")
    print(f"Generating report: {report_path}")
    generate_html_report(result, report_path)

    print(f"\nDone!  {len(gcps)} GCPs, RMS={rms:.3f} m")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Detect coordinate crosses and georeference a scanned "
                    "map image."
    )
    parser.add_argument("image", help="Path to the scanned map image (JPG/PNG/TIFF)")
    parser.add_argument("--source-crs", required=True,
                        help="Source CRS, e.g. EPSG:3152")
    parser.add_argument("--target-crs", default="EPSG:3006",
                        help="Target CRS (default: EPSG:3006)")
    parser.add_argument("--grid-origin", required=True,
                        help="Map coordinates of the first (top-left) cross, "
                             "comma-separated: X,Y")
    parser.add_argument("--grid-spacing", required=True,
                        help="Grid spacing per column and per row, "
                             "comma-separated: dX,dY  "
                             "(dY is typically negative for north-to-south)")
    parser.add_argument("--grid-size", default=None,
                        help="Expected grid dimensions: COLSxROWS "
                             "(e.g. 4x4). Omit for auto-detection.")
    parser.add_argument("--dpi", type=int, default=None,
                        help="Override the image DPI (default: read from metadata)")
    parser.add_argument("--review-dir", default=None,
                        help="Directory for VRT output (default: review/)")
    parser.add_argument("--reports-dir", default=None,
                        help="Directory for HTML report (default: reports/)")
    args = parser.parse_args()

    origin = tuple(float(v) for v in args.grid_origin.split(","))
    spacing = tuple(float(v) for v in args.grid_spacing.split(","))
    grid_size = None
    if args.grid_size:
        parts = args.grid_size.lower().split("x")
        grid_size = (int(parts[0]), int(parts[1]))

    run_pipeline(
        args.image,
        source_crs=args.source_crs,
        target_crs=args.target_crs,
        grid_origin=origin,
        grid_spacing=spacing,
        grid_size=grid_size,
        dpi_override=args.dpi,
        review_dir=args.review_dir,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
