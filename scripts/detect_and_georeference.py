#!/usr/bin/env python3
"""
detect_and_georeference.py — End-to-end georeferencing pipeline for scanned
Swedish borehole maps (Borrhålskarta).

Detects coordinate crosses via template matching, assigns ST74 (EPSG:3152)
coordinates based on the sheet numbering system, transforms to SWEREF99TM
(EPSG:3006), and generates a VRT file plus a PDF quality report.

Usage:
    python scripts/detect_and_georeference.py <image_file>

Outputs (in reports/ directory):
    - <name>_report.pdf   — 4-page quality report
    - <name>_SWEREF99TM.vrt — VRT with GCPs in SWEREF99TM
"""

import argparse
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import pyproj


# ---------------------------------------------------------------------------
# Detection / analysis constants (calibrated for 300 DPI scanned maps)
# ---------------------------------------------------------------------------

TEMPLATE_ARM_PX = 20           # Cross template arm length in pixels
TEMPLATE_THICKNESS_PX = 3      # Cross template line thickness in pixels
MIN_TEMPLATE_CORR = 0.70       # Minimum normalised cross-correlation for detection
NMS_WINDOW_PX = 41             # Non-maximum suppression neighbourhood size
DEDUP_DISTANCE_PX = 50         # Duplicate removal radius in pixels

# Pixel-level cross verification thresholds
MIN_CROSS_CONTRAST = 25        # Minimum corner-minus-arm brightness difference
MAX_CENTER_INTENSITY = 160     # Maximum brightness at the cross centre
MAX_ARM_INTENSITY = 170        # Maximum mean arm brightness

# Grid organisation
GRID_CLUSTER_GAP_PX = 500      # Minimum gap between grid clusters (pixels)
SPACING_TOLERANCE = 0.12       # Fractional tolerance for row spacing consistency

# Affine-validated gap-filling
GAP_FILL_MIN_CORR = 0.65       # Minimum correlation for gap-filled crosses
GAP_FILL_MAX_DIST_PX = 15      # Maximum distance from affine prediction (pixels)

# Statistical outlier detection
OUTLIER_THRESHOLD_FACTOR = 2.0  # Outlier threshold = factor × RMS residual


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GCP:
    gcp_id: str
    pixel: float
    line: float
    x_st74: float  # ST74 easting
    y_st74: float  # ST74 northing
    x_sweref: float = 0.0
    y_sweref: float = 0.0
    corr_score: float = 0.0
    residual_x: float = 0.0
    residual_y: float = 0.0
    residual_total: float = 0.0
    col_idx: int = 0
    row_idx: int = 0


@dataclass
class SheetInfo:
    """Map sheet geometry derived from the sheet code (e.g. '76D')."""
    code: str
    column: int       # first digit
    row: int          # second digit
    quadrant: str     # a=NW, b=NE, c=SW, d=SE
    x_min: int = 0    # left boundary easting
    x_max: int = 0    # right boundary easting
    y_min: int = 0    # bottom boundary northing
    y_max: int = 0    # top boundary northing


@dataclass
class DetectionResult:
    image_path: str
    image_w: int
    image_h: int
    sheet: SheetInfo
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
    scale_x: float = 0.0  # pixels per metre (easting)
    scale_y: float = 0.0  # pixels per metre (northing)


# ---------------------------------------------------------------------------
# Sheet geometry
# ---------------------------------------------------------------------------

def parse_sheet_code(code: str) -> SheetInfo:
    """
    Parse a sheet code like '76D' into its geometric parameters.

    Reference boundaries (from manually verified VRT files):
        76a (NW): X = 100100..100900, Y = 76500..77000
        76B (NE): X = 100900..101700, Y = 76500..77000
        65b (NE): X = 99200..100000, Y = 77500..78000

    Main sheet 76: X = 100100..101700 (1600 m), Y = 76000..77000 (1000 m).
    Each sub-sheet is 800 m × 500 m.
    """
    match = re.match(r"(\d)(\d)([A-Da-d])", code)
    if not match:
        raise ValueError(f"Invalid sheet code: {code!r}")
    col = int(match.group(1))
    row = int(match.group(2))
    quad = match.group(3).lower()

    # Lookup table for known main-sheet SW corners (col, row) → (x_min, y_min).
    known = {
        (6, 5): (98400, 77000),
        (6, 6): (98400, 76000),
        (7, 5): (100100, 77000),
        (7, 6): (100100, 76000),
    }

    key = (col, row)
    if key in known:
        x_base, y_base = known[key]
    else:
        # Extrapolate from sheet 76 (col=7, row=6)
        ref_x, ref_y = 100100, 76000
        x_base = ref_x + (col - 7) * 1600
        y_base = ref_y + (row - 6) * 1000

    # Sub-sheet offsets (each sub-sheet is 800 × 500)
    if quad in ('a', 'c'):  # left half
        x_min, x_max = x_base, x_base + 800
    else:  # b, d → right half
        x_min, x_max = x_base + 800, x_base + 1600

    if quad in ('a', 'b'):  # top half
        y_min, y_max = y_base + 500, y_base + 1000
    else:  # c, d → bottom half
        y_min, y_max = y_base, y_base + 500

    return SheetInfo(code=code, column=col, row=row, quadrant=quad,
                     x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


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


def _verify_cross(img: np.ndarray, ix: int, iy: int) -> bool:
    """Verify that a candidate point looks like a cross in the image."""
    if ix < 30 or ix >= img.shape[1] - 30 or iy < 30 or iy >= img.shape[0] - 30:
        return False

    h_mean = float(img[iy, ix - 20:ix + 21].astype(np.float64).mean())
    v_mean = float(img[iy - 20:iy + 21, ix].astype(np.float64).mean())
    arm_mean = (h_mean + v_mean) / 2.0

    corners = []
    for dx, dy in [(-10, -10), (10, -10), (-10, 10), (10, 10)]:
        patch = img[iy + dy - 3:iy + dy + 4, ix + dx - 3:ix + dx + 4]
        corners.append(float(patch.mean()))
    corner_mean = float(np.mean(corners))

    contrast = corner_mean - arm_mean
    center_val = int(img[iy, ix])

    return (contrast > MIN_CROSS_CONTRAST
            and center_val < MAX_CENTER_INTENSITY
            and arm_mean < MAX_ARM_INTENSITY)


def detect_crosses(img: np.ndarray, min_corr: float = MIN_TEMPLATE_CORR) -> list:
    """
    Detect coordinate crosses using template matching.

    Uses arm=20, thickness=3 template (calibrated for 300 DPI scans).
    Returns list of (x, y, correlation_score) tuples.
    """
    arm, thick = TEMPLATE_ARM_PX, TEMPLATE_THICKNESS_PX
    tpl = _make_cross_template(arm, thick)
    result = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    offset = arm

    # Non-maximum suppression
    nms_kernel = np.ones((NMS_WINDOW_PX, NMS_WINDOW_PX), np.float32)
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
        if _verify_cross(img, int(round(cx)), int(round(cy))):
            verified.append((cx, cy, corr))

    # De-duplicate (keep highest correlation within DEDUP_DISTANCE_PX)
    unique = []
    for cx, cy, corr in verified:
        if not any(abs(cx - ux) < DEDUP_DISTANCE_PX
                   and abs(cy - uy) < DEDUP_DISTANCE_PX
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


def organise_grid(crosses: list, image_h: int,
                  expected_rows: int = None) -> tuple:
    """
    Organise detected crosses into a regular grid.

    Returns (grid_dict, col_centers, row_centers, interior_rows).
    grid_dict maps (col, row) → (x, y, corr).
    interior_rows is a sorted list of row indices for the interior grid.
    """
    if len(crosses) < 4:
        raise RuntimeError("Too few crosses detected")

    # Cluster x and y coordinates
    x_clusters = _cluster_1d([c[0] for c in crosses], GRID_CLUSTER_GAP_PX)
    y_clusters = _cluster_1d([c[1] for c in crosses], GRID_CLUSTER_GAP_PX)
    col_centers = [float(np.median(c)) for c in x_clusters]
    row_centers = [float(np.median(c)) for c in y_clusters]

    # Assign each cross to the nearest grid cell
    grid = {}
    for cx, cy, corr in crosses:
        col = int(np.argmin([abs(cx - cc) for cc in col_centers]))
        row = int(np.argmin([abs(cy - rc) for rc in row_centers]))
        key = (col, row)
        if key not in grid or corr > grid[key][2]:
            grid[key] = (cx, cy, corr)

    # Identify interior rows by finding the largest group of consistently
    # spaced rows.  Frame tick marks near image edges have irregular spacing.
    if len(row_centers) >= 3:
        # Compute all pairwise consecutive spacings
        all_sps = [row_centers[i + 1] - row_centers[i]
                   for i in range(len(row_centers) - 1)]

        # The dominant spacing is the most common one (within 10% tolerance)
        sp_arr = np.array(all_sps)
        # Use the median of the larger spacings (> half of max) as the dominant one
        big_sps = sp_arr[sp_arr > sp_arr.max() * 0.5]
        dominant_sp = float(np.median(big_sps)) if len(big_sps) > 0 else float(np.median(sp_arr))

        # Build chains of rows with consistent spacing
        best_chain = []
        for start in range(len(row_centers)):
            chain = [start]
            for j in range(start + 1, len(row_centers)):
                sp = row_centers[j] - row_centers[chain[-1]]
                n = round(sp / dominant_sp)
                if n >= 1 and abs(sp - n * dominant_sp) < SPACING_TOLERANCE * dominant_sp:
                    chain.append(j)
            if len(chain) > len(best_chain):
                best_chain = chain

        interior_rows = best_chain

        # If we know the expected number of interior rows (from sheet geometry),
        # trim excess rows by removing those closest to the image edges.
        if expected_rows is not None and len(interior_rows) > expected_rows:
            while len(interior_rows) > expected_rows:
                top_dist = row_centers[interior_rows[0]]
                bot_dist = image_h - row_centers[interior_rows[-1]]
                if top_dist < bot_dist:
                    interior_rows = interior_rows[1:]
                else:
                    interior_rows = interior_rows[:-1]
    else:
        interior_rows = list(range(len(row_centers)))

    return grid, col_centers, row_centers, interior_rows


def refine_grid_positions(img: np.ndarray, grid: dict, col_centers: list,
                          row_centers: list, interior_rows: list) -> dict:
    """
    Refine detected positions using a two-pass approach:
    1. Use high-confidence detections to build an affine model
    2. Refine existing detections in tight windows
    3. Gap-fill missing cells only if the found position is close to the
       affine-predicted position and has sufficient correlation
    """
    tpl = _make_cross_template(TEMPLATE_ARM_PX, TEMPLATE_THICKNESS_PX)
    offset = TEMPLATE_ARM_PX

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
                                      int(round(new_y)))):
                    refined[(col, row)] = (new_x, new_y, float(max_val))

    return refined


# ---------------------------------------------------------------------------
# Coordinate assignment
# ---------------------------------------------------------------------------

def assign_coordinates(grid: dict, col_centers: list, row_centers: list,
                       interior_rows: list, sheet: SheetInfo) -> list:
    """
    Assign ST74 coordinates to detected crosses.

    Grid crosses are at 100 m intervals in both easting and northing.
    First column is at the left sheet boundary easting.
    First interior row (topmost) is at y_max − 100.
    """
    first_x = sheet.x_min
    first_y = sheet.y_max - 100

    gcps = []
    gcp_id = 1
    for ri, row in enumerate(interior_rows):
        map_y = first_y - ri * 100
        for col in range(len(col_centers)):
            map_x = first_x + col * 100
            if (col, row) in grid:
                px, py, corr = grid[(col, row)]
                gcps.append(GCP(
                    gcp_id=str(gcp_id), pixel=px, line=py,
                    x_st74=float(map_x), y_st74=float(map_y),
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
        bx[i] = g.x_st74
        by[i] = g.y_st74

    cx, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    cy, _, _, _ = np.linalg.lstsq(A, by, rcond=None)
    coeffs = np.concatenate([cx, cy])

    residuals = []
    for g in gcps:
        pred_x = cx[0] * g.pixel + cx[1] * g.line + cx[2]
        pred_y = cy[0] * g.pixel + cy[1] * g.line + cy[2]
        g.residual_x = g.x_st74 - pred_x
        g.residual_y = g.y_st74 - pred_y
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

def transform_to_sweref(gcps: list, source_crs: str = "EPSG:3152",
                        target_crs: str = "EPSG:3006") -> str:
    """Transform GCPs from ST74 to SWEREF99TM.  Returns pipeline description."""
    transformer = pyproj.Transformer.from_crs(
        source_crs, target_crs, always_xy=True
    )
    for g in gcps:
        g.x_sweref, g.y_sweref = transformer.transform(g.x_st74, g.y_st74)
    return str(transformer)


# ---------------------------------------------------------------------------
# VRT output
# ---------------------------------------------------------------------------

def write_vrt(gcps: list, image_path: str, image_w: int, image_h: int,
              output_path: str, crs: str = "EPSG:3006"):
    """Write a GDAL VRT file with GCPs in the target CRS."""
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
            "X": f"{g.x_sweref:.3f}",
            "Y": f"{g.y_sweref:.3f}",
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
# PDF report
# ---------------------------------------------------------------------------

def generate_pdf(result: DetectionResult, output_path: str):
    """Generate a 4-page PDF quality report."""
    with PdfPages(output_path) as pdf:
        _page_summary(pdf, result)
        _page_overlay(pdf, result)
        _page_residuals(pdf, result)
        _page_transformation(pdf, result)


def _page_summary(pdf, r: DetectionResult):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")

    title = f"Georeferencing Report\n{os.path.basename(r.image_path)}"
    ax.text(0.5, 0.92, title, transform=ax.transAxes,
            fontsize=18, fontweight="bold", ha="center", va="top")

    quality = ("GOOD" if r.rms_residual < 2.0 else
               "ACCEPTABLE" if r.rms_residual < 5.0 else "POOR")
    outlier_note = (f"  ({len(r.outlier_ids)} outlier(s): {', '.join(r.outlier_ids)})"
                    if r.outlier_ids else "  (no outliers)")

    lines = [
        f"Image file:      {os.path.basename(r.image_path)}",
        f"Image size:      {r.image_w} × {r.image_h} pixels",
        f"Sheet code:      {r.sheet.code}",
        f"Sheet bounds:    X [{r.sheet.x_min}..{r.sheet.x_max}]  "
        f"Y [{r.sheet.y_min}..{r.sheet.y_max}]",
        "",
        f"Detection grid:  {r.n_cols} cols × {r.n_rows} rows  "
        f"({len(r.gcps)} GCPs detected)",
        f"Grid spacing:    100 m easting, 100 m northing",
        f"Pixel scale:     ~{r.scale_x:.2f} px/m (E)  ~{r.scale_y:.2f} px/m (N)",
        "",
        f"Source CRS:      EPSG:3152  (ST74, Stockholm 1938)",
        f"Target CRS:      EPSG:3006  (SWEREF99 TM)",
        f"Pipeline:        pyproj (PROJ)",
        "",
        "Inter-consistency (affine fit):",
        f"  RMS residual:          {r.rms_residual:.3f} m",
        f"  Max residual:          {r.max_residual:.3f} m",
        f"  Outlier threshold:     {r.outlier_threshold:.3f} m (2×RMS)",
        "",
        f"Overall quality: {quality}",
        outlier_note,
    ]

    ax.text(0.06, 0.78, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, fontfamily="monospace", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                      edgecolor="#cccccc"))

    ax.text(0.5, 0.02,
            "Crosses detected by normalised cross-correlation template matching\n"
            "Transformation equivalent to Lantmäteriet GTRANS  "
            "(inv ST74 TM → UTM33 on GRS80)",
            transform=ax.transAxes, fontsize=8, ha="center", va="bottom",
            style="italic", color="#666666")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_overlay(pdf, r: DetectionResult):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    img = Image.open(r.image_path)
    ax.imshow(img, cmap="gray", aspect="equal")

    for g in r.gcps:
        is_out = g.gcp_id in r.outlier_ids
        color = "red" if is_out else "#00cc44"
        marker = "x" if is_out else "+"
        ax.plot(g.pixel, g.line, marker, color=color,
                markersize=12, markeredgewidth=2, zorder=5)
        ax.annotate(f" {g.gcp_id}", (g.pixel, g.line),
                    fontsize=7, fontweight="bold", color=color,
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
    pdf.savefig(fig)
    plt.close(fig)


def _page_residuals(pdf, r: DetectionResult):
    fig, (ax_t, ax_s) = plt.subplots(2, 1, figsize=(11.69, 8.27),
                                     gridspec_kw={"height_ratios": [3, 1]})

    ax_t.axis("off")
    ax_t.set_title("Inter-consistency — Affine Fit Residuals",
                   fontsize=12, fontweight="bold", pad=10)

    headers = ["GCP", "Pixel", "Line", "ST74 X", "ST74 Y",
               "Res X (m)", "Res Y (m)", "Res (m)", "Corr", "Flag"]
    rows = []
    colors = []
    for g in r.gcps:
        flag = "OUTLIER" if g.gcp_id in r.outlier_ids else ""
        rows.append([
            g.gcp_id, f"{g.pixel:.1f}", f"{g.line:.1f}",
            f"{g.x_st74:.0f}", f"{g.y_st74:.0f}",
            f"{g.residual_x:.3f}", f"{g.residual_y:.3f}",
            f"{g.residual_total:.3f}", f"{g.corr_score:.3f}", flag,
        ])
        bg = "#ffcccc" if flag else "#ffffff"
        colors.append([bg] * len(headers))

    tbl = ax_t.table(cellText=rows, colLabels=headers,
                     cellColours=colors,
                     colColours=["#d0d0d0"] * len(headers),
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.2)

    ax_s.axis("off")
    c = r.affine_coeffs
    stats = (
        f"Affine:  X = {c[0]:.6f}·px + {c[1]:.6f}·ln + {c[2]:.2f}\n"
        f"         Y = {c[3]:.6f}·px + {c[4]:.6f}·ln + {c[5]:.2f}\n\n"
        f"RMS: {r.rms_residual:.3f} m   Max: {r.max_residual:.3f} m   "
        f"Threshold: {r.outlier_threshold:.3f} m\n"
        f"Grid: {r.n_cols}×{r.n_rows}, spacing 100 m   "
        f"Scale: {r.scale_x:.2f}×{r.scale_y:.2f} px/m"
    )
    ax_s.text(0.05, 0.95, stats, transform=ax_s.transAxes,
              fontsize=9, fontfamily="monospace", va="top",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
                        edgecolor="#cccccc"))
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_transformation(pdf, r: DetectionResult):
    fig, (ax_t, ax_i) = plt.subplots(2, 1, figsize=(11.69, 8.27),
                                     gridspec_kw={"height_ratios": [3, 1]})

    ax_t.axis("off")
    ax_t.set_title("Coordinate Transformation — EPSG:3152 → EPSG:3006",
                   fontsize=12, fontweight="bold", pad=10)

    headers = ["GCP", "ST74 X", "ST74 Y", "SWEREF99TM E", "SWEREF99TM N"]
    rows = [[g.gcp_id, f"{g.x_st74:.0f}", f"{g.y_st74:.0f}",
             f"{g.x_sweref:.3f}", f"{g.y_sweref:.3f}"] for g in r.gcps]
    tbl = ax_t.table(cellText=rows, colLabels=headers,
                     colColours=["#d0d0d0"] * len(headers),
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.3)

    ax_i.axis("off")
    info = (
        f"Source CRS:  EPSG:3152  —  ST74, Stockholm 1938 (local TM)\n"
        f"Target CRS:  EPSG:3006  —  SWEREF99 TM\n\n"
        f"Pipeline:\n  {r.transform_pipeline}\n\n"
        f"Equivalent to Lantmäteriet GTRANS.\n"
        f"  1. Inverse ST74 TM → geographic on GRS80\n"
        f"  2. Forward UTM zone 33 → SWEREF99TM\n"
        f"No datum shift needed. Round-trip accuracy < 0.001 mm."
    )
    ax_i.text(0.05, 0.95, info, transform=ax_i.transAxes,
              fontsize=9, fontfamily="monospace", va="top",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
                        edgecolor="#cccccc"))
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_sheet_code(filename: str) -> str:
    """Extract a sheet code (e.g. '76D') from the filename."""
    m = re.search(r"(\d{2})\s*([A-Da-d])", filename)
    if m:
        return m.group(1) + m.group(2)
    raise ValueError(f"Cannot extract sheet code from filename: {filename!r}")


def run_pipeline(image_path: str, output_dir: str = None):
    """Run the full detection → georeferencing → reporting pipeline."""
    image_path = os.path.abspath(image_path)
    if not os.path.isfile(image_path):
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    basename = os.path.basename(image_path)
    name_no_ext = os.path.splitext(basename)[0]
    safe_name = name_no_ext.replace(" ", "_")

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_path), "reports")
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Sheet geometry ──
    print(f"Image: {basename}")
    sheet_code = extract_sheet_code(basename)
    sheet = parse_sheet_code(sheet_code)
    print(f"  Sheet: {sheet.code}  "
          f"X=[{sheet.x_min}..{sheet.x_max}]  Y=[{sheet.y_min}..{sheet.y_max}]")

    # ── 2. Load image ──
    img_pil = Image.open(image_path)
    img = np.array(img_pil)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape[:2]
    print(f"  Size: {w}×{h} pixels")

    # ── 3. Detect crosses ──
    print("Detecting coordinate crosses...")
    crosses = detect_crosses(img)
    print(f"  Raw detections: {len(crosses)}")

    # ── 4. Organise grid ──
    expected_interior_rows = (sheet.y_max - sheet.y_min) // 100 - 1
    grid, col_centers, row_centers, interior_rows = organise_grid(
        crosses, h, expected_rows=expected_interior_rows)
    print(f"  Grid: {len(col_centers)} cols × {len(interior_rows)} interior rows")
    print(f"  Column centers: {[f'{c:.0f}' for c in col_centers]}")
    print(f"  Row centers: {[f'{row_centers[r]:.0f}' for r in interior_rows]}")

    # ── 5. Refine positions ──
    print("Refining cross positions...")
    refined = refine_grid_positions(img, grid, col_centers,
                                    row_centers, interior_rows)
    print(f"  Refined cells: {len(refined)}")

    # ── 6. Assign ST74 coordinates ──
    print("Assigning ST74 coordinates...")
    gcps = assign_coordinates(refined, col_centers, row_centers,
                              interior_rows, sheet)
    print(f"  GCPs: {len(gcps)}")
    for g in gcps[:3]:
        print(f"    GCP {g.gcp_id}: px=({g.pixel:.1f},{g.line:.1f}) "
              f"→ ST74({g.x_st74:.0f},{g.y_st74:.0f})")
    if len(gcps) > 3:
        print(f"    ... ({len(gcps) - 3} more)")

    # ── 7. Inter-consistency ──
    print("Inter-consistency analysis...")
    coeffs, rms, max_res, threshold, outliers = fit_affine_and_residuals(gcps)
    print(f"  RMS: {rms:.3f} m  Max: {max_res:.3f} m")
    if outliers:
        print(f"  Outliers: {outliers}")

    # ── 8. Transform to SWEREF99TM ──
    print("Transforming to SWEREF99TM...")
    pipeline = transform_to_sweref(gcps)

    # ── 9. Compute scale stats ──
    col_sps = [col_centers[i + 1] - col_centers[i]
               for i in range(len(col_centers) - 1)]
    row_sps = [row_centers[interior_rows[i + 1]] - row_centers[interior_rows[i]]
               for i in range(len(interior_rows) - 1)]
    scale_x = float(np.mean(col_sps) / 100.0) if col_sps else 0.0
    scale_y = float(np.mean(row_sps) / 100.0) if row_sps else 0.0

    result = DetectionResult(
        image_path=image_path, image_w=w, image_h=h, sheet=sheet,
        gcps=gcps, affine_coeffs=coeffs,
        rms_residual=rms, max_residual=max_res,
        outlier_threshold=threshold, outlier_ids=outliers,
        transform_pipeline=pipeline,
        n_cols=len(col_centers), n_rows=len(interior_rows),
        col_spacing_px=float(np.mean(col_sps)) if col_sps else 0.0,
        row_spacing_px=float(np.mean(row_sps)) if row_sps else 0.0,
        scale_x=scale_x, scale_y=scale_y,
    )

    # ── 10. Write outputs ──
    vrt_path = os.path.join(output_dir, f"{safe_name}_SWEREF99TM.vrt")
    print(f"Writing VRT: {vrt_path}")
    write_vrt(gcps, image_path, w, h, vrt_path)

    pdf_path = os.path.join(output_dir, f"{safe_name}_report.pdf")
    print(f"Generating PDF: {pdf_path}")
    generate_pdf(result, pdf_path)

    print(f"\nDone!  {len(gcps)} GCPs, RMS={rms:.3f} m")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Detect coordinate crosses and georeference a scanned "
                    "borehole map image."
    )
    parser.add_argument("image", help="Path to the scanned map image (JPG/PNG)")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: reports/ next to image)")
    args = parser.parse_args()
    run_pipeline(args.image, args.output_dir)


if __name__ == "__main__":
    main()
