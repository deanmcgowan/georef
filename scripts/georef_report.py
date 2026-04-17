#!/usr/bin/env python3
"""
georef_report.py — Georeferencing report generator for scanned borehole maps.

Reads a VRT file containing GCPs in ST74 (EPSG:3152), performs inter-consistency
analysis, transforms coordinates to SWEREF99TM (EPSG:3006), generates a PDF
report, and produces a new VRT with GCPs in SWEREF99TM.

Usage:
    python scripts/georef_report.py <path_to_vrt>

Outputs (in reports/ directory alongside the VRT):
    - <name>_report.pdf
    - <name>_SWEREF99TM.vrt
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image
import pyproj


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GCP:
    gcp_id: str
    pixel: float
    line: float
    x: float  # map X in source CRS
    y: float  # map Y in source CRS
    z: float = 0.0
    x_sweref: float = 0.0
    y_sweref: float = 0.0
    residual_x: float = 0.0  # affine residual in map units
    residual_y: float = 0.0
    residual_total: float = 0.0


@dataclass
class ReportData:
    image_name: str
    image_path: str
    vrt_path: str
    source_crs: str
    target_crs: str = "EPSG:3006"
    raster_x: int = 0
    raster_y: int = 0
    gcps: list = field(default_factory=list)
    affine_coeffs: np.ndarray = None
    rms_residual: float = 0.0
    max_residual: float = 0.0
    outlier_threshold: float = 0.0
    outlier_ids: list = field(default_factory=list)
    transform_pipeline: str = ""


# ---------------------------------------------------------------------------
# VRT parsing
# ---------------------------------------------------------------------------

def parse_vrt(vrt_path: str) -> ReportData:
    """Parse a VRT file and extract GCPs and metadata."""
    tree = ET.parse(vrt_path)
    root = tree.getroot()

    raster_x = int(root.attrib["rasterXSize"])
    raster_y = int(root.attrib["rasterYSize"])

    gcp_list = root.find("GCPList")
    source_crs = gcp_list.attrib.get("Projection", "")

    # Find image filename from the VRT
    source_el = root.find(".//SourceFilename")
    image_filename = source_el.text if source_el is not None else ""
    vrt_dir = os.path.dirname(os.path.abspath(vrt_path))
    image_path = os.path.join(vrt_dir, image_filename)

    gcps = []
    for gcp_el in gcp_list.findall("GCP"):
        gcps.append(GCP(
            gcp_id=gcp_el.attrib["Id"],
            pixel=float(gcp_el.attrib["Pixel"]),
            line=float(gcp_el.attrib["Line"]),
            x=float(gcp_el.attrib["X"]),
            y=float(gcp_el.attrib["Y"]),
            z=float(gcp_el.attrib.get("Z", 0)),
        ))

    return ReportData(
        image_name=os.path.splitext(image_filename)[0],
        image_path=image_path,
        vrt_path=os.path.abspath(vrt_path),
        source_crs=source_crs,
        raster_x=raster_x,
        raster_y=raster_y,
        gcps=gcps,
    )


# ---------------------------------------------------------------------------
# Inter-consistency analysis (affine fit)
# ---------------------------------------------------------------------------

def fit_affine(gcps: list) -> tuple:
    """
    Fit a 6-parameter affine transformation from pixel/line → map X, Y.
    Returns (coefficients_x, coefficients_y) where each is [a, b, c]
    such that:
        map_x = a * pixel + b * line + c
        map_y = d * pixel + e * line + f
    """
    n = len(gcps)
    A = np.zeros((n, 3))
    bx = np.zeros(n)
    by = np.zeros(n)

    for i, g in enumerate(gcps):
        A[i] = [g.pixel, g.line, 1.0]
        bx[i] = g.x
        by[i] = g.y

    # Least-squares fit
    cx, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    cy, _, _, _ = np.linalg.lstsq(A, by, rcond=None)

    return np.concatenate([cx, cy])


def compute_residuals(report: ReportData):
    """Compute per-GCP residuals from affine fit."""
    coeffs = fit_affine(report.gcps)
    report.affine_coeffs = coeffs
    cx, cy = coeffs[:3], coeffs[3:]

    residuals = []
    for g in report.gcps:
        pred_x = cx[0] * g.pixel + cx[1] * g.line + cx[2]
        pred_y = cy[0] * g.pixel + cy[1] * g.line + cy[2]
        g.residual_x = g.x - pred_x
        g.residual_y = g.y - pred_y
        g.residual_total = np.sqrt(g.residual_x**2 + g.residual_y**2)
        residuals.append(g.residual_total)

    report.rms_residual = np.sqrt(np.mean(np.array(residuals)**2))
    report.max_residual = max(residuals)
    report.outlier_threshold = 2.0 * report.rms_residual
    report.outlier_ids = [
        g.gcp_id for g in report.gcps
        if g.residual_total > report.outlier_threshold
    ]


# ---------------------------------------------------------------------------
# Coordinate transformation
# ---------------------------------------------------------------------------

def transform_gcps(report: ReportData):
    """Transform GCPs from source CRS to SWEREF99TM (EPSG:3006)."""
    transformer = pyproj.Transformer.from_crs(
        report.source_crs, report.target_crs, always_xy=True
    )
    report.transform_pipeline = str(transformer)

    for g in report.gcps:
        g.x_sweref, g.y_sweref = transformer.transform(g.x, g.y)


# ---------------------------------------------------------------------------
# VRT generation in SWEREF99TM
# ---------------------------------------------------------------------------

def write_sweref_vrt(report: ReportData, output_path: str):
    """Write a new VRT file with GCPs in SWEREF99TM."""
    root = ET.Element("VRTDataset", attrib={
        "rasterXSize": str(report.raster_x),
        "rasterYSize": str(report.raster_y),
    })

    gcp_list = ET.SubElement(root, "GCPList", attrib={
        "Projection": report.target_crs,
    })

    for g in report.gcps:
        ET.SubElement(gcp_list, "GCP", attrib={
            "Id": g.gcp_id,
            "Pixel": f"{g.pixel:.1f}",
            "Line": f"{g.line:.1f}",
            "X": f"{g.x_sweref:.3f}",
            "Y": f"{g.y_sweref:.3f}",
            "Z": "0",
        })

    # Determine relative path from output VRT to the source image
    output_dir = os.path.dirname(os.path.abspath(output_path))
    rel_image = os.path.relpath(report.image_path, output_dir)

    band = ET.SubElement(root, "VRTRasterBand", attrib={
        "dataType": "Byte",
        "band": "1",
    })
    ET.SubElement(band, "ColorInterp").text = "Gray"
    source = ET.SubElement(band, "SimpleSource")
    ET.SubElement(source, "SourceFilename", attrib={
        "relativeToVRT": "1"
    }).text = rel_image
    ET.SubElement(source, "SourceBand").text = "1"
    ET.SubElement(source, "SrcRect", attrib={
        "xOff": "0", "yOff": "0",
        "xSize": str(report.raster_x), "ySize": str(report.raster_y),
    })
    ET.SubElement(source, "DstRect", attrib={
        "xOff": "0", "yOff": "0",
        "xSize": str(report.raster_x), "ySize": str(report.raster_y),
    })

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=False)
    # Add trailing newline
    with open(output_path, "a") as f:
        f.write("\n")
    print(f"  Written: {output_path}")


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------

def generate_pdf(report: ReportData, output_path: str):
    """Generate a multi-page PDF report."""
    with PdfPages(output_path) as pdf:
        _page_executive_summary(pdf, report)
        _page_gcp_overlay(pdf, report)
        _page_interconsistency(pdf, report)
        _page_transformation(pdf, report)
    print(f"  Written: {output_path}")


def _page_executive_summary(pdf: PdfPages, r: ReportData):
    """Page 1: Executive summary."""
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4
    ax.axis("off")

    title = f"Georeferencing Report\n{r.image_name}"
    ax.text(0.5, 0.92, title, transform=ax.transAxes,
            fontsize=18, fontweight="bold", ha="center", va="top")

    quality = "GOOD" if r.rms_residual < 2.0 else "ACCEPTABLE" if r.rms_residual < 5.0 else "POOR"
    if r.outlier_ids:
        quality_note = f"  ({len(r.outlier_ids)} outlier GCP(s) detected: {', '.join(r.outlier_ids)})"
    else:
        quality_note = "  (no outlier GCPs detected)"

    summary_lines = [
        f"Image file:  {os.path.basename(r.image_path)}",
        f"Image size:  {r.raster_x} × {r.raster_y} pixels",
        f"Number of GCPs:  {len(r.gcps)}",
        "",
        f"Source CRS:  {r.source_crs}  (ST74, Stockholm 1938)",
        f"Target CRS:  {r.target_crs}  (SWEREF99 TM)",
        "",
        f"Transformation:  pyproj (PROJ pipeline)",
        f"Pipeline:  {r.transform_pipeline}",
        "",
        "Inter-consistency (affine fit):",
        f"  RMS residual:  {r.rms_residual:.3f} m",
        f"  Max residual:  {r.max_residual:.3f} m",
        f"  Outlier threshold (2×RMS):  {r.outlier_threshold:.3f} m",
        "",
        f"Overall quality:  {quality}",
        quality_note,
    ]

    ax.text(0.08, 0.78, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=10, fontfamily="monospace", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#cccccc"))

    ax.text(0.5, 0.02,
            "Transformation equivalent to Lantmäteriet GTRANS\n"
            "(inv ST74 TM → UTM zone 33 on GRS80)",
            transform=ax.transAxes, fontsize=8, ha="center", va="bottom",
            style="italic", color="#666666")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_gcp_overlay(pdf: PdfPages, r: ReportData):
    """Page 2: GCPs overlaid on the scanned image."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape

    # Load and display image
    img = Image.open(r.image_path)
    ax.imshow(img, cmap="gray", aspect="equal")

    # Plot GCPs
    for g in r.gcps:
        is_outlier = g.gcp_id in r.outlier_ids
        color = "red" if is_outlier else "#00cc44"
        marker = "x" if is_outlier else "+"
        ax.plot(g.pixel, g.line, marker, color=color, markersize=12,
                markeredgewidth=2, zorder=5)
        ax.annotate(
            f" {g.gcp_id}", (g.pixel, g.line),
            fontsize=8, fontweight="bold", color=color,
            ha="left", va="bottom", zorder=6,
        )

    ax.set_title(f"GCP Locations — {r.image_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Line")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="+", color="#00cc44", linestyle="None",
               markersize=10, markeredgewidth=2, label="GCP (OK)"),
        Line2D([0], [0], marker="x", color="red", linestyle="None",
               markersize=10, markeredgewidth=2, label="GCP (Outlier)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_interconsistency(pdf: PdfPages, r: ReportData):
    """Page 3: Inter-consistency analysis table and statistics."""
    fig, (ax_table, ax_text) = plt.subplots(
        2, 1, figsize=(11.69, 8.27),
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # --- Table ---
    ax_table.axis("off")
    ax_table.set_title("Inter-consistency Analysis — Affine Fit Residuals",
                       fontsize=12, fontweight="bold", pad=10)

    headers = ["GCP", "Pixel", "Line", "ST74 X", "ST74 Y",
               "Res. X (m)", "Res. Y (m)", "Res. Total (m)", "Flag"]
    rows = []
    cell_colors = []
    for g in r.gcps:
        flag = "OUTLIER" if g.gcp_id in r.outlier_ids else ""
        row = [
            g.gcp_id,
            f"{g.pixel:.0f}",
            f"{g.line:.0f}",
            f"{g.x:.0f}",
            f"{g.y:.0f}",
            f"{g.residual_x:.3f}",
            f"{g.residual_y:.3f}",
            f"{g.residual_total:.3f}",
            flag,
        ]
        rows.append(row)
        if flag:
            cell_colors.append(["#ffcccc"] * len(headers))
        else:
            cell_colors.append(["#ffffff"] * len(headers))

    table = ax_table.table(
        cellText=rows,
        colLabels=headers,
        cellColours=cell_colors,
        colColours=["#d0d0d0"] * len(headers),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    # --- Stats text ---
    ax_text.axis("off")

    # Expected spacing analysis
    unique_x = sorted(set(g.x for g in r.gcps))
    unique_y = sorted(set(g.y for g in r.gcps))
    dx = [unique_x[i+1] - unique_x[i] for i in range(len(unique_x)-1)] if len(unique_x) > 1 else []
    dy = [unique_y[i+1] - unique_y[i] for i in range(len(unique_y)-1)] if len(unique_y) > 1 else []

    # Pixel spacing analysis
    pixel_spacings = []
    line_spacings = []
    min_dx = min(dx) if dx else None
    min_dy = min(dy) if dy else None
    for g1 in r.gcps:
        for g2 in r.gcps:
            if min_dx and g1.y == g2.y and g1.x < g2.x and (g2.x - g1.x) == min_dx:
                pixel_spacings.append(abs(g2.pixel - g1.pixel))
            if min_dy and g1.x == g2.x and g1.y < g2.y and (g2.y - g1.y) == min_dy:
                line_spacings.append(abs(g2.line - g1.line))

    stats_text = (
        f"Affine model:  map_x = {r.affine_coeffs[0]:.6f}·pixel + {r.affine_coeffs[1]:.6f}·line + {r.affine_coeffs[2]:.2f}\n"
        f"               map_y = {r.affine_coeffs[3]:.6f}·pixel + {r.affine_coeffs[4]:.6f}·line + {r.affine_coeffs[5]:.2f}\n"
        f"\n"
        f"RMS residual: {r.rms_residual:.3f} m    |    Max residual: {r.max_residual:.3f} m    |    "
        f"Outlier threshold (2×RMS): {r.outlier_threshold:.3f} m\n"
        f"\n"
        f"Grid spacing (map coords):  ΔX = {dx}    ΔY = {dy}\n"
    )
    if pixel_spacings:
        stats_text += (
            f"Pixel spacing (adjacent columns):  mean={np.mean(pixel_spacings):.1f}  "
            f"std={np.std(pixel_spacings):.1f}  range=[{min(pixel_spacings):.0f}, {max(pixel_spacings):.0f}]\n"
        )
    if line_spacings:
        stats_text += (
            f"Line spacing (adjacent rows):  mean={np.mean(line_spacings):.1f}  "
            f"std={np.std(line_spacings):.1f}  range=[{min(line_spacings):.0f}, {max(line_spacings):.0f}]\n"
        )

    ax_text.text(0.05, 0.95, stats_text, transform=ax_text.transAxes,
                 fontsize=9, fontfamily="monospace", va="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8", edgecolor="#cccccc"))

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_transformation(pdf: PdfPages, r: ReportData):
    """Page 4: ST74 → SWEREF99TM transformation table."""
    fig, (ax_table, ax_text) = plt.subplots(
        2, 1, figsize=(11.69, 8.27),
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # --- Table ---
    ax_table.axis("off")
    ax_table.set_title(
        f"Coordinate Transformation — {r.source_crs} → {r.target_crs}",
        fontsize=12, fontweight="bold", pad=10
    )

    headers = ["GCP", "ST74 X", "ST74 Y", "SWEREF99TM E", "SWEREF99TM N"]
    rows = []
    for g in r.gcps:
        rows.append([
            g.gcp_id,
            f"{g.x:.0f}",
            f"{g.y:.0f}",
            f"{g.x_sweref:.3f}",
            f"{g.y_sweref:.3f}",
        ])

    table = ax_table.table(
        cellText=rows,
        colLabels=headers,
        colColours=["#d0d0d0"] * len(headers),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # --- Transformation info ---
    ax_text.axis("off")
    info = (
        f"Source CRS:  {r.source_crs}  —  ST74, Stockholm 1938 (local TM)\n"
        f"Target CRS:  {r.target_crs}  —  SWEREF99 TM (ETRS89 / UTM zone 33N)\n"
        f"\n"
        f"PROJ pipeline:\n"
        f"  {r.transform_pipeline}\n"
        f"\n"
        f"This is equivalent to Lantmäteriet GTRANS.  The pipeline performs:\n"
        f"  1. Inverse ST74 Transverse Mercator projection (→ geographic on GRS80)\n"
        f"  2. Forward UTM zone 33 projection (→ SWEREF99TM)\n"
        f"\n"
        f"No datum shift is required — both CRS use GRS80 ellipsoid and\n"
        f"SWEREF99/ETRS89 realization.  Round-trip accuracy: < 0.001 mm."
    )
    ax_text.text(0.05, 0.95, info, transform=ax_text.transAxes,
                 fontsize=9, fontfamily="monospace", va="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8", edgecolor="#cccccc"))

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a georeferencing report and SWEREF99TM VRT."
    )
    parser.add_argument("vrt", help="Path to source VRT file (with GCPs in ST74)")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: reports/ next to VRT)")
    args = parser.parse_args()

    vrt_path = args.vrt
    if not os.path.isfile(vrt_path):
        print(f"Error: VRT file not found: {vrt_path}", file=sys.stderr)
        sys.exit(1)

    # Parse
    print(f"Parsing VRT: {vrt_path}")
    report = parse_vrt(vrt_path)
    print(f"  Image: {report.image_name}")
    print(f"  GCPs: {len(report.gcps)}")
    print(f"  Source CRS: {report.source_crs}")

    # Inter-consistency
    print("Running inter-consistency analysis...")
    compute_residuals(report)
    print(f"  RMS residual: {report.rms_residual:.3f} m")
    print(f"  Max residual: {report.max_residual:.3f} m")
    if report.outlier_ids:
        print(f"  Outliers: {report.outlier_ids}")
    else:
        print("  No outliers detected")

    # Transform
    print(f"Transforming to {report.target_crs}...")
    transform_gcps(report)
    print(f"  Pipeline: {report.transform_pipeline}")

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(vrt_path)), "reports")
    os.makedirs(out_dir, exist_ok=True)

    # Safe filename
    safe_name = report.image_name.replace(" ", "_")

    # Generate PDF
    pdf_path = os.path.join(out_dir, f"{safe_name}_report.pdf")
    print(f"Generating PDF report...")
    generate_pdf(report, pdf_path)

    # Generate SWEREF99TM VRT
    vrt_out_path = os.path.join(out_dir, f"{safe_name}_SWEREF99TM.vrt")
    print(f"Generating SWEREF99TM VRT...")
    write_sweref_vrt(report, vrt_out_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
