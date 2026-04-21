#!/usr/bin/env python3
"""
georef_report.py — Georeferencing report generator.

Reads a VRT file containing GCPs, performs inter-consistency analysis,
optionally transforms coordinates to a target CRS, generates an HTML
report, and produces a new VRT with GCPs in the target CRS.

The VRT output is written to the same directory as the source image
(typically review/) so that the VRT and image sit side by side.

Usage:
    python scripts/georef_report.py <path_to_vrt> [--target-crs EPSG:3006]

Outputs:
    - reports/<name>_report.html
    - <image_dir>/<name>_<target_crs_tag>.vrt
"""

import argparse
import base64
import io
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    x_out: float = 0.0
    y_out: float = 0.0
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
    """Transform GCPs from source CRS to target CRS."""
    if report.source_crs == report.target_crs:
        for g in report.gcps:
            g.x_out, g.y_out = g.x, g.y
        report.transform_pipeline = "(identity — source and target CRS are the same)"
        return

    transformer = pyproj.Transformer.from_crs(
        report.source_crs, report.target_crs, always_xy=True
    )
    report.transform_pipeline = str(transformer)

    for g in report.gcps:
        g.x_out, g.y_out = transformer.transform(g.x, g.y)


# ---------------------------------------------------------------------------
# VRT generation in target CRS
# ---------------------------------------------------------------------------

def write_target_vrt(report: ReportData, output_path: str):
    """Write a new VRT file with GCPs in the target CRS."""
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
            "X": f"{g.x_out:.3f}",
            "Y": f"{g.y_out:.3f}",
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
# HTML report generation
# ---------------------------------------------------------------------------

def _generate_overlay_image(r: ReportData) -> str:
    """Render the GCP overlay plot and return it as a base64-encoded PNG."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    img = Image.open(r.image_path)
    ax.imshow(img, cmap="gray", aspect="equal")

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

    legend_elements = [
        Line2D([0], [0], marker="+", color="#00cc44", linestyle="None",
               markersize=10, markeredgewidth=2, label="GCP (OK)"),
        Line2D([0], [0], marker="x", color="red", linestyle="None",
               markersize=10, markeredgewidth=2, label="GCP (Outlier)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def generate_html_report(report: ReportData, output_path: str):
    """Generate an HTML quality report."""
    r = report
    overlay_b64 = _generate_overlay_image(r)

    quality = "GOOD" if r.rms_residual < 2.0 else "ACCEPTABLE" if r.rms_residual < 5.0 else "POOR"
    quality_colour = ("#27ae60" if quality == "GOOD" else
                      "#e67e22" if quality == "ACCEPTABLE" else "#c0392b")
    n_outliers = len(r.outlier_ids)

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
            f"<td>{g.pixel:.0f}</td><td>{g.line:.0f}</td>"
            f"<td>{g.x:.0f}</td><td>{g.y:.0f}</td>"
            f"<td>{g.residual_x:.3f}</td><td>{g.residual_y:.3f}</td>"
            f"<td>{g.residual_total:.3f}</td><td>{flag}</td>"
            f"</tr>\n"
        )

    # Build transformation table rows
    transform_rows = ""
    for g in r.gcps:
        transform_rows += (
            f"<tr>"
            f"<td>{html_escape(g.gcp_id)}</td>"
            f"<td>{g.x:.0f}</td><td>{g.y:.0f}</td>"
            f"<td>{g.x_out:.3f}</td><td>{g.y_out:.3f}</td>"
            f"</tr>\n"
        )

    c = r.affine_coeffs
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<title>Georeferencing Report — {html_escape(r.image_name)}</title>
<style>
  :root {{ --accent: #2c3e50; --bg: #f8f9fa; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         font-size: 14px; line-height: 1.6; color: #333; background: #fff;
         max-width: 1100px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 22px; color: var(--accent); border-bottom: 2px solid var(--accent);
       padding-bottom: 8px; margin-bottom: 16px; }}
  h2 {{ font-size: 17px; color: var(--accent); margin: 28px 0 10px 0;
       border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  .summary-box {{ background: var(--bg); border: 1px solid #ddd; border-radius: 6px;
                  padding: 18px 22px; margin: 12px 0; }}
  .quality-badge {{ display: inline-block; padding: 4px 14px; border-radius: 4px;
                    font-weight: bold; font-size: 15px; color: #fff; }}
  .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px 24px; }}
  .meta-grid dt {{ font-weight: 600; color: #555; }}
  .meta-grid dd {{ margin: 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }}
  th {{ background: #34495e; color: #fff; padding: 8px 10px; text-align: center; font-weight: 600; }}
  td {{ padding: 6px 10px; text-align: center; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  tr.outlier {{ background: #fce4e4; }}
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
<p><strong>{html_escape(r.image_name)}</strong></p>

<h2>Executive Summary</h2>
<div class="summary-box">
  <p>
    <span class="quality-badge" style="background:{quality_colour}">{quality}</span>
    &nbsp; Overall quality: <strong>{quality.lower()}</strong>.
  </p>
  <p>
    {len(r.gcps)} GCPs &ensp;|&ensp;
    RMS: {r.rms_residual:.3f}&nbsp;m &ensp;|&ensp;
    Max: {r.max_residual:.3f}&nbsp;m
    {f" &ensp;|&ensp; {n_outliers} outlier(s)" if n_outliers else ""}
  </p>
</div>

<dl class="meta-grid">
  <dt>Image file</dt><dd>{html_escape(os.path.basename(r.image_path))}</dd>
  <dt>Image size</dt><dd>{r.raster_x} &times; {r.raster_y} pixels</dd>
  <dt>Source CRS</dt><dd>{src_label}</dd>
  <dt>Target CRS</dt><dd>{tgt_label}</dd>
  <dt>Number of GCPs</dt><dd>{len(r.gcps)}</dd>
</dl>

<h2>Control Point Overlay</h2>
<img class="overlay-img" src="data:image/png;base64,{overlay_b64}"
     alt="GCP overlay on scanned map">

<h2>Residual Analysis</h2>
<table>
  <thead>
    <tr>
      <th>GCP</th><th>Pixel</th><th>Line</th>
      <th>Src&nbsp;X</th><th>Src&nbsp;Y</th>
      <th>Res&nbsp;X</th><th>Res&nbsp;Y</th><th>Res&nbsp;Total</th><th>Status</th>
    </tr>
  </thead>
  <tbody>
{residual_rows}  </tbody>
</table>

<div class="tech-section">
  <p>
    Affine model:<br>
    X = {c[0]:.6f} &times; pixel + {c[1]:.6f} &times; line + {c[2]:.2f}<br>
    Y = {c[3]:.6f} &times; pixel + {c[4]:.6f} &times; line + {c[5]:.2f}
  </p>
  <p>RMS: {r.rms_residual:.3f}&nbsp;m &ensp;|&ensp;
     Max: {r.max_residual:.3f}&nbsp;m &ensp;|&ensp;
     Outlier threshold (2&times;RMS): {r.outlier_threshold:.3f}&nbsp;m</p>
</div>

<h2>Coordinate Transformation</h2>
<table>
  <thead>
    <tr>
      <th>GCP</th><th>Source&nbsp;X</th><th>Source&nbsp;Y</th>
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

<div class="footer">
  Report generated {timestamp} by <code>georef_report.py</code>.
</div>

</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Written: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _crs_tag(crs: str) -> str:
    return crs.replace(":", "")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a georeferencing report and target-CRS VRT."
    )
    parser.add_argument("vrt", help="Path to source VRT file with GCPs")
    parser.add_argument("--target-crs", default="EPSG:3006",
                        help="Target CRS (default: EPSG:3006)")
    parser.add_argument("--report-dir", "-o", default=None,
                        help="Output directory for the HTML report (default: reports/)")
    parser.add_argument("--vrt-dir", default=None,
                        help="Output directory for the transformed VRT "
                             "(default: same directory as the source image)")
    args = parser.parse_args()

    vrt_path = args.vrt
    if not os.path.isfile(vrt_path):
        print(f"Error: VRT file not found: {vrt_path}", file=sys.stderr)
        sys.exit(1)

    # Parse
    print(f"Parsing VRT: {vrt_path}")
    report = parse_vrt(vrt_path)
    report.target_crs = args.target_crs
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

    # Report output directory
    if args.report_dir:
        report_dir = args.report_dir
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        report_dir = os.path.join(repo_root, "reports")
    os.makedirs(report_dir, exist_ok=True)

    # VRT output directory — default to same directory as the source image
    if args.vrt_dir:
        vrt_out_dir = args.vrt_dir
    else:
        vrt_out_dir = os.path.dirname(report.image_path)
    os.makedirs(vrt_out_dir, exist_ok=True)

    # Safe filename
    safe_name = report.image_name.replace(" ", "_")
    crs_tag = _crs_tag(report.target_crs)

    # Generate HTML report
    html_path = os.path.join(report_dir, f"{safe_name}_report.html")
    print("Generating HTML report...")
    generate_html_report(report, html_path)

    # Generate target-CRS VRT (next to the image)
    vrt_out_path = os.path.join(vrt_out_dir, f"{safe_name}_{crs_tag}.vrt")
    print(f"Generating {report.target_crs} VRT...")
    write_target_vrt(report, vrt_out_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
