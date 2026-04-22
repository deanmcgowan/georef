#!/usr/bin/env python3
"""
write_qgis_layerdefs.py — Generate QGIS QML style files and QLR layer
definition files for georeferenced images.

Usage:
    python3 scripts/write_qgis_layerdefs.py <vrt_path> [--rgb] \
        [--output-dir <dir>]
"""

import argparse
import os
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path


def write_qml(image_path: str, vrt_path: str, output_path: str,
              is_rgb: bool = True) -> str:
    """Write a QGIS 3.x QML style file.

    Parameters
    ----------
    image_path : str
        Path to the source image (used for metadata only).
    vrt_path : str
        Path to the VRT file being styled.
    output_path : str
        Where to write the QML file.
    is_rgb : bool
        True for RGB (MultiBandColor), False for grayscale (SingleBandGray).

    Returns
    -------
    str
        The output path.
    """
    root = ET.Element("qgis", attrib={
        "version": "3.0",
        "styleCategories": "AllStyleCategories",
    })

    # Flags
    flags = ET.SubElement(root, "flags")
    ET.SubElement(flags, "Identifiable").text = "1"
    ET.SubElement(flags, "Removable").text = "1"
    ET.SubElement(flags, "Searchable").text = "1"

    # Pipe
    pipe = ET.SubElement(root, "pipe")

    if is_rgb:
        renderer = ET.SubElement(pipe, "rasterrenderer", attrib={
            "type": "multibandcolor",
            "opacity": "0.8",
            "alphaBand": "-1",
            "redBand": "1",
            "greenBand": "2",
            "blueBand": "3",
        })
        # Contrast enhancement for each band
        for band_elem_name in ("redContrastEnhancement",
                               "greenContrastEnhancement",
                               "blueContrastEnhancement"):
            ce = ET.SubElement(renderer, band_elem_name)
            _add_contrast_enhancement(ce)
    else:
        renderer = ET.SubElement(pipe, "rasterrenderer", attrib={
            "type": "singlebandgray",
            "opacity": "0.8",
            "alphaBand": "-1",
            "grayBand": "1",
            "gradient": "BlackToWhite",
        })
        ce = ET.SubElement(renderer, "contrastEnhancement")
        _add_contrast_enhancement(ce)

    # Transparency (near-white pixels treated as no-data)
    transparency = ET.SubElement(pipe, "rasterTransparency")
    single_values = ET.SubElement(transparency, "singleValuePixelList")
    for band_idx in ([1, 2, 3] if is_rgb else [1]):
        pixel = ET.SubElement(single_values, "pixelListEntry", attrib={
            "min": "245",
            "max": "255",
            "percentTransparent": "100",
        })
        if is_rgb:
            pixel.set("band", str(band_idx))

    # Brightness/contrast filter
    bright_contrast = ET.SubElement(pipe, "brightnesscontrast", attrib={
        "brightness": "0",
        "contrast": "0",
        "gamma": "1",
    })

    # Hue/saturation filter
    hue_saturation = ET.SubElement(pipe, "huesaturation", attrib={
        "colorizeRed": "255",
        "colorizeGreen": "128",
        "colorizeBlue": "128",
        "colorizeOn": "0",
        "colorizeStrength": "100",
        "grayscaleMode": "0",
        "saturation": "0",
    })

    # Resampler
    resampler = ET.SubElement(pipe, "rasterresampler", attrib={
        "maxOversampling": "2",
    })

    # Layer transparency
    ET.SubElement(root, "layerTransparency").text = "20"

    # Blending
    ET.SubElement(root, "blendMode").text = "0"
    ET.SubElement(root, "layerGeometryType").text = "4"

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n')
        tree.write(fh, encoding="unicode", xml_declaration=False)
        fh.write("\n")

    return output_path


def _add_contrast_enhancement(parent: ET.Element) -> None:
    """Add contrast enhancement child elements to parent."""
    ET.SubElement(parent, "minValue").text = "0"
    ET.SubElement(parent, "maxValue").text = "255"
    ET.SubElement(parent, "algorithm").text = "StretchToMinimumMaximum"


def write_qlr(image_path: str, vrt_path: str, output_path: str,
              layer_name: str = None) -> str:
    """Write a QGIS layer definition file (QLR).

    The QLR embeds the QML styling and points to the VRT datasource.

    Parameters
    ----------
    image_path : str
        Path to the source image (for naming).
    vrt_path : str
        Path to the VRT file (the QGIS datasource).
    output_path : str
        Where to write the QLR file.
    layer_name : str or None
        Layer name in QGIS. Defaults to the VRT filename stem.

    Returns
    -------
    str
        The output path.
    """
    if layer_name is None:
        layer_name = Path(vrt_path).stem

    layer_id = "georef_" + uuid.uuid4().hex[:12]

    # Make the datasource path relative to the QLR output directory
    qlr_dir = os.path.dirname(os.path.abspath(output_path))
    try:
        rel_vrt = os.path.relpath(os.path.abspath(vrt_path), qlr_dir)
    except ValueError:
        rel_vrt = os.path.abspath(vrt_path)

    # Detect if RGB from number of image bands
    is_rgb = _detect_rgb(image_path)

    root = ET.Element("qlr")

    # Layer tree group
    layer_tree_group = ET.SubElement(root, "layer-tree-group", attrib={
        "name": "",
        "checked": "Qt::Checked",
        "expanded": "1",
    })
    ET.SubElement(layer_tree_group, "layer-tree-layer", attrib={
        "id": layer_id,
        "name": layer_name,
        "source": rel_vrt,
        "providerKey": "gdal",
        "checked": "Qt::Checked",
        "expanded": "1",
    })
    ET.SubElement(root, "layer-tree-canvas")
    ET.SubElement(root, "custom-order", attrib={"enabled": "0"})

    # Layer definition
    layer = ET.SubElement(root, "layer", attrib={
        "id": layer_id,
        "autoRefreshEnabled": "0",
        "autoRefreshTime": "0",
        "type": "raster",
    })
    ET.SubElement(layer, "id").text = layer_id
    ET.SubElement(layer, "datasource").text = rel_vrt
    ET.SubElement(layer, "layername").text = layer_name
    ET.SubElement(layer, "provider", attrib={"encoding": ""}).text = "gdal"
    ET.SubElement(layer, "layerGeometryType").text = "4"

    # Spatial reference
    srs = ET.SubElement(layer, "srs")
    spatialref = ET.SubElement(srs, "spatialrefsys")
    ET.SubElement(spatialref, "wkt").text = ""
    ET.SubElement(spatialref, "proj4").text = "+proj=utm +zone=33 +ellps=GRS80 +units=m +no_defs"
    ET.SubElement(spatialref, "srsid").text = "3006"
    ET.SubElement(spatialref, "authid").text = "EPSG:3006"
    ET.SubElement(spatialref, "description").text = "SWEREF99 TM"
    ET.SubElement(spatialref, "projectionacronym").text = "utm"
    ET.SubElement(spatialref, "ellipsoidacronym").text = "GRS80"

    # Embedded pipe (QML-style)
    pipe = ET.SubElement(layer, "pipe")
    if is_rgb:
        renderer = ET.SubElement(pipe, "rasterrenderer", attrib={
            "type": "multibandcolor",
            "opacity": "0.8",
            "alphaBand": "-1",
            "redBand": "1",
            "greenBand": "2",
            "blueBand": "3",
        })
        for band_elem_name in ("redContrastEnhancement",
                               "greenContrastEnhancement",
                               "blueContrastEnhancement"):
            ce = ET.SubElement(renderer, band_elem_name)
            _add_contrast_enhancement(ce)
    else:
        renderer = ET.SubElement(pipe, "rasterrenderer", attrib={
            "type": "singlebandgray",
            "opacity": "0.8",
            "alphaBand": "-1",
            "grayBand": "1",
        })
        ce = ET.SubElement(renderer, "contrastEnhancement")
        _add_contrast_enhancement(ce)

    ET.SubElement(layer, "layerTransparency").text = "20"
    ET.SubElement(layer, "blendMode").text = "0"

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    os.makedirs(qlr_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write('<!DOCTYPE qgis-layer-definition>\n')
        tree.write(fh, encoding="unicode", xml_declaration=False)
        fh.write("\n")

    return output_path


def _detect_rgb(image_path: str) -> bool:
    """Return True if the image appears to be RGB (3+ bands)."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.mode in ("RGB", "RGBA")
    except Exception:
        return True  # assume RGB if detection fails


def main():
    parser = argparse.ArgumentParser(
        description="Generate QGIS QML and QLR files for a georeferenced VRT."
    )
    parser.add_argument("vrt", help="Path to the VRT file")
    parser.add_argument("--rgb", action="store_true",
                        help="Force RGB (MultiBandColor) renderer")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as VRT)")
    args = parser.parse_args()

    vrt_path = args.vrt
    vrt_stem = Path(vrt_path).stem

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(vrt_path))
    os.makedirs(out_dir, exist_ok=True)

    # Guess image path from VRT location
    vrt_dir = os.path.dirname(os.path.abspath(vrt_path))
    # Try common image extensions next to the VRT
    image_path = vrt_path  # fallback for RGB detection
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        candidate = os.path.join(vrt_dir, vrt_stem + ext)
        if os.path.exists(candidate):
            image_path = candidate
            break

    is_rgb = args.rgb or _detect_rgb(image_path)

    qml_path = os.path.join(out_dir, f"{vrt_stem}.qml")
    qlr_path = os.path.join(out_dir, f"{vrt_stem}.qlr")

    write_qml(image_path, vrt_path, qml_path, is_rgb=is_rgb)
    print(f"QML written: {qml_path}")

    write_qlr(image_path, vrt_path, qlr_path)
    print(f"QLR written: {qlr_path}")


if __name__ == "__main__":
    main()
