#!/usr/bin/env python3
"""
infer_map_metadata.py — Best-effort inference of map metadata (CRS, grid
parameters) from a scanned image, using OCR and keyword analysis.

Returns a structured result dict with confidence scores. Fails gracefully
when OCR is unavailable or evidence is insufficient.

Usage:
    python3 scripts/infer_map_metadata.py <image_path> \
        [--sidecar <yaml_path>] [--output-json <path>]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import yaml

# pytesseract is optional
try:
    import pytesseract
    from PIL import Image as PILImage
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

_SCRIPT_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _SCRIPT_DIR.parent / "config" / "swedish_crs_candidates.yaml"


def _load_config() -> dict:
    """Load the Swedish CRS candidates YAML config."""
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_sidecar(sidecar_path: Optional[str]) -> Optional[dict]:
    """Load a sidecar YAML file, returning None if not found or invalid."""
    if not sidecar_path:
        return None
    p = Path(sidecar_path)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _derive_sidecar_path(image_path: str) -> str:
    """Return the default sidecar path for an image (same dir, .job.yaml)."""
    p = Path(image_path)
    return str(p.with_suffix(".job.yaml"))


def _run_ocr(image_path: str) -> str:
    """Run OCR on the image and return extracted text (lowercase).

    Returns empty string if pytesseract is not available or OCR fails.
    """
    if not _TESSERACT_AVAILABLE:
        return ""
    try:
        img = PILImage.open(image_path)
        # Use Swedish + English language packs if available; fall back silently.
        try:
            text = pytesseract.image_to_string(img, lang="swe+eng")
        except Exception:
            text = pytesseract.image_to_string(img)
        return text
    except Exception:
        return ""


def _extract_numbers(text: str) -> list:
    """Extract all numeric tokens from OCR text."""
    return [float(m) for m in re.findall(r"\b\d{4,7}(?:\.\d+)?\b", text)]


def _score_crs_candidates(ocr_text: str, config: dict) -> dict:
    """Score each candidate CRS based on OCR text and config keyword rules.

    Returns a dict mapping EPSG string → cumulative weight.
    """
    scores: dict = {}
    lower_text = ocr_text.lower()

    keyword_rules = config.get("keyword_scoring", [])
    evidence = []

    for rule in keyword_rules:
        kw = rule.get("keyword", "").lower()
        crs_hint = rule.get("crs_hint", "")
        weight = float(rule.get("weight", 0.0))
        if kw and kw in lower_text:
            scores[crs_hint] = scores.get(crs_hint, 0.0) + weight
            evidence.append({
                "type": "keyword",
                "text": kw,
                "crs_hint": crs_hint,
                "weight": weight,
            })

    # Score numeric tokens against known coordinate ranges
    numbers = _extract_numbers(ocr_text)
    crs_candidates = config.get("crs_candidates", [])
    for num in numbers:
        for cand in crs_candidates:
            epsg = f"EPSG:{cand['epsg']}"
            xr = cand.get("x_range", [0, 0])
            yr = cand.get("y_range", [0, 0])
            if xr[0] <= num <= xr[1] or yr[0] <= num <= yr[1]:
                w = 0.4
                scores[epsg] = scores.get(epsg, 0.0) + w
                evidence.append({
                    "type": "ocr_coords",
                    "text": str(num),
                    "parsed_value": num,
                    "crs_hint": epsg,
                    "weight": w,
                })

    return scores, evidence


def _infer_grid_from_numbers(numbers: list, config: dict,
                              source_crs: Optional[str]) -> dict:
    """Attempt to infer grid origin and spacing from extracted numeric values.

    Returns dict with grid_origin, grid_spacing, grid_size (or None values).
    """
    if not numbers or not source_crs:
        return {"grid_origin": None, "grid_spacing": None, "grid_size": None}

    # Find candidate CRS ranges
    crs_candidates = config.get("crs_candidates", [])
    epsg_num = int(source_crs.split(":")[-1]) if ":" in source_crs else None
    x_range = y_range = None
    for cand in crs_candidates:
        if cand.get("epsg") == epsg_num:
            x_range = cand.get("x_range")
            y_range = cand.get("y_range")
            break

    if not x_range or not y_range:
        return {"grid_origin": None, "grid_spacing": None, "grid_size": None}

    x_vals = sorted([n for n in numbers if x_range[0] <= n <= x_range[1]])
    y_vals = sorted([n for n in numbers if y_range[0] <= n <= y_range[1]])

    grid_origin = None
    grid_spacing = None
    grid_size = None

    if len(x_vals) >= 2:
        diffs_x = sorted(set(round(x_vals[i+1] - x_vals[i], 1)
                             for i in range(len(x_vals)-1)))
        dx = diffs_x[0] if diffs_x else None
    else:
        dx = None

    if len(y_vals) >= 2:
        diffs_y = sorted(set(round(y_vals[i+1] - y_vals[i], 1)
                             for i in range(len(y_vals)-1)))
        dy = diffs_y[0] if diffs_y else None
    else:
        dy = None

    if x_vals and y_vals:
        grid_origin = [x_vals[0], y_vals[-1]]  # top-left: min-x, max-y

    if dx and dy:
        # dy is typically negative (north-to-south on page)
        grid_spacing = [dx, -dy]
        nx = round((x_vals[-1] - x_vals[0]) / dx) + 1 if dx else None
        ny = round((y_vals[-1] - y_vals[0]) / dy) + 1 if dy else None
        if nx and ny:
            grid_size = [int(nx), int(ny)]

    return {
        "grid_origin": grid_origin,
        "grid_spacing": grid_spacing,
        "grid_size": grid_size,
    }


def infer_metadata(image_path: str,
                   sidecar_path: Optional[str] = None,
                   config: Optional[dict] = None) -> dict:
    """Infer map metadata from an image.

    Parameters
    ----------
    image_path : str
        Path to the scanned image.
    sidecar_path : str or None
        Explicit sidecar YAML path. If None, the default path is tried.
    config : dict or None
        Pre-loaded config dict. Loaded from disk if None.

    Returns
    -------
    dict
        Structured result with CRS, confidence, grid parameters, evidence.
    """
    if config is None:
        config = _load_config()

    # -- Sidecar --
    if sidecar_path is None:
        sidecar_path = _derive_sidecar_path(image_path)
    sidecar = _load_sidecar(sidecar_path)
    sidecar_used = sidecar is not None

    warnings = []

    # -- OCR --
    ocr_text = _run_ocr(image_path)
    if not ocr_text and not _TESSERACT_AVAILABLE:
        warnings.append("pytesseract not available; OCR skipped")

    # -- Score candidates from OCR --
    scores, evidence = _score_crs_candidates(ocr_text, config)

    # -- Determine best CRS --
    source_crs = None
    source_crs_confidence = 0.0
    target_crs = "EPSG:3006"

    if scores:
        best_epsg, best_score = max(scores.items(), key=lambda kv: kv[1])
        # Normalise confidence to 0-1 (cap at 1.0)
        total_possible = sum(
            r.get("weight", 0) for r in config.get("keyword_scoring", [])
        )
        if total_possible > 0:
            confidence = min(best_score / max(total_possible * 0.5, 1.0), 1.0)
        else:
            confidence = min(best_score, 1.0)

        if confidence >= 0.4:
            source_crs = best_epsg
            source_crs_confidence = round(confidence, 3)
        else:
            warnings.append(
                f"Best CRS confidence {confidence:.2f} < 0.4 threshold; "
                "returning source_crs=None"
            )

    # -- Grid inference from OCR numbers --
    numbers = _extract_numbers(ocr_text)
    grid_info = _infer_grid_from_numbers(numbers, config, source_crs)

    grid_origin = grid_info["grid_origin"]
    grid_spacing = grid_info["grid_spacing"]
    grid_size = grid_info["grid_size"]

    # -- Sidecar overrides (highest priority) --
    if sidecar_used:
        if sidecar.get("likely_source_crs"):
            source_crs = sidecar["likely_source_crs"]
            source_crs_confidence = 1.0  # sidecar is authoritative
        if sidecar.get("target_crs"):
            target_crs = sidecar["target_crs"]
        if sidecar.get("expected_grid_origin"):
            raw = sidecar["expected_grid_origin"]
            grid_origin = [float(raw[0]), float(raw[1])]
        if sidecar.get("expected_grid_spacing"):
            raw = sidecar["expected_grid_spacing"]
            grid_spacing = [float(raw[0]), float(raw[1])]
        if sidecar.get("expected_grid_size"):
            raw = sidecar["expected_grid_size"]
            grid_size = [int(raw[0]), int(raw[1])]

    return {
        "source_crs": source_crs,
        "source_crs_confidence": source_crs_confidence,
        "target_crs": target_crs,
        "grid_origin": grid_origin,
        "grid_spacing": grid_spacing,
        "grid_size": grid_size,
        "evidence": evidence,
        "warnings": warnings,
        "sidecar_used": sidecar_used,
        "sidecar_path": sidecar_path if sidecar_used else None,
        "ocr_text": ocr_text,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Infer map metadata (CRS, grid) from a scanned image."
    )
    parser.add_argument("image", help="Path to the scanned map image")
    parser.add_argument("--sidecar", default=None,
                        help="Path to sidecar YAML (default: <image>.job.yaml)")
    parser.add_argument("--output-json", default=None,
                        help="Write result JSON to this path")
    args = parser.parse_args()

    result = infer_metadata(args.image, sidecar_path=args.sidecar)

    output = json.dumps(result, indent=2, ensure_ascii=False,
                        default=str)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            fh.write(output)
        print(f"Result written to {args.output_json}", file=sys.stderr)
    else:
        print(output)

    if result["source_crs"] is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
