#!/usr/bin/env python3
"""
process_input_batch.py — Batch orchestration script for the georeferencing
pipeline.

Discovers images in input/, infers metadata, runs the detection pipeline,
generates QGIS artefacts, and performs external verification.

Usage:
    python3 scripts/process_input_batch.py \
        [--images img1.jpg img2.jpg ...] \
        [--all] \
        [--input-dir input/] \
        [--review-dir review/] \
        [--reports-dir reports/] \
        [--target-crs EPSG:3006] \
        [--fail-on-poor] \
        [--output-summary summary.json]
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from infer_map_metadata import infer_metadata
from verify_sweden import verify_location
from write_qgis_layerdefs import write_qml, write_qlr, _detect_rgb
import detect_and_georeference as dag

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_MIN_CRS_CONFIDENCE = 0.4
_MAX_RMS_FOR_REVIEW = 10.0  # metres


def _discover_images(input_dir: str) -> list:
    """Return sorted list of supported image paths in input_dir."""
    input_dir = Path(input_dir)
    images = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS:
            images.append(str(p))
    return images


def _safe_name(image_path: str) -> str:
    stem = Path(image_path).stem
    return stem.replace(" ", "_")


def _write_failed_qa(image_path: str, reason: str,
                     reports_dir: str, target_crs: str) -> dict:
    """Write a FAILED QA JSON and return the dict."""
    safe = _safe_name(image_path)
    qa = {
        "image_path": str(image_path),
        "safe_name": safe,
        "source_crs": None,
        "target_crs": target_crs,
        "grid_origin": None,
        "grid_spacing": None,
        "grid_size": None,
        "n_bands": None,
        "gcp_count": 0,
        "outlier_count": 0,
        "rms_residual": None,
        "max_residual": None,
        "quality_label": "FAILED",
        "internal_validation": "FAIL",
        "vrt_path": None,
        "report_path": None,
        "dpi": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script_version": "1.0.0",
        "failure_reason": reason,
    }
    os.makedirs(reports_dir, exist_ok=True)
    qa_path = os.path.join(reports_dir, f"{safe}_qa.json")
    with open(qa_path, "w", encoding="utf-8") as fh:
        json.dump(qa, fh, indent=2)
    return qa


def process_image(image_path: str, target_crs: str,
                  review_dir: str, reports_dir: str) -> dict:
    """Process a single image through the full pipeline.

    Returns a result dict summarising the outcome.
    """
    image_path = str(Path(image_path).resolve())
    safe = _safe_name(image_path)

    print(f"\n{'='*60}")
    print(f"Processing: {Path(image_path).name}")
    print(f"{'='*60}")

    # Step 1: Infer metadata
    meta = infer_metadata(image_path)

    if meta["sidecar_used"]:
        print(f"  Sidecar: {meta['sidecar_path']}")

    if meta["source_crs"] is None:
        reason = (
            f"CRS inference failed (confidence "
            f"{meta['source_crs_confidence']:.2f} < {_MIN_CRS_CONFIDENCE})"
        )
        print(f"  SKIP: {reason}", file=sys.stderr)
        qa = _write_failed_qa(image_path, reason, reports_dir, target_crs)
        return {"image": Path(image_path).name, "status": "FAILED",
                "reason": reason, "quality": "FAILED", "rms": None}

    source_crs = meta["source_crs"]
    effective_target = meta.get("target_crs") or target_crs
    print(f"  CRS: {source_crs} → {effective_target} "
          f"(confidence {meta['source_crs_confidence']:.2f})")

    # Step 2: Validate grid parameters are available
    grid_origin = meta.get("grid_origin")
    grid_spacing = meta.get("grid_spacing")
    grid_size = meta.get("grid_size")

    if not grid_origin or not grid_spacing:
        reason = (
            "Could not infer grid_origin or grid_spacing; "
            "provide a sidecar YAML with expected_grid_origin and "
            "expected_grid_spacing"
        )
        print(f"  SKIP: {reason}", file=sys.stderr)
        qa = _write_failed_qa(image_path, reason, reports_dir, effective_target)
        return {"image": Path(image_path).name, "status": "FAILED",
                "reason": reason, "quality": "FAILED", "rms": None}

    # Step 3: Copy image to review/ (preserving original in input/)
    os.makedirs(review_dir, exist_ok=True)
    review_image_path = os.path.join(review_dir, Path(image_path).name)
    if not os.path.exists(review_image_path):
        shutil.copy2(image_path, review_image_path)
        print(f"  Copied to review/")
    else:
        print(f"  Already in review/ (skipping copy)")

    # Step 4: Run georeferencing pipeline
    try:
        result, qa_json = dag.run_pipeline(
            image_path=image_path,
            source_crs=source_crs,
            target_crs=effective_target,
            grid_origin=tuple(grid_origin),
            grid_spacing=tuple(grid_spacing),
            grid_size=tuple(grid_size) if grid_size else None,
            review_dir=review_dir,
            reports_dir=reports_dir,
        )
    except Exception as exc:
        reason = f"Pipeline error: {exc}"
        print(f"  ERROR: {reason}", file=sys.stderr)
        qa = _write_failed_qa(image_path, reason, reports_dir, effective_target)
        return {"image": Path(image_path).name, "status": "FAILED",
                "reason": reason, "quality": "FAILED", "rms": None}

    rms = qa_json["rms_residual"]
    quality = qa_json["quality_label"]

    # Step 5: Fail-closed on very high RMS
    if rms is not None and rms > _MAX_RMS_FOR_REVIEW:
        reason = f"RMS {rms:.2f} m exceeds {_MAX_RMS_FOR_REVIEW} m threshold"
        print(f"  REJECT: {reason}", file=sys.stderr)
        qa_json["quality_label"] = "FAILED"
        qa_json["internal_validation"] = "FAIL"
        qa_json["failure_reason"] = reason
        qa_path = os.path.join(reports_dir, f"{safe}_qa.json")
        with open(qa_path, "w", encoding="utf-8") as fh:
            json.dump(qa_json, fh, indent=2)
        return {"image": Path(image_path).name, "status": "FAILED",
                "reason": reason, "quality": "FAILED",
                "rms": round(rms, 3)}

    # Step 6: Generate QGIS artefacts
    vrt_path = qa_json.get("vrt_path", "")
    if vrt_path and os.path.exists(vrt_path):
        vrt_stem = Path(vrt_path).stem
        vrt_dir = os.path.dirname(vrt_path)
        is_rgb = _detect_rgb(image_path)

        qml_path = os.path.join(vrt_dir, f"{vrt_stem}.qml")
        qlr_path = os.path.join(vrt_dir, f"{vrt_stem}.qlr")
        try:
            write_qml(image_path, vrt_path, qml_path, is_rgb=is_rgb)
            write_qlr(image_path, vrt_path, qlr_path)
            print(f"  QGIS artefacts: {Path(qml_path).name}, "
                  f"{Path(qlr_path).name}")
            qa_json["qml_path"] = qml_path
            qa_json["qlr_path"] = qlr_path
        except Exception as exc:
            print(f"  WARNING: QGIS artefact generation failed: {exc}",
                  file=sys.stderr)

    # Step 7: External verification
    center_x = grid_origin[0]
    center_y = grid_origin[1]
    try:
        verify_result = verify_location(source_crs, center_x, center_y)
        qa_json["verification"] = verify_result
        vstatus = verify_result.get("verification_status", "SKIPPED")
        print(f"  Verification: {vstatus} "
              f"({verify_result.get('verification_notes', '')[:80]})")
    except Exception as exc:
        print(f"  WARNING: Verification failed: {exc}", file=sys.stderr)
        qa_json["verification"] = {"verification_status": "SKIPPED",
                                   "verification_notes": str(exc)}

    # Step 8: Write QA JSON
    qa_path = os.path.join(reports_dir, f"{safe}_qa.json")
    with open(qa_path, "w", encoding="utf-8") as fh:
        json.dump(qa_json, fh, indent=2, default=str)
    print(f"  QA JSON: {qa_path}")
    print(f"  Result: {quality}  RMS={f'{rms:.3f}' if rms is not None else 'N/A'} m")

    return {
        "image": Path(image_path).name,
        "status": "OK",
        "quality": quality,
        "rms": round(rms, 3),
        "gcp_count": qa_json.get("gcp_count", 0),
        "vrt_path": vrt_path,
        "qa_path": qa_path,
    }


def _print_summary_table(results: list) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "="*70)
    print(f"{'Image':<40} {'Quality':<12} {'RMS (m)':<10} {'Status'}")
    print("-"*70)
    for r in results:
        name = r["image"][:39]
        quality = r.get("quality", "N/A")
        rms = f"{r['rms']:.3f}" if r.get("rms") is not None else "N/A"
        status = r.get("status", "N/A")
        print(f"{name:<40} {quality:<12} {rms:<10} {status}")
    print("="*70)

    total = len(results)
    good = sum(1 for r in results if r.get("quality") == "GOOD")
    acceptable = sum(1 for r in results if r.get("quality") == "ACCEPTABLE")
    poor = sum(1 for r in results if r.get("quality") == "POOR")
    failed = sum(1 for r in results if r.get("status") == "FAILED")
    print(f"\nTotal: {total}  GOOD: {good}  ACCEPTABLE: {acceptable}  "
          f"POOR: {poor}  FAILED: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch georeferencing pipeline for scanned Swedish maps."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--images", nargs="+", metavar="IMAGE",
                       help="Specific image files to process")
    group.add_argument("--all", action="store_true",
                       help="Process all images in --input-dir")
    parser.add_argument("--input-dir", default="input",
                        help="Directory to scan for images (default: input/)")
    parser.add_argument("--review-dir", default="review",
                        help="Output directory for VRT/QML/QLR (default: review/)")
    parser.add_argument("--reports-dir", default="reports",
                        help="Output directory for HTML/JSON reports "
                             "(default: reports/)")
    parser.add_argument("--target-crs", default="EPSG:3006",
                        help="Target CRS for output (default: EPSG:3006)")
    parser.add_argument("--fail-on-poor", action="store_true",
                        help="Exit non-zero if any image gets POOR rating")
    parser.add_argument("--output-summary", default=None,
                        help="Write overall summary JSON to this path")
    args = parser.parse_args()

    # Resolve directories relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    input_dir = str(repo_root / args.input_dir) if not os.path.isabs(
        args.input_dir) else args.input_dir
    review_dir = str(repo_root / args.review_dir) if not os.path.isabs(
        args.review_dir) else args.review_dir
    reports_dir = str(repo_root / args.reports_dir) if not os.path.isabs(
        args.reports_dir) else args.reports_dir

    # Discover images
    if args.images:
        images = []
        for img in args.images:
            p = Path(img)
            if not p.is_absolute():
                # Try relative to input_dir first, then cwd
                candidate = Path(input_dir) / p
                if candidate.exists():
                    images.append(str(candidate))
                elif p.exists():
                    images.append(str(p.resolve()))
                else:
                    print(f"WARNING: Image not found: {img}", file=sys.stderr)
            elif p.exists():
                images.append(str(p))
            else:
                print(f"WARNING: Image not found: {img}", file=sys.stderr)
    elif args.all:
        images = _discover_images(input_dir)
        if not images:
            print(f"No supported images found in {input_dir}", file=sys.stderr)
            sys.exit(0)
        print(f"Found {len(images)} image(s) in {input_dir}")
    else:
        parser.error("Specify --images or --all")

    if not images:
        print("No images to process.", file=sys.stderr)
        sys.exit(0)

    # Process each image
    results = []
    for img_path in images:
        result = process_image(
            image_path=img_path,
            target_crs=args.target_crs,
            review_dir=review_dir,
            reports_dir=reports_dir,
        )
        results.append(result)

    # Summary
    _print_summary_table(results)

    # Write summary JSON
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "results": results,
    }
    if args.output_summary:
        summary_path = args.output_summary
        os.makedirs(os.path.dirname(os.path.abspath(summary_path)) or ".",
                    exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"\nSummary written: {summary_path}")

    # Exit code
    has_poor = any(r.get("quality") == "POOR" for r in results)
    has_failed = any(r.get("status") == "FAILED" for r in results)

    if args.fail_on_poor and (has_poor or has_failed):
        sys.exit(1)


if __name__ == "__main__":
    main()
