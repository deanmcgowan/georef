#!/usr/bin/env python3
"""
verify_sweden.py — Query external sources to verify that inferred coordinates
fall within Sweden and are plausible.

Designed to be imported by the pipeline or run standalone. Fails gracefully
if network is unavailable.

Usage:
    python3 scripts/verify_sweden.py \
        --source-crs EPSG:3152 --center-x 99300 --center-y 77000
"""

import argparse
import base64
import json
import os
import sys
import time
from typing import Optional

# pyproj is required for coordinate transformation
try:
    from pyproj import Transformer
    _PYPROJ_AVAILABLE = True
except ImportError:
    _PYPROJ_AVAILABLE = False

# requests is optional (graceful degradation)
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

_SWEDEN_LAT_RANGE = (55.0, 70.0)
_SWEDEN_LON_RANGE = (10.0, 25.0)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_NOMINATIM_HEADERS = {
    "User-Agent": "georef-pipeline/1.0 (github.com/deanmcgowan/georef)"
}


def _to_wgs84(source_crs: str, x: float, y: float) -> Optional[tuple]:
    """Transform a coordinate to WGS84 (lat, lon). Returns None on failure."""
    if not _PYPROJ_AVAILABLE:
        return None
    try:
        transformer = Transformer.from_crs(source_crs, "EPSG:4326",
                                           always_xy=True)
        lon, lat = transformer.transform(x, y)
        return (lat, lon)
    except Exception:
        return None


def _within_sweden(lat: float, lon: float) -> bool:
    return (_SWEDEN_LAT_RANGE[0] <= lat <= _SWEDEN_LAT_RANGE[1] and
            _SWEDEN_LON_RANGE[0] <= lon <= _SWEDEN_LON_RANGE[1])


def _reverse_geocode_nominatim(lat: float, lon: float) -> dict:
    """Query Nominatim for reverse geocoding. Returns result dict."""
    if not _REQUESTS_AVAILABLE:
        return {}
    try:
        resp = requests.get(
            _NOMINATIM_URL,
            params={
                "lat": lat,
                "lon": lon,
                "format": "json",
                "zoom": 10,
                "addressdetails": 1,
            },
            headers=_NOMINATIM_HEADERS,
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def _fetch_lantmateriet_thumbnail(lat: float, lon: float) -> Optional[str]:
    """Attempt to fetch an orthophoto thumbnail from Lantmäteriet WMS.

    Requires LANTMATERIET_USERNAME and LANTMATERIET_PASSWORD env vars.
    Returns base64-encoded PNG string or None.
    """
    username = os.environ.get("LANTMATERIET_USERNAME")
    password = os.environ.get("LANTMATERIET_PASSWORD")
    if not username or not password:
        return None
    if not _REQUESTS_AVAILABLE:
        return None

    try:
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:3006", always_xy=True)
        x, y = t.transform(lon, lat)
        half = 500  # 500 m buffer
        bbox = f"{x-half},{y-half},{x+half},{y+half}"

        wms_url = "https://api.lantmateriet.se/open/topowebb-ccby/v1/wmts/token/"
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": "topowebb",
            "STYLES": "",
            "CRS": "EPSG:3006",
            "BBOX": bbox,
            "WIDTH": "256",
            "HEIGHT": "256",
            "FORMAT": "image/png",
        }
        resp = requests.get(
            wms_url,
            params=params,
            auth=(username, password),
            timeout=15,
        )
        if resp.status_code == 200 and resp.headers.get(
                "content-type", "").startswith("image/"):
            return base64.b64encode(resp.content).decode("ascii")
    except Exception:
        pass
    return None


def verify_location(source_crs: str, center_x: float,
                    center_y: float) -> dict:
    """Verify that a coordinate is within Sweden and reverse-geocode it.

    Parameters
    ----------
    source_crs : str
        EPSG code for the input coordinate (e.g. "EPSG:3152").
    center_x : float
        X (easting) coordinate in source_crs.
    center_y : float
        Y (northing) coordinate in source_crs.

    Returns
    -------
    dict
        Structured verification result.
    """
    result = {
        "center_wgs84": None,
        "within_sweden": None,
        "place_names_found": [],
        "verification_status": "SKIPPED",
        "verification_source": None,
        "verification_notes": "",
        "thumbnail_b64": None,
    }

    # Step 1: convert to WGS84
    wgs84 = _to_wgs84(source_crs, center_x, center_y)
    if wgs84 is None:
        result["verification_notes"] = (
            "Could not convert to WGS84 (pyproj unavailable or CRS error)"
        )
        return result

    lat, lon = wgs84
    result["center_wgs84"] = [round(lat, 6), round(lon, 6)]
    result["within_sweden"] = _within_sweden(lat, lon)

    if not result["within_sweden"]:
        result["verification_status"] = "FAIL"
        result["verification_notes"] = (
            f"Coordinate ({lat:.4f}, {lon:.4f}) is outside Sweden bounding box"
        )
        return result

    # Step 2: reverse geocode
    geo_data = _reverse_geocode_nominatim(lat, lon)
    if geo_data:
        result["verification_source"] = "nominatim"
        addr = geo_data.get("address", {})
        place_names = []
        for key in ("city", "town", "village", "municipality",
                    "county", "state", "country"):
            if val := addr.get(key):
                place_names.append(val)
        result["place_names_found"] = place_names

        country = addr.get("country_code", "").upper()
        if country and country != "SE":
            result["verification_status"] = "FAIL"
            result["verification_notes"] = (
                f"Nominatim returned country_code={country} (expected SE)"
            )
        else:
            result["verification_status"] = "PASS"
            display = geo_data.get("display_name", "")
            result["verification_notes"] = (
                f"Nominatim: {display[:120]}" if display else "OK"
            )
    else:
        # Network unavailable — partial (bounds check passed)
        result["verification_status"] = "PARTIAL"
        result["verification_notes"] = (
            "Within Sweden bounding box but Nominatim unreachable"
        )

    # Step 3: optional Lantmäteriet thumbnail
    result["thumbnail_b64"] = _fetch_lantmateriet_thumbnail(lat, lon)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Verify that inferred coordinates fall within Sweden."
    )
    parser.add_argument("--source-crs", required=True,
                        help="Source CRS, e.g. EPSG:3152")
    parser.add_argument("--center-x", required=True, type=float,
                        help="X (easting) coordinate in source CRS")
    parser.add_argument("--center-y", required=True, type=float,
                        help="Y (northing) coordinate in source CRS")
    parser.add_argument("--output-json", default=None,
                        help="Write result JSON to this path")
    args = parser.parse_args()

    result = verify_location(args.source_crs, args.center_x, args.center_y)
    output = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            fh.write(output)
        print(f"Result written to {args.output_json}", file=sys.stderr)
    else:
        print(output)

    if result["verification_status"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
