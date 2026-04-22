# Review

Images awaiting human validation after automated georeferencing.

Each image here has:
- A corresponding `_EPSG3006.vrt` VRT file
- A `_EPSG3006.qml` QGIS style file
- A `_EPSG3006.qlr` QGIS layer definition
- An HTML quality report in `reports/`
- A QA JSON in `reports/`

## Validation Steps

1. Open the HTML report in `reports/` and check the quality rating and residual table.
2. Open the `.qlr` file in QGIS and visually verify alignment with reference data.
3. If alignment is correct, move files to `done/`:

```bash
mv "review/<filename>" "done/"
mv "review/<safe_name>_EPSG3006.vrt" "done/"
mv "review/<safe_name>_EPSG3006.qml" "done/"
mv "review/<safe_name>_EPSG3006.qlr" "done/"
```

4. If alignment is incorrect, see the HTML report for diagnostics.

## Important

- Files here are generated automatically. Do NOT edit them manually.
- Files in `done/` have been manually validated and approved.
