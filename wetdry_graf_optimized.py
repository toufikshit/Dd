#!/usr/bin/env python3
"""Fast Graf/RSTD wet-dry detection for QC-controlled CML NetCDF files.

The script reads the QC output created by ``cml_qc_filter.py`` and appends a
link-level wet/dry flag without changing the original dataset structure.  It
also writes a contingency table when a reference wet/rain variable is supplied.

Example demo-style run::

    python wetdry_graf_optimized.py \
      --qc-cml-nc CML_QC.nc \
      --out-nc CML_QC_wetdry.nc \
      --contingency-csv contingency_global.csv

If the QC file already contains a reference variable, e.g. ``radar_rain_rate``
with dimensions ``(cml_id, time)``, add ``--reference-var radar_rain_rate``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import csv
import xarray as xr


def _signal_names(ds: xr.Dataset) -> tuple[str, str]:
    if {"tsl", "rsl"}.issubset(ds.data_vars):
        return "tsl", "rsl"
    if {"tx", "rx"}.issubset(ds.data_vars):
        return "tx", "rx"
    raise KeyError("QC CML dataset must contain tsl/rsl, tx/rx, or trsl variables.")


def _as_sublink_cml_time(da: xr.DataArray) -> xr.DataArray:
    missing = {"time", "sublink_id", "cml_id"} - set(da.dims)
    if missing:
        raise ValueError(f"{da.name!r} is missing required dimensions: {sorted(missing)}")
    return da.transpose("sublink_id", "cml_id", "time")


def _rolling_std_centered_2d(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Centered rolling nan-std over axis 0 for a ``(time, series)`` matrix.

    Uses cumulative sums instead of pandas/xarray rolling objects, which is much
    faster on small/demo datasets and avoids repeatedly constructing DataArrays.
    """
    x = np.asarray(values, dtype="float64")
    finite = np.isfinite(x)
    x0 = np.where(finite, x, 0.0)
    x2 = x0 * x0
    cs = np.vstack([np.zeros((1, x.shape[1]), dtype="float64"), np.cumsum(x0, axis=0)])
    cs2 = np.vstack([np.zeros((1, x.shape[1]), dtype="float64"), np.cumsum(x2, axis=0)])
    cn = np.vstack([np.zeros((1, x.shape[1]), dtype="int64"), np.cumsum(finite, axis=0)])

    n_time = x.shape[0]
    left = window // 2
    right = window - left
    starts = np.maximum(np.arange(n_time) - left, 0)
    stops = np.minimum(np.arange(n_time) + right, n_time)

    count = cn[stops] - cn[starts]
    total = cs[stops] - cs[starts]
    total2 = cs2[stops] - cs2[starts]
    out = np.full(x.shape, np.nan, dtype="float32")
    ok = count >= min_periods
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        var = (total2 / count) - mean * mean
    var = np.maximum(var, 0.0)
    out[ok] = np.sqrt(var[ok]).astype("float32")
    return out


def graf_wetdry_fast(
    trsl: xr.DataArray,
    raw_trsl: xr.DataArray,
    sane_sublink: xr.DataArray | None,
    window: int,
    quantile: float,
    factor: float,
    min_periods_threshold: int,
    min_periods_detection: int,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return link wet flag, sublink wet flag, and per-sublink thresholds."""
    trsl3 = _as_sublink_cml_time(trsl)
    raw3 = _as_sublink_cml_time(raw_trsl)
    n_sub, n_cml, n_time = trsl3.shape
    interp2 = trsl3.values.reshape(n_sub * n_cml, n_time).T
    raw2 = raw3.values.reshape(n_sub * n_cml, n_time).T

    threshold_std = _rolling_std_centered_2d(interp2, window, min_periods_threshold)
    thresholds = (factor * np.nanquantile(threshold_std, quantile, axis=0)).astype("float32")
    detect_std = _rolling_std_centered_2d(raw2, window, min_periods_detection)
    wet2 = detect_std > thresholds[None, :]
    wet = wet2.T.reshape(n_sub, n_cml, n_time)

    if sane_sublink is not None:
        sane = _as_sublink_cml_time(sane_sublink).values.astype(bool)
        wet &= sane[:, :, None]

    wet_sub = xr.DataArray(
        wet,
        coords=trsl3.coords,
        dims=trsl3.dims,
        name="cml_wet_flag_sublink",
        attrs={"description": "Graf/RSTD wet flag per sublink from centered rolling standard deviation."},
    ).transpose(*raw_trsl.dims)
    wet_link_values = wet.any(axis=0)
    wet_link = xr.DataArray(
        wet_link_values,
        coords={"cml_id": trsl3.cml_id, "time": trsl3.time},
        dims=("cml_id", "time"),
        name="cml_wet_flag",
        attrs={"description": "Link-level Graf/RSTD wet flag; true when any QC-sane sublink is wet."},
    )
    thr = xr.DataArray(
        thresholds.reshape(n_sub, n_cml),
        coords={"sublink_id": trsl3.sublink_id, "cml_id": trsl3.cml_id},
        dims=("sublink_id", "cml_id"),
        name="wet_threshold_dB",
        attrs={"description": "Per-sublink Graf/RSTD rolling-sigma threshold in dB."},
    )
    return wet_link, wet_sub, thr


def contingency_counts(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray | None = None) -> dict[str, int]:
    pred = np.asarray(pred, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    mask = np.isfinite(ref) if ref.dtype.kind == "f" else np.ones(ref.shape, dtype=bool)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    p = pred[mask]
    r = ref[mask]
    return {
        "hit": int(np.sum(p & r)),
        "false_alarm": int(np.sum(p & ~r)),
        "miss": int(np.sum(~p & r)),
        "correct_negative": int(np.sum(~p & ~r)),
    }


def contingency_scores(c: dict[str, int]) -> dict[str, float | int]:
    h, f, m, cn = c["hit"], c["false_alarm"], c["miss"], c["correct_negative"]
    n = h + f + m + cn
    den_mcc = np.sqrt((h + f) * (h + m) * (cn + f) * (cn + m))
    return {
        "n_total": n,
        "POD": h / (h + m) if h + m else np.nan,
        "FAR": f / (h + f) if h + f else np.nan,
        "CSI": h / (h + f + m) if h + f + m else np.nan,
        "ACC": (h + cn) / n if n else np.nan,
        "MCC": ((h * cn) - (f * m)) / den_mcc if den_mcc else np.nan,
    }


def _find_reference(ds: xr.Dataset, name: str | None) -> xr.DataArray | None:
    if name:
        if name not in ds:
            raise KeyError(f"Reference variable {name!r} not found.")
        return ds[name]
    for candidate in ("reference_wet", "radar_wet", "gauge_wet", "radar_rain_rate", "gauge_rain_rate"):
        if candidate in ds:
            return ds[candidate]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast Graf/RSTD wet-dry flags for QC-controlled CML datasets.")
    parser.add_argument("--qc-cml-nc", required=True, help="QC-controlled CML NetCDF from cml_qc_filter.py.")
    parser.add_argument("--out-nc", required=True, help="Output NetCDF with the same structure plus wet flag variables.")
    parser.add_argument("--contingency-csv", default=None, help="CSV path for global contingency table.")
    parser.add_argument("--reference-var", default=None, help="Reference wet/rain variable in the QC dataset.")
    parser.add_argument("--reference-wet-thr", type=float, default=0.1, help="Wet threshold for numeric rain-rate references.")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--quantile", type=float, default=0.8)
    parser.add_argument("--factor", type=float, default=1.12)
    parser.add_argument("--min-periods-threshold", type=int, default=60)
    parser.add_argument("--min-periods-detection", type=int, default=45)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_nc = Path(args.out_nc)
    if out_nc.exists() and not args.overwrite:
        raise FileExistsError(f"{out_nc} exists. Use --overwrite to replace it.")

    t0 = time.perf_counter()
    with xr.open_dataset(args.qc_cml_nc) as src:
        ds = src.load()

    raw_trsl = ds["trsl_raw"] if "trsl_raw" in ds else None
    if raw_trsl is None:
        tx_name, rx_name = _signal_names(ds)
        raw_trsl = ds[tx_name] - ds[rx_name]
        raw_trsl.name = "trsl_raw"
    trsl = ds["trsl"] if "trsl" in ds else raw_trsl.interpolate_na(dim="time", method="linear", max_gap="5min")
    sane = ds["isSane_sublink"] if "isSane_sublink" in ds else None

    wet_link, wet_sub, threshold = graf_wetdry_fast(
        trsl, raw_trsl, sane, args.window, args.quantile, args.factor,
        args.min_periods_threshold, args.min_periods_detection,
    )
    out = ds.copy()
    out["cml_wet_flag"] = wet_link.astype("int8")
    out["cml_wet_flag_sublink"] = wet_sub.astype("int8")
    out["wet_threshold_dB"] = threshold
    out.attrs["wetdry_method"] = "Graf/RSTD rolling-sigma classifier on QC-controlled CML data"

    out_nc.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True} for name, var in out.variables.items() if var.dims}
    out.to_netcdf(out_nc, encoding=encoding)

    ref = _find_reference(out, args.reference_var)
    if args.contingency_csv:
        if ref is None:
            rows = [{"reference": "none", "note": "No reference variable supplied or detected."}]
        else:
            ref2 = ref.transpose("cml_id", "time") if set(ref.dims) >= {"cml_id", "time"} else ref
            ref_valid = np.isfinite(ref2.values)
            ref_wet = ref2.astype(bool) if ref2.dtype == bool else (ref2 >= args.reference_wet_thr)
            counts = contingency_counts(out["cml_wet_flag"].values.astype(bool), ref_wet.values, ref_valid)
            rows = [{"reference": ref.name, **counts, **contingency_scores(counts)}]
        csv_path = Path(args.contingency_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(rows[0])

    print(f"Saved wet/dry dataset: {out_nc}")
    print(f"Done in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
