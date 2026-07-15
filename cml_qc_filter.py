#!/usr/bin/env python3
"""Create a QC-filtered CML NetCDF dataset.

This script follows the processing steps in the provided Blettner/Graf-style
notebook code and writes one output file with per-sample and per-link QC flags:

* plateau/period RSL sanity mask
* blackout-gap fill flag
* TRSL interpolation for ``graf_2020``/``full`` processing lines
* detection-limit, long-STD, short-STD, and availability flags
* final per-sublink and per-link ``isSane`` flags

Example:
    python cml_qc_filter.py --cml-nc CML_S2020E2021_1MIN_CZ.nc --proc-line full
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


def _infer_output_path(cml_nc: str | os.PathLike[str], out_nc: str | None) -> Path:
    if out_nc:
        return Path(out_nc)
    path = Path(cml_nc)
    return path.with_name(f"{path.stem}_QC{path.suffix}")


def _signal_names(ds: xr.Dataset) -> tuple[str, str]:
    """Return transmit/receive signal variable names used by the input file."""
    if {"tsl", "rsl"}.issubset(ds.data_vars):
        return "tsl", "rsl"
    if {"tx", "rx"}.issubset(ds.data_vars):
        return "tx", "rx"
    raise KeyError("Input dataset must contain either tsl/rsl or tx/rx variables.")


def _as_sublink_cml_time(da: xr.DataArray) -> xr.DataArray:
    """Put CML data in sublink_id, cml_id, time order for QC calculations."""
    missing = {"time", "sublink_id", "cml_id"} - set(da.dims)
    if missing:
        raise ValueError(f"{da.name!r} is missing required dimensions: {sorted(missing)}")
    return da.transpose("sublink_id", "cml_id", "time")


def detect_plateaus(series: xr.DataArray, time_span: int = 3, max_thld: float = -85,
                    std_thld: float = 0.5, pad: int = 5) -> xr.DataArray:
    """Match the notebook plateau detector on an RSL time series."""
    low = series.rolling(time=time_span, center=True).max("time") < max_thld
    steady = series.rolling(time=time_span, center=True).std("time") < std_thld
    is_sane = ~(low & steady)
    return ~(is_sane.rolling(time=pad, center=True).min("time") == 0)


def _blackout_mask_1d(rsl: np.ndarray, rsl_threshold: float = -65,
                      max_gap_length: int = 60) -> np.ndarray:
    """Return True for NaN samples in blackout gaps, matching both scan directions."""
    rsl = np.asarray(rsl, dtype="float64")
    isn = np.isnan(rsl)
    below = np.zeros(rsl.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        below = rsl < rsl_threshold

    def scan(values_isn: np.ndarray, values_below: np.ndarray) -> np.ndarray:
        mask = np.zeros(values_isn.shape, dtype=bool)
        n = values_isn.size
        i = 0
        while i < n:
            if not values_isn[i]:
                i += 1
                continue
            j = i
            while j < n and values_isn[j]:
                j += 1
            if (j - i) <= max_gap_length and i > 0 and values_below[i - 1]:
                mask[i:j] = True
            i = j
        return mask

    return scan(isn, below) | scan(isn[::-1], below[::-1])[::-1]


def blackout_filling(tsl: xr.DataArray, rsl: xr.DataArray,
                     max_gap_length: int = 60) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Fill blackout gaps with per-sublink/link TSL max and RSL min values."""
    template = _as_sublink_cml_time(rsl)
    mask_values = np.zeros(template.shape, dtype=bool)
    rsl_values = template.values
    for sublink_idx in tqdm(range(template.sizes["sublink_id"]), desc="blackout sublinks"):
        for cml_idx in range(template.sizes["cml_id"]):
            mask_values[sublink_idx, cml_idx, :] = _blackout_mask_1d(
                rsl_values[sublink_idx, cml_idx, :], max_gap_length=max_gap_length
            )
    mask = xr.DataArray(mask_values, coords=template.coords, dims=template.dims, name="flag_blackout_fill")
    mask = mask.transpose(*rsl.dims)
    tsl_max = tsl.max("time", skipna=True)
    rsl_min = rsl.min("time", skipna=True)
    return tsl.where(~mask, tsl_max), rsl.where(~mask, rsl_min), mask


def _demo_a_b(f_ghz: np.ndarray, pol: xr.DataArray | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Small ITU-like fallback if pycomlink is unavailable."""
    f = np.asarray(f_ghz, dtype="float64")
    a = 4.21e-5 * np.power(f, 2.42)
    b = np.ones_like(a)
    return a, b


def detection_limit_from_kR(length_km: xr.DataArray, frequency: xr.DataArray,
                            polarization: xr.DataArray, A_quantization_dB: float = 0.33) -> xr.DataArray:
    """Calculate detection limit using pycomlink's k-R relation when available."""
    f_ghz = frequency / 1e9 if float(frequency.max()) > 1000 else frequency
    try:
        import pycomlink as pycml
        try:
            a, _ = pycml.processing.k_R_relation.a_b(f_GHz=f_ghz, pol=polarization, approx_type="ITU")
        except TypeError:
            a, _ = pycml.processing.k_R_relation.a_b(f_GHz=f_ghz, pol=polarization, a_b_approximation="ITU")
    except Exception:
        a, _ = _demo_a_b(f_ghz, polarization)
    sensitivity = xr.DataArray(a, coords=frequency.coords, dims=frequency.dims) * length_km
    return A_quantization_dB / sensitivity.where(sensitivity > 0)


def temporal_sanity_check(trsl: xr.DataArray, tspan: int, thld: float, perc: float) -> xr.DataArray:
    """Graf-style temporal standard-deviation sanity check."""
    roll_std = trsl.rolling(time=tspan, center=True).std()
    high_std = roll_std > thld
    low_std = roll_std <= thld
    denom = high_std.sum(dim="time") + low_std.sum(dim="time")
    return (high_std.sum(dim="time") / denom) <= perc


def availability_check(trsl: xr.DataArray, perc_thld: float = 0.5) -> xr.DataArray:
    """Return True where less than perc_thld of TRSL samples are missing."""
    return (trsl.isnull().sum("time") / trsl.sizes["time"]) < perc_thld


def _normalize_pol(pol: xr.DataArray) -> xr.DataArray:
    if pol.dtype.kind == "S":
        return pol.astype(str)
    return pol


def build_qc_dataset(ds: xr.Dataset, proc_line: str = "full") -> xr.Dataset:
    tx_name, rx_name = _signal_names(ds)
    out = ds.copy()
    tsl = out[tx_name]
    rsl = out[rx_name]

    if proc_line == "full":
        out["isSane_plateaus_rsl"] = detect_plateaus(rsl)
        out[tx_name] = tsl.where(out["isSane_plateaus_rsl"])
        out[rx_name] = rsl.where(out["isSane_plateaus_rsl"])
        out[tx_name], out[rx_name], out["flag_blackout_fill"] = blackout_filling(out[tx_name], out[rx_name])
    else:
        out["isSane_plateaus_rsl"] = xr.ones_like(rsl, dtype=bool)
        out["flag_blackout_fill"] = xr.zeros_like(rsl, dtype=bool)

    out["trsl"] = out[tx_name] - out[rx_name]
    if proc_line in {"graf_2020", "full"}:
        out["trsl"] = out.trsl.interpolate_na(dim="time", method="linear", max_gap="5min")

    length = out["length"]
    frequency = out["frequency"]
    polarization = _normalize_pol(out["polarization"])
    if frequency.dims != polarization.dims:
        frequency = frequency.transpose(*polarization.dims)
    out["detection_limit"] = detection_limit_from_kR(length, frequency, polarization)
    out["isSane_detection_limit"] = out.detection_limit < 2
    out["isSane_std_long"] = temporal_sanity_check(out.trsl, tspan=300, thld=2, perc=0.1)
    out["isSane_std_short"] = temporal_sanity_check(out.trsl, tspan=60, thld=0.8, perc=0.33)
    out["isSane_available"] = availability_check(out.trsl, perc_thld=0.5)

    if proc_line == "graf_2020":
        sane_sub = out.isSane_std_long & out.isSane_std_short
    elif proc_line == "full":
        sane_sub = (out.isSane_std_long & out.isSane_std_short &
                    out.isSane_detection_limit & out.isSane_available)
    elif proc_line == "no_filter":
        sane_sub = xr.ones_like(out.isSane_std_long, dtype=bool)
    else:  # pragma: no cover
        raise ValueError("proc_line must be no_filter, graf_2020, or full")

    out["isSane_sublink"] = sane_sub
    out["isSane"] = sane_sub.any("sublink_id")
    out = out.assign_coords(proc_line=proc_line)
    out.attrs["qc_processing_line"] = proc_line
    out.attrs["qc_notes"] = (
        "QC follows plateau detection, blackout filling, TRSL interpolation, "
        "detection-limit, temporal-STD, and availability checks from the supplied workflow."
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CML QC flags and save CML_*_QC.nc.")
    parser.add_argument("--cml-nc", required=True, help="Input CML NetCDF file.")
    parser.add_argument("--out-nc", default=None, help="Output NetCDF. Defaults to <input_stem>_QC.nc.")
    parser.add_argument("--proc-line", choices=["no_filter", "graf_2020", "full"], default="full")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    args = parser.parse_args()

    out_nc = _infer_output_path(args.cml_nc, args.out_nc)
    if out_nc.exists() and not args.overwrite:
        raise FileExistsError(f"{out_nc} exists. Use --overwrite or choose --out-nc.")

    with xr.open_dataset(args.cml_nc) as ds:
        qc = build_qc_dataset(ds, proc_line=args.proc_line)
        encoding = {name: {"zlib": True} for name, var in qc.variables.items() if var.dims}
        out_nc.parent.mkdir(parents=True, exist_ok=True)
        qc.to_netcdf(out_nc, encoding=encoding)
    print(f"Saved QC CML dataset: {out_nc}")


if __name__ == "__main__":
    main()
