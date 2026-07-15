#!/usr/bin/env python3
"""Fast, chunked CML QC filtering and flag export.

Run from a shell with every continuation backslash at the *end* of a line::

    python cml_real_pipeline_package/cml_qc_filter.py \
      --cml-nc CML_S2020E2021_1MIN_CZ.nc \
      --out-nc CML_S2020E2021_1MIN_CZ_QC.nc \
      --proc-line full \
      --n-workers 48 \
      --link-batch 96 \
      --overwrite

For a quick runtime estimate first::

    python cml_real_pipeline_package/cml_qc_filter.py \
      --cml-nc CML_S2020E2021_1MIN_CZ.nc \
      --out-nc CML_S2020E2021_1MIN_CZ_QC.nc \
      --proc-line full \
      --n-workers 48 \
      --benchmark-links 10 \
      --estimate-total-links 2000 \
      --overwrite
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
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
    if {"tsl", "rsl"}.issubset(ds.data_vars):
        return "tsl", "rsl"
    if {"tx", "rx"}.issubset(ds.data_vars):
        return "tx", "rx"
    raise KeyError("Input dataset must contain either tsl/rsl or tx/rx variables.")


def _signal_as_time_sublink_cml(da: xr.DataArray) -> xr.DataArray:
    missing = {"time", "sublink_id", "cml_id"} - set(da.dims)
    if missing:
        raise ValueError(f"{da.name!r} is missing required dimensions: {sorted(missing)}")
    return da.transpose("time", "sublink_id", "cml_id")


def _rolling_std_1d(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    finite = np.isfinite(x)
    vals = np.where(finite, x, 0.0)
    cnt = np.concatenate(([0.0], np.cumsum(finite, dtype=np.float64)))
    s1 = np.concatenate(([0.0], np.cumsum(vals)))
    s2 = np.concatenate(([0.0], np.cumsum(vals * vals)))
    half_left = window // 2
    half_right = window - half_left
    out = np.full(n, np.nan, dtype=np.float32)
    idx = np.arange(n)
    lo = np.maximum(0, idx - half_left)
    hi = np.minimum(n, idx + half_right)
    c = cnt[hi] - cnt[lo]
    good = c > 0
    mean = np.zeros(n, dtype=np.float64)
    mean[good] = (s1[hi[good]] - s1[lo[good]]) / c[good]
    var = np.zeros(n, dtype=np.float64)
    var[good] = (s2[hi[good]] - s2[lo[good]]) / c[good] - mean[good] ** 2
    out[good] = np.sqrt(np.maximum(var[good], 0.0)).astype(np.float32)
    return out


def _rolling_max_minperiod1(x: np.ndarray, window: int) -> np.ndarray:
    # Small windows are used for plateau detection, so this simple loop is fast.
    n = x.size
    out = np.full(n, np.nan, dtype=np.float32)
    half_left = window // 2
    half_right = window - half_left
    for i in range(n):
        part = x[max(0, i - half_left): min(n, i + half_right)]
        if np.isfinite(part).any():
            out[i] = np.nanmax(part)
    return out


def _detect_plateaus_np(rsl: np.ndarray, time_span: int = 3, max_thld: float = -85,
                        std_thld: float = 0.5, pad: int = 5) -> np.ndarray:
    low = _rolling_max_minperiod1(rsl, time_span) < max_thld
    steady = _rolling_std_1d(rsl, time_span) < std_thld
    bad = low & steady
    if pad > 1:
        padded = _rolling_max_minperiod1(bad.astype(np.float32), pad) > 0
        bad = padded
    return ~bad


def detect_plateaus(series: xr.DataArray, time_span: int = 3, max_thld: float = -85,
                    std_thld: float = 0.5, pad: int = 5) -> xr.DataArray:
    """Compatibility wrapper matching the original xarray notebook helper name."""
    template = series.transpose("time", ... ) if "time" in series.dims else series
    values = np.apply_along_axis(
        _detect_plateaus_np,
        template.get_axis_num("time"),
        template.values,
        time_span,
        max_thld,
        std_thld,
        pad,
    )
    return xr.DataArray(values, coords=template.coords, dims=template.dims).transpose(*series.dims)


def _blackout_mask_1d(rsl: np.ndarray, rsl_threshold: float = -65,
                      max_gap_length: int = 60) -> np.ndarray:
    rsl = np.asarray(rsl, dtype=np.float32)
    isn = np.isnan(rsl)
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


def _interp_nan_maxgap(x: np.ndarray, max_gap: int = 5) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    n = y.size
    finite = np.isfinite(y)
    i = 0
    while i < n:
        if finite[i]:
            i += 1
            continue
        j = i
        while j < n and not finite[j]:
            j += 1
        if (j - i) <= max_gap and i > 0 and j < n:
            y[i:j] = np.linspace(float(y[i - 1]), float(y[j]), j - i + 2, dtype=np.float32)[1:-1]
        i = j
    return y


def _temporal_pass_np(trsl: np.ndarray, tspan: int, thld: float, perc: float) -> bool:
    rs = _rolling_std_1d(trsl, tspan)
    good = np.isfinite(rs)
    if not good.any():
        return False
    high = rs[good] > thld
    low = rs[good] <= thld
    denom = int(high.sum() + low.sum())
    return bool(denom > 0 and (high.sum() / denom) <= perc)


def temporal_sanity_check(trsl: xr.DataArray, tspan: int, thld: float, perc: float) -> xr.DataArray:
    """Compatibility wrapper for the original Graf temporal sanity helper."""
    other_dims = [dim for dim in trsl.dims if dim != "time"]
    stacked = trsl.stack(_series=other_dims) if other_dims else trsl.expand_dims(_series=[0])
    stacked = stacked.transpose("_series", "time")
    vals = [_temporal_pass_np(row, tspan, thld, perc) for row in stacked.values]
    out = xr.DataArray(np.asarray(vals, dtype=bool), coords={"_series": stacked._series}, dims=("_series",))
    if other_dims:
        return out.unstack("_series").transpose(*other_dims)
    return out.isel(_series=0, drop=True)


def availability_check(trsl: xr.DataArray, perc_thld: float = 0.5) -> xr.DataArray:
    """Compatibility wrapper for the original availability helper."""
    return (trsl.isnull().sum("time") / trsl.sizes["time"]) < perc_thld


def _fallback_a(freq_ghz: np.ndarray) -> np.ndarray:
    return 4.21e-5 * np.power(np.asarray(freq_ghz, dtype=np.float64), 2.42)


def _detection_limit_np(length_km: float, frequency: np.ndarray) -> np.ndarray:
    f_ghz = np.asarray(frequency, dtype=np.float64)
    f_ghz = np.where(f_ghz > 1000, f_ghz / 1e9, f_ghz)
    sensitivity = _fallback_a(f_ghz) * float(length_km)
    out = np.full(f_ghz.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(sensitivity) & (sensitivity > 0)
    out[ok] = (0.33 / sensitivity[ok]).astype(np.float32)
    return out


def _process_one_link(payload):
    ci, tx_link, rx_link, length, frequency, proc_line = payload
    # tx/rx link shape: sublink, time
    n_sub, n_time = rx_link.shape
    plateau = np.ones((n_sub, n_time), dtype=bool)
    blackout = np.zeros((n_sub, n_time), dtype=bool)
    tx_out = tx_link.astype(np.float32, copy=True)
    rx_out = rx_link.astype(np.float32, copy=True)
    trsl = np.empty((n_sub, n_time), dtype=np.float32)
    det = _detection_limit_np(length, frequency)
    std_long = np.zeros(n_sub, dtype=bool)
    std_short = np.zeros(n_sub, dtype=bool)
    available = np.zeros(n_sub, dtype=bool)
    for sl in range(n_sub):
        if proc_line == "full":
            plateau[sl] = _detect_plateaus_np(rx_out[sl])
            tx_out[sl, ~plateau[sl]] = np.nan
            rx_out[sl, ~plateau[sl]] = np.nan
            blackout[sl] = _blackout_mask_1d(rx_out[sl])
            if blackout[sl].any():
                if np.isfinite(tx_out[sl]).any() and np.isfinite(rx_out[sl]).any():
                    tx_out[sl, blackout[sl]] = np.nanmax(tx_out[sl])
                    rx_out[sl, blackout[sl]] = np.nanmin(rx_out[sl])
        tr = tx_out[sl] - rx_out[sl]
        if proc_line in {"graf_2020", "full"}:
            tr = _interp_nan_maxgap(tr, 5)
        trsl[sl] = tr
        std_long[sl] = _temporal_pass_np(tr, 300, 2.0, 0.1)
        std_short[sl] = _temporal_pass_np(tr, 60, 0.8, 0.33)
        available[sl] = bool((np.isnan(tr).sum() / tr.size) < 0.5)
    det_pass = det < 2
    if proc_line == "graf_2020":
        sane_sub = std_long & std_short
    elif proc_line == "full":
        sane_sub = std_long & std_short & det_pass & available
    else:
        sane_sub = np.ones(n_sub, dtype=bool)
    return ci, tx_out, rx_out, plateau, blackout, trsl, det, det_pass, std_long, std_short, available, sane_sub, bool(sane_sub.any())


def _make_attrs(ds: xr.Dataset) -> xr.Dataset:
    meanings = {
        "isSane_plateaus_rsl": "Per-sample True where RSL is not a low, steady plateau; full processing sets tx/rx to NaN where False.",
        "flag_blackout_fill": "Per-sample True where a short NaN gap bracketed by low RSL was filled with link/sublink tx max and rx min.",
        "trsl": "Total received signal loss after plateau filtering, blackout filling, and optional short-gap interpolation.",
        "detection_limit": "Per-link/sublink rain-rate detection limit in mm/h from A_quantization_dB/(a*length).",
        "isSane_detection_limit": "Per-link/sublink True where detection_limit < 2 mm/h.",
        "isSane_std_long": "Per-link/sublink Graf temporal STD long-window flag: rolling 300-min STD >2 dB for <=10% of valid windows.",
        "isSane_std_short": "Per-link/sublink Graf temporal STD short-window flag: rolling 60-min STD >0.8 dB for <=33% of valid windows.",
        "isSane_available": "Per-link/sublink True where less than 50% of TRSL samples are missing.",
        "isSane_sublink": "Per-link/sublink final QC flag for selected proc_line.",
        "isSane": "Per-link final QC flag, True if any sublink passes isSane_sublink.",
    }
    for name, meaning in meanings.items():
        if name in ds:
            ds[name].attrs["meaning"] = meaning
    ds.attrs["qc_flag_meanings"] = " | ".join(f"{k}: {v}" for k, v in meanings.items())
    return ds


def build_qc_dataset(ds: xr.Dataset, proc_line: str = "full", n_workers: int = 1,
                     link_batch: int = 64, max_links: int | None = None) -> xr.Dataset:
    tx_name, rx_name = _signal_names(ds)
    if max_links is not None:
        ds = ds.isel(cml_id=slice(0, int(max_links)))
    out = ds.copy(deep=False)
    tx_da = _signal_as_time_sublink_cml(ds[tx_name])
    rx_da = _signal_as_time_sublink_cml(ds[rx_name])
    n_time, n_sub, n_cml = tx_da.shape
    shape_ts = (n_time, n_sub, n_cml)
    shape_sc = (n_sub, n_cml)
    tx_qc = np.empty(shape_ts, dtype=np.float32)
    rx_qc = np.empty(shape_ts, dtype=np.float32)
    plateau = np.empty(shape_ts, dtype=bool)
    blackout = np.empty(shape_ts, dtype=bool)
    trsl = np.empty(shape_ts, dtype=np.float32)
    detection_limit = np.empty(shape_sc, dtype=np.float32)
    det_pass = np.empty(shape_sc, dtype=bool)
    std_long = np.empty(shape_sc, dtype=bool)
    std_short = np.empty(shape_sc, dtype=bool)
    available = np.empty(shape_sc, dtype=bool)
    sane_sub = np.empty(shape_sc, dtype=bool)
    sane_link = np.empty(n_cml, dtype=bool)
    lengths = ds["length"].values
    freq = ds["frequency"].transpose("sublink_id", "cml_id").values

    start = time.perf_counter()
    executor = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
    try:
        for i0 in tqdm(range(0, n_cml, link_batch), desc="QC link batches"):
            i1 = min(i0 + link_batch, n_cml)
            tx_block = tx_da.isel(cml_id=slice(i0, i1)).values.transpose(2, 1, 0)
            rx_block = rx_da.isel(cml_id=slice(i0, i1)).values.transpose(2, 1, 0)
            jobs = [(i0 + j, tx_block[j], rx_block[j], lengths[i0 + j], freq[:, i0 + j], proc_line)
                    for j in range(i1 - i0)]
            if executor is not None:
                results = list(executor.map(_process_one_link, jobs, chunksize=1))
            else:
                results = [_process_one_link(job) for job in jobs]
            for res in results:
                ci, tx_o, rx_o, plat, black, tr, det, dpass, slong, sshort, avail, ssub, slink = res
                tx_qc[:, :, ci] = tx_o.T
                rx_qc[:, :, ci] = rx_o.T
                plateau[:, :, ci] = plat.T
                blackout[:, :, ci] = black.T
                trsl[:, :, ci] = tr.T
                detection_limit[:, ci] = det
                det_pass[:, ci] = dpass
                std_long[:, ci] = slong
                std_short[:, ci] = sshort
                available[:, ci] = avail
                sane_sub[:, ci] = ssub
                sane_link[ci] = slink
    finally:
        if executor is not None:
            executor.shutdown()
    elapsed = time.perf_counter() - start
    out[tx_name] = (tx_da.dims, tx_qc)
    out[rx_name] = (rx_da.dims, rx_qc)
    out["isSane_plateaus_rsl"] = (tx_da.dims, plateau)
    out["flag_blackout_fill"] = (tx_da.dims, blackout)
    out["trsl"] = (tx_da.dims, trsl)
    sdims = ("sublink_id", "cml_id")
    out["detection_limit"] = (sdims, detection_limit)
    out["isSane_detection_limit"] = (sdims, det_pass)
    out["isSane_std_long"] = (sdims, std_long)
    out["isSane_std_short"] = (sdims, std_short)
    out["isSane_available"] = (sdims, available)
    out["isSane_sublink"] = (sdims, sane_sub)
    out["isSane"] = (("cml_id",), sane_link)
    out = out.assign_coords(proc_line=proc_line)
    out.attrs["qc_processing_line"] = proc_line
    out.attrs["qc_elapsed_seconds_excluding_save"] = float(elapsed)
    out.attrs["qc_n_workers"] = int(n_workers)
    out.attrs["qc_link_batch"] = int(link_batch)
    out.attrs["qc_notes"] = "Fast chunked implementation of the supplied plateau, blackout, detection-limit, temporal-STD, availability, and final CML sanity workflow."
    return _make_attrs(out)


def _encoding(qc: xr.Dataset, compression: bool) -> dict:
    enc = {}
    for name, var in qc.variables.items():
        if not var.dims:
            continue
        e = {}
        if compression:
            e.update({"zlib": True, "complevel": 1, "shuffle": True})
        if var.dtype == bool:
            e["dtype"] = "i1"
        elif var.dtype == np.float64:
            e["dtype"] = "float32"
        enc[name] = e
    return enc


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fast CML QC flags and save CML_*_QC.nc.")
    parser.add_argument("--cml-nc", required=True, help="Input CML NetCDF file.")
    parser.add_argument("--out-nc", default=None, help="Output NetCDF. Defaults to <input_stem>_QC.nc.")
    parser.add_argument("--proc-line", choices=["no_filter", "graf_2020", "full"], default="full")
    parser.add_argument("--n-workers", type=int, default=min(8, os.cpu_count() or 1), help="Parallel worker processes. Use 48 on your server.")
    parser.add_argument("--link-batch", type=int, default=64, help="Links loaded and processed at once.")
    parser.add_argument("--benchmark-links", type=int, default=0, help="Process only this many links and print a runtime estimate.")
    parser.add_argument("--estimate-total-links", type=int, default=2000, help="Total links used for benchmark extrapolation.")
    parser.add_argument("--no-compression", action="store_true", help="Faster/larger output; useful for speed tests.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    args = parser.parse_args()

    out_nc = _infer_output_path(args.cml_nc, args.out_nc)
    if out_nc.exists() and not args.overwrite:
        raise FileExistsError(f"{out_nc} exists. Use --overwrite or choose --out-nc.")

    t0 = time.perf_counter()
    n_links = args.benchmark_links or None
    with xr.open_dataset(args.cml_nc) as ds:
        qc = build_qc_dataset(ds, proc_line=args.proc_line, n_workers=args.n_workers,
                              link_batch=args.link_batch, max_links=n_links)
        out_nc.parent.mkdir(parents=True, exist_ok=True)
        save_t0 = time.perf_counter()
        qc.to_netcdf(out_nc, encoding=_encoding(qc, compression=not args.no_compression))
        save_elapsed = time.perf_counter() - save_t0
    elapsed = time.perf_counter() - t0
    print(f"Saved QC CML dataset: {out_nc}")
    print(f"Elapsed total: {elapsed:.1f} s (save: {save_elapsed:.1f} s)")
    if args.benchmark_links:
        est = elapsed / float(args.benchmark_links) * float(args.estimate_total_links)
        print(f"Estimated {args.estimate_total_links} links: {est/60:.1f} min using {args.n_workers} workers; rerun full without --benchmark-links.")


if __name__ == "__main__":
    main()
