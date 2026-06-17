#!/usr/bin/env python3
"""Optimized SEVIRI -> CML path-weighted extraction.

Creates reusable CML/SEVIRI path weights and extracts
seviri_path_mean(seviri_time, seviri_channel, cml_id).

Typical use with your data:
    python extract_seviri_pathmean_optimized.py \
      --cml-file cml_2020_2021_1min_merged.nc \
      --seviri-dir SEVIRI_nc \
      --start '2021-01-01 00:00:00' --end '2021-09-07 00:00:00' \
      --reset-output

Demo/benchmark:
    python extract_seviri_pathmean_optimized.py --make-demo --demo-dir demo_data
    python extract_seviri_pathmean_optimized.py --benchmark-demo --demo-dir demo_data
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import time
import traceback
from collections import deque, namedtuple
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp

import netCDF4 as nc4
import numpy as np
from pyproj import CRS, Proj, Transformer

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

FlatWeights = namedtuple("FlatWeights", "pix wts link_ids n_cml n_weights ny nx n_pixels intersects")
_GLOBAL_CHANNELS = None
_GLOBAL_FW = None

DEFAULT_CHANNELS = ["IR_039", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134", "WV_062", "WV_073"]
EPOCH = np.datetime64("1970-01-01T00:00:00", "ns")


def dt64_ns(value: str | np.datetime64) -> np.datetime64:
    return np.datetime64(value, "ns")


def ns_from_dt64(values) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[ns]").astype("int64")


def parse_seviri_time_from_name(path):
    m = re.search(r"SEVIRI_(\d{8})_(\d{6})\.nc$", Path(path).name)
    if not m:
        return None
    return np.datetime64(datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"), "ns")


def read_seviri_time(path):
    try:
        with nc4.Dataset(path) as ds:
            if "time" not in ds.variables or ds.variables["time"].size == 0:
                return parse_seviri_time_from_name(path)
            v = ds.variables["time"]
            raw = np.asarray(v[:]).ravel()[0]
            units = getattr(v, "units", None)
            if units:
                return np.datetime64(nc4.num2date(raw, units, only_use_cftime_datetimes=False), "ns")
            return np.datetime64(int(raw), "ns")
    except Exception:
        return parse_seviri_time_from_name(path)


def list_seviri_files(seviri_dir, start, end):
    rows = []
    for f in sorted(glob.glob(str(Path(seviri_dir) / "SEVIRI_*.nc"))):
        t = parse_seviri_time_from_name(f)
        if t is None:
            t = read_seviri_time(f)
        if t is not None and start <= t <= end:
            rows.append((t, f))
    if not rows:
        raise FileNotFoundError(f"No SEVIRI files found in {seviri_dir} between {start} and {end}")
    rows.sort(key=lambda x: x[0])
    return [f for _, f in rows], np.asarray([t for t, _ in rows], dtype="datetime64[ns]")


def make_edges(c):
    c = np.asarray(c, dtype="float64")
    if c.size < 2:
        raise ValueError("Need at least two coordinate values")
    return np.concatenate(([c[0] - 0.5 * (c[1] - c[0])], 0.5 * (c[:-1] + c[1:]), [c[-1] + 0.5 * (c[-1] - c[-2])]))


def read_var_1d(ds, names):
    for name in names:
        if name in ds.variables:
            return np.asarray(ds.variables[name][:])
    raise KeyError(f"Missing one of {names}")


def get_projection_from_seviri(seviri_ref):
    with nc4.Dataset(seviri_ref) as ds:
        H = float(getattr(ds, "satellite_height_m"))
        lon0 = float(getattr(ds, "lon_0_deg"))
        a = float(getattr(ds, "semi_major_axis_m"))
        b = float(getattr(ds, "semi_minor_axis_m"))
        sweep = getattr(ds, "sweep_angle_axis", "y")
    geos = Proj(proj="geos", h=H, lon_0=lon0, a=a, b=b, sweep=sweep)
    return Transformer.from_proj(CRS.from_proj4(f"+proj=longlat +a={a} +b={b}"), geos, always_xy=True), H


def project_cml(cml_file, seviri_ref, x_grid, y_grid):
    with nc4.Dataset(cml_file) as ds:
        lon_a = read_var_1d(ds, ["site_a_longitude"]).astype("float64").ravel()
        lat_a = read_var_1d(ds, ["site_a_latitude"]).astype("float64").ravel()
        lon_b = read_var_1d(ds, ["site_b_longitude"]).astype("float64").ravel()
        lat_b = read_var_1d(ds, ["site_b_latitude"]).astype("float64").ravel()
        cml_ids = np.asarray(ds.variables["cml_id"][:] if "cml_id" in ds.variables else np.arange(lon_a.size))
    transformer, H = get_projection_from_seviri(seviri_ref)
    x0, y0 = transformer.transform(lon_a, lat_a)
    x1, y1 = transformer.transform(lon_b, lat_b)
    if max(float(np.nanmax(np.abs(x_grid))), float(np.nanmax(np.abs(y_grid)))) < 10.0:
        x0, y0, x1, y1 = x0 / H, y0 / H, x1 / H, y1 / H
    return x0, y0, x1, y1, cml_ids


def interval_index(edges_inc, x):
    return int(np.searchsorted(edges_inc, x, side="right") - 1)


def raw_weights_fast(x0, y0, x1, y1, x_edges, y_edges):
    """Return row, col, normalized path weights using grid-line crossing parameters."""
    if not all(map(np.isfinite, (x0, y0, x1, y1))):
        return []
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length <= 0:
        return []

    x_inc, y_inc = x_edges[0] < x_edges[-1], y_edges[0] < y_edges[-1]
    xe = x_edges if x_inc else x_edges[::-1]
    ye = y_edges if y_inc else y_edges[::-1]
    xmin, xmax, ymin, ymax = xe[0], xe[-1], ye[0], ye[-1]

    # Liang-Barsky clip to whole grid.
    t0, t1 = 0.0, 1.0
    for p, q in [(-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)]:
        if abs(p) < 1e-15:
            if q < 0:
                return []
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 >= t1:
                return []

    ts = [t0, t1]
    if abs(dx) > 1e-15:
        lo, hi = sorted((x0 + t0 * dx, x0 + t1 * dx))
        for e in xe[np.searchsorted(xe, lo, side="right"):np.searchsorted(xe, hi, side="left")]:
            ts.append((e - x0) / dx)
    if abs(dy) > 1e-15:
        lo, hi = sorted((y0 + t0 * dy, y0 + t1 * dy))
        for e in ye[np.searchsorted(ye, lo, side="right"):np.searchsorted(ye, hi, side="left")]:
            ts.append((e - y0) / dy)

    ts = np.unique(np.clip(np.asarray(ts, dtype="float64"), t0, t1))
    out = []
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    for a, b in zip(ts[:-1], ts[1:]):
        if b <= a:
            continue
        tm = 0.5 * (a + b)
        ix_inc = interval_index(xe, x0 + tm * dx)
        iy_inc = interval_index(ye, y0 + tm * dy)
        if 0 <= ix_inc < nx and 0 <= iy_inc < ny:
            ix = ix_inc if x_inc else nx - 1 - ix_inc
            iy = iy_inc if y_inc else ny - 1 - iy_inc
            out.append((iy, ix, float(b - a)))
    return out


def build_flat_weights(cml_file, seviri_ref, weights_file):
    with nc4.Dataset(seviri_ref) as ds:
        x_grid = np.asarray(ds.variables["x"][:], dtype="float64")
        y_grid = np.asarray(ds.variables["y"][:], dtype="float64")
    x_edges, y_edges = make_edges(x_grid), make_edges(y_grid)
    ny, nx = len(y_grid), len(x_grid)
    x0, y0, x1, y1, cml_ids = project_cml(cml_file, seviri_ref, x_grid, y_grid)
    all_pix, all_wts, all_ids = [], [], []
    n_pixels = np.zeros(len(cml_ids), dtype="int32")
    intersects = np.zeros(len(cml_ids), dtype="int8")
    for i in range(len(cml_ids)):
        raw = raw_weights_fast(x0[i], y0[i], x1[i], y1[i], x_edges, y_edges)
        if not raw:
            continue
        rows = np.asarray([r for r, _, _ in raw], dtype="int64")
        cols = np.asarray([c for _, c, _ in raw], dtype="int64")
        wts = np.asarray([w for _, _, w in raw], dtype="float32")
        wts /= wts.sum()
        pix = rows * nx + cols
        all_pix.append(pix); all_wts.append(wts); all_ids.append(np.full(pix.size, i, dtype="int32"))
        n_pixels[i] = pix.size; intersects[i] = 1
    if not all_pix:
        raise RuntimeError("No CML intersects the SEVIRI grid")
    np.savez_compressed(weights_file, pix=np.concatenate(all_pix).astype("int64"), wts=np.concatenate(all_wts).astype("float32"), link_ids=np.concatenate(all_ids).astype("int32"), n_cml=np.array(len(cml_ids)), n_weights=np.array(sum(map(len, all_pix))), ny=np.array(ny), nx=np.array(nx), n_pixels=n_pixels, intersects=intersects, cml_ids=cml_ids)


def load_flat_weights(weights_file):
    d = np.load(weights_file, allow_pickle=True)
    return FlatWeights(d["pix"].astype("int64"), d["wts"].astype("float32"), d["link_ids"].astype("int32"), int(d["n_cml"]), int(d["n_weights"]), int(d["ny"]), int(d["nx"]), d["n_pixels"].astype("int32"), d["intersects"].astype("int8")), d["cml_ids"]


def init_worker(channels, fw):
    global _GLOBAL_CHANNELS, _GLOBAL_FW
    _GLOBAL_CHANNELS, _GLOBAL_FW = list(channels), fw


def path_mean_one_channel(arr, fw):
    if np.ma.isMaskedArray(arr):
        arr = np.ma.filled(arr, np.nan)
    arr = np.asarray(arr, dtype="float32")
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    vals = arr.reshape(-1)[fw.pix].astype("float64", copy=False)
    good = np.isfinite(vals)
    out = np.full(fw.n_cml, np.nan, dtype="float32")
    if good.any():
        ids = fw.link_ids[good]
        ws = fw.wts[good].astype("float64", copy=False)
        sums = np.bincount(ids, weights=vals[good] * ws, minlength=fw.n_cml)
        wsum = np.bincount(ids, weights=ws, minlength=fw.n_cml)
        ok = wsum > 0
        out[ok] = (sums[ok] / wsum[ok]).astype("float32")
    return out


def process_scene(task):
    time_idx, scene_path = task
    result = np.full((len(_GLOBAL_CHANNELS), _GLOBAL_FW.n_cml), np.nan, dtype="float32")
    try:
        with nc4.Dataset(scene_path) as ds:
            for ci, ch in enumerate(_GLOBAL_CHANNELS):
                if ch in ds.variables:
                    result[ci] = path_mean_one_channel(ds.variables[ch][:], _GLOBAL_FW)
        return time_idx, scene_path, result, None
    except Exception:
        return time_idx, scene_path, result, traceback.format_exc()


def create_output(out_file, times, channels, cml_ids, fw, chunk_time, chunk_cml, complevel):
    with nc4.Dataset(out_file, "w", format="NETCDF4") as dst:
        dst.createDimension("seviri_time", len(times)); dst.createDimension("seviri_channel", len(channels)); dst.createDimension("cml_id", fw.n_cml)
        vt = dst.createVariable("seviri_time", "i8", ("seviri_time",), zlib=True, complevel=complevel)
        vt.units = "nanoseconds since 1970-01-01 00:00:00 UTC"; vt.long_name = "SEVIRI scan time"; vt[:] = ns_from_dt64(times)
        vc = dst.createVariable("seviri_channel", str, ("seviri_channel",)); vc[:] = np.asarray(channels, dtype=object)
        ids = np.asarray(cml_ids)
        vi = dst.createVariable("cml_id", "i8" if np.issubdtype(ids.dtype, np.integer) else str, ("cml_id",)); vi[:] = ids.astype("int64") if np.issubdtype(ids.dtype, np.integer) else ids.astype(str).astype(object)
        vn = dst.createVariable("seviri_n_pixels", "i4", ("cml_id",), zlib=True, complevel=complevel); vn[:] = fw.n_pixels
        vg = dst.createVariable("seviri_intersects_grid", "i1", ("cml_id",), zlib=True, complevel=complevel); vg[:] = fw.intersects
        v = dst.createVariable("seviri_path_mean", "f4", ("seviri_time", "seviri_channel", "cml_id"), zlib=True, complevel=complevel, fill_value=np.float32(np.nan), chunksizes=(min(chunk_time, len(times)), len(channels), min(chunk_cml, fw.n_cml)))
        v.long_name = "SEVIRI path-weighted mean brightness temperature along CML path"; v.units = "K"
        dst.description = "Path-weighted SEVIRI brightness temperatures along CML paths."


def extract(seviri_files, seviri_times, channels, fw, cml_ids, out_file, workers=6, max_inflight=None, write_buffer=32, chunk_time=32, chunk_cml=1024, complevel=1, start_method="fork"):
    if Path(out_file).exists():
        raise FileExistsError(f"{out_file} already exists; use --reset-output")
    create_output(out_file, seviri_times, channels, cml_ids, fw, chunk_time, chunk_cml, complevel)
    tasks = list(enumerate(seviri_files)); max_inflight = max_inflight or 4 * workers
    buf = np.full((write_buffer, len(channels), fw.n_cml), np.nan, dtype="float32")
    buf_start = buf_count = written = 0; errors = []
    ctx = mp.get_context(start_method)
    pool = ctx.Pool(processes=workers, initializer=init_worker, initargs=(channels, fw))
    try:
        task_iter, pending = iter(tasks), deque()
        for _ in range(min(max_inflight, len(tasks))):
            pending.append(pool.apply_async(process_scene, (next(task_iter),)))
        progress = tqdm(total=len(tasks), desc="SEVIRI scenes", unit="scene") if tqdm else None
        with nc4.Dataset(out_file, "a") as dst:
            out_var = dst.variables["seviri_path_mean"]
            def flush():
                nonlocal buf_start, buf_count
                if buf_count:
                    out_var[buf_start:buf_start + buf_count] = buf[:buf_count]
                    buf_start += buf_count; buf_count = 0; buf[:] = np.nan
            while pending:
                idx, scene, result, err = pending.popleft().get()
                try:
                    pending.append(pool.apply_async(process_scene, (next(task_iter),)))
                except StopIteration:
                    pass
                if err is None:
                    buf[buf_count] = result; written += 1
                else:
                    errors.append((scene, err))
                buf_count += 1; result = None
                if progress: progress.update(1)
                if buf_count == write_buffer: flush()
            flush(); dst.sync()
        if progress: progress.close()
    finally:
        pool.close(); pool.join()
    return {"scenes": len(tasks), "written": written, "failed": len(errors), "errors": errors[:10]}


def make_demo(demo_dir, n_scenes=96, ny=80, nx=100, n_cml=500, seed=7):
    demo = Path(demo_dir); sev = demo / "SEVIRI_nc"
    if demo.exists(): shutil.rmtree(demo)
    sev.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    H, lon0, a, b = 35785831.0, 0.0, 6378169.0, 6356583.8
    x = np.linspace(-0.05, 0.05, nx); y = np.linspace(0.01, 0.11, ny)
    with nc4.Dataset(demo / "cml_demo.nc", "w") as ds:
        ds.createDimension("cml_id", n_cml)
        ds.createVariable("cml_id", "i8", ("cml_id",))[:] = np.arange(n_cml)
        for name in ("site_a_longitude", "site_b_longitude"):
            ds.createVariable(name, "f8", ("cml_id",))[:] = rng.uniform(-8, 8, n_cml)
        for name in ("site_a_latitude", "site_b_latitude"):
            ds.createVariable(name, "f8", ("cml_id",))[:] = rng.uniform(5, 35, n_cml)
    base = np.datetime64("2021-01-01T00:00:00", "s")
    yy, xx = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), indexing="ij")
    for i in range(n_scenes):
        t = base + np.timedelta64(15 * i, "m")
        stamp = str(t).replace("-", "").replace(":", "").replace("T", "_")
        with nc4.Dataset(sev / f"SEVIRI_{stamp}.nc", "w", format="NETCDF4") as ds:
            ds.createDimension("time", 1); ds.createDimension("y", ny); ds.createDimension("x", nx)
            ds.satellite_height_m = H; ds.lon_0_deg = lon0; ds.semi_major_axis_m = a; ds.semi_minor_axis_m = b; ds.sweep_angle_axis = "y"
            ds.createVariable("x", "f8", ("x",))[:] = x; ds.createVariable("y", "f8", ("y",))[:] = y
            tv = ds.createVariable("time", "i8", ("time",)); tv.units = "nanoseconds since 1970-01-01 00:00:00 UTC"; tv[:] = ns_from_dt64([t])
            for ci, ch in enumerate(DEFAULT_CHANNELS):
                ds.createVariable(ch, "f4", ("y", "x"), zlib=True, complevel=1)[:] = (250 + ci * 3 + 5 * xx + 2 * yy + rng.normal(0, 0.5, (ny, nx))).astype("float32")
    return demo


def benchmark_demo(args):
    demo = make_demo(args.demo_dir, args.demo_scenes, args.demo_ny, args.demo_nx, args.demo_cml) if args.make_demo or not Path(args.demo_dir).exists() else Path(args.demo_dir)
    times = []
    for workers in args.benchmark_workers:
        weights = demo / f"weights_w{workers}.npz"; out = demo / f"out_w{workers}.nc"
        if weights.exists(): weights.unlink()
        if out.exists(): out.unlink()
        start = time.perf_counter(); build_flat_weights(demo / "cml_demo.nc", sorted((demo / "SEVIRI_nc").glob("SEVIRI_*.nc"))[0], weights); weight_s = time.perf_counter() - start
        files, stimes = list_seviri_files(demo / "SEVIRI_nc", dt64_ns("2021-01-01"), dt64_ns("2021-12-31"))
        start = time.perf_counter(); fw, cml_ids = load_flat_weights(weights); stats = extract(files, stimes, DEFAULT_CHANNELS, fw, cml_ids, out, workers=workers, max_inflight=4*workers, write_buffer=args.write_buffer, chunk_time=args.write_buffer, chunk_cml=args.chunk_cml, complevel=args.complevel); extract_s = time.perf_counter() - start
        row = {"workers": workers, "weight_seconds": round(weight_s, 3), "extract_seconds": round(extract_s, 3), "scenes_per_second": round(stats["scenes"] / extract_s, 3)}
        print(json.dumps(row)); times.append(row)
    best = max(times, key=lambda r: r["scenes_per_second"]); print("BEST", json.dumps(best))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cml-file", default="cml_2020_2021_1min_merged.nc"); p.add_argument("--seviri-dir", default="SEVIRI_nc"); p.add_argument("--weights-file", default="cml_seviri_path_weights.npz"); p.add_argument("--out-file", default="seviri_path_mean_cml.nc")
    p.add_argument("--start", default="2021-01-01 00:00:00"); p.add_argument("--end", default="2021-09-07 00:00:00"); p.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    p.add_argument("--workers", type=int, default=6); p.add_argument("--max-inflight", type=int); p.add_argument("--write-buffer", type=int, default=32); p.add_argument("--chunk-cml", type=int, default=1024); p.add_argument("--complevel", type=int, default=1); p.add_argument("--start-method", default="fork", choices=("fork", "spawn", "forkserver"))
    p.add_argument("--reset-weights", action="store_true"); p.add_argument("--reset-output", action="store_true")
    p.add_argument("--make-demo", action="store_true"); p.add_argument("--benchmark-demo", action="store_true"); p.add_argument("--demo-dir", default="demo_seviri_cml"); p.add_argument("--demo-scenes", type=int, default=96); p.add_argument("--demo-ny", type=int, default=80); p.add_argument("--demo-nx", type=int, default=100); p.add_argument("--demo-cml", type=int, default=500); p.add_argument("--benchmark-workers", type=int, nargs="+", default=[1, 2, 4, 6])
    args = p.parse_args()
    if args.benchmark_demo:
        benchmark_demo(args); return
    if args.make_demo:
        demo = make_demo(args.demo_dir, args.demo_scenes, args.demo_ny, args.demo_nx, args.demo_cml); print(f"Demo data written to {demo}"); return
    if args.reset_weights and Path(args.weights_file).exists(): Path(args.weights_file).unlink()
    if args.reset_output and Path(args.out_file).exists(): Path(args.out_file).unlink()
    files, times = list_seviri_files(args.seviri_dir, dt64_ns(args.start), dt64_ns(args.end))
    if not Path(args.weights_file).exists(): build_flat_weights(args.cml_file, files[0], args.weights_file)
    fw, cml_ids = load_flat_weights(args.weights_file)
    t0 = time.perf_counter(); stats = extract(files, times, args.channels, fw, cml_ids, args.out_file, args.workers, args.max_inflight, args.write_buffer, args.write_buffer, args.chunk_cml, args.complevel, args.start_method); elapsed = time.perf_counter() - t0
    print(json.dumps({"elapsed_seconds": round(elapsed, 3), "scenes_per_second": round(len(files) / elapsed, 3), **{k: v for k, v in stats.items() if k != "errors"}}, indent=2))


if __name__ == "__main__":
    main()
