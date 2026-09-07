"""Standalone SEVIRI-CML CNN wet/dry inference.

This module intentionally contains only the transformations required by the
trained CNN itself:

1. use/construct the normalized Pmin signal,
2. assemble the exact model features in training order,
3. build four-step 15-minute sequences,
4. apply the saved training scalers,
5. run the saved Keras model,
6. convert the CNN probability to a wet/dry flag.

It does not perform CML QC, resampling, interpolation, CPP extraction, radar or
gauge matching, QPE, WAA, RSTD, Mode classification, fusion, or evaluation.
The input dataset must already be aligned on the model's 15-minute time grid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import xarray as xr


# Latest v0.13 NL PMIN_CPP model contract.
DEFAULT_FEATURES = (
    "Pmin_normalized",
    "cot",
    "cre",
    "cwp",
    "cdnc",
    "cgt",
    "cph",
)
DEFAULT_SEQUENCE_LENGTH = 4
DEFAULT_CADENCE = pd.Timedelta(minutes=15)
DEFAULT_ROLLING_WINDOW = 96       # 24 h at 15-minute cadence
DEFAULT_ROLLING_MIN_PERIODS = 8   # 2 h at 15-minute cadence
DEFAULT_THRESHOLD = 0.5
DEFAULT_SIGNAL_VAR = "rsl_min"
DEFAULT_NORMALIZED_SIGNAL = "Pmin_normalized"


def _load_manifest(model_dir: Path) -> dict[str, Any]:
    """Load optional model metadata without requiring the full validation package."""
    for filename in ("model_manifest.json", "config.json"):
        path = model_dir / filename
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise TypeError(f"{path} must contain a JSON object")
            return data
    return {}


def _manifest_value(manifest: Mapping[str, Any], keys: Sequence[str], default: Any) -> Any:
    for key in keys:
        if key in manifest and manifest[key] is not None:
            return manifest[key]
    return default


def _load_keras_model(model_path: Path):
    """Load model.keras while supporting TensorFlow Keras and standalone Keras."""
    try:
        from tensorflow import keras  # type: ignore
    except ImportError:
        import keras  # type: ignore

    return keras.models.load_model(model_path, compile=False)


def _validate_dataset(
    ds: xr.Dataset,
    *,
    signal_var: str,
    features: Sequence[str],
    time_dim: str,
    cml_dim: str,
) -> None:
    if time_dim not in ds.dims:
        raise ValueError(f"Missing required time dimension: {time_dim!r}")
    if cml_dim not in ds.dims:
        raise ValueError(f"Missing required CML dimension: {cml_dim!r}")

    if DEFAULT_NORMALIZED_SIGNAL not in ds and signal_var not in ds:
        raise KeyError(
            f"Need either {DEFAULT_NORMALIZED_SIGNAL!r} or raw signal {signal_var!r}."
        )

    missing_cpp = [
        name
        for name in features
        if name != DEFAULT_NORMALIZED_SIGNAL and name not in ds
    ]
    if missing_cpp:
        raise KeyError(f"Missing CNN input feature(s): {missing_cpp}")

    time = pd.DatetimeIndex(pd.to_datetime(ds[time_dim].values))
    if not time.is_monotonic_increasing:
        raise ValueError("Input time coordinate must be monotonically increasing.")
    if time.has_duplicates:
        raise ValueError("Input time coordinate contains duplicate timestamps.")



def _select_series(
    da: xr.DataArray,
    *,
    cml_label: Any,
    sublink_label: Any | None,
    cml_dim: str,
    sublink_dim: str,
    time_dim: str,
) -> np.ndarray:
    """Select by coordinate label, never by positional sublink index."""
    indexers: dict[str, Any] = {}
    if cml_dim in da.dims:
        indexers[cml_dim] = cml_label
    if sublink_label is not None and sublink_dim in da.dims:
        indexers[sublink_dim] = sublink_label

    selected = da.sel(indexers) if indexers else da

    extra_dims = [d for d in selected.dims if d != time_dim]
    if extra_dims:
        raise ValueError(
            f"Variable {da.name!r} still has unsupported dimensions after selection: "
            f"{extra_dims}"
        )

    selected = selected.transpose(time_dim)
    return np.asarray(selected.values, dtype=np.float64)


def _normalized_pmin(
    ds: xr.Dataset,
    *,
    cml_label: Any,
    sublink_label: Any | None,
    signal_var: str,
    cml_dim: str,
    sublink_dim: str,
    time_dim: str,
    rolling_window: int,
    rolling_min_periods: int,
) -> np.ndarray:
    """Return Pmin_normalized, using it directly when it is already present.

    If it is absent, construct the v0.13 rolling-median anomaly from the raw
    selected Pmin signal: raw signal minus its trailing rolling median.
    """
    if DEFAULT_NORMALIZED_SIGNAL in ds:
        return _select_series(
            ds[DEFAULT_NORMALIZED_SIGNAL],
            cml_label=cml_label,
            sublink_label=sublink_label,
            cml_dim=cml_dim,
            sublink_dim=sublink_dim,
            time_dim=time_dim,
        )

    raw = _select_series(
        ds[signal_var],
        cml_label=cml_label,
        sublink_label=sublink_label,
        cml_dim=cml_dim,
        sublink_dim=sublink_dim,
        time_dim=time_dim,
    )
    s = pd.Series(raw)
    baseline = s.rolling(
        window=int(rolling_window),
        min_periods=int(rolling_min_periods),
    ).median()
    return (s - baseline).to_numpy(dtype=np.float64)


def _build_feature_matrix(
    ds: xr.Dataset,
    *,
    cml_label: Any,
    sublink_label: Any | None,
    signal_var: str,
    features: Sequence[str],
    cml_dim: str,
    sublink_dim: str,
    time_dim: str,
    rolling_window: int,
    rolling_min_periods: int,
) -> np.ndarray:
    columns: list[np.ndarray] = []

    for feature in features:
        if feature == DEFAULT_NORMALIZED_SIGNAL:
            values = _normalized_pmin(
                ds,
                cml_label=cml_label,
                sublink_label=sublink_label,
                signal_var=signal_var,
                cml_dim=cml_dim,
                sublink_dim=sublink_dim,
                time_dim=time_dim,
                rolling_window=rolling_window,
                rolling_min_periods=rolling_min_periods,
            )
        else:
            values = _select_series(
                ds[feature],
                cml_label=cml_label,
                sublink_label=sublink_label,
                cml_dim=cml_dim,
                sublink_dim=sublink_dim,
                time_dim=time_dim,
            )
        columns.append(values)

    lengths = {len(v) for v in columns}
    if len(lengths) != 1:
        raise ValueError("CNN input variables do not share the same time length.")

    return np.column_stack(columns).astype(np.float64, copy=False)


def _build_sequences(
    matrix: np.ndarray,
    time: pd.DatetimeIndex,
    *,
    sequence_length: int,
    cadence: pd.Timedelta,
) -> tuple[np.ndarray, np.ndarray]:
    """Build strict consecutive sequences and return their ending time indices."""
    seqs: list[np.ndarray] = []
    end_indices: list[int] = []

    for end in range(sequence_length - 1, len(time)):
        start = end - sequence_length + 1
        block = matrix[start : end + 1]
        block_time = time[start : end + 1]

        # No tolerance, interpolation, resampling, or time mismatch is allowed.
        if len(block_time) > 1:
            deltas = np.diff(block_time.asi8)
            if not np.all(deltas == cadence.value):
                continue

        if not np.isfinite(block).all():
            continue

        seqs.append(block)
        end_indices.append(end)

    if not seqs:
        return (
            np.empty((0, sequence_length, matrix.shape[1]), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )

    return np.stack(seqs), np.asarray(end_indices, dtype=np.int64)


def _apply_scalers(
    sequences: np.ndarray,
    scalers: Any,
    feature_names: Sequence[str],
) -> np.ndarray:
    """Apply scalers saved during training without refitting anything.

    Supported saved layouts:
      * mapping feature_name -> fitted scaler,
      * mapping with nested ``scalers`` or ``feature_scalers`` mapping,
      * sequence of fitted scalers in feature order,
      * one fitted scaler trained jointly on all features.
    """
    x = np.asarray(sequences, dtype=np.float64).copy()
    if x.size == 0:
        return x

    if isinstance(scalers, Mapping):
        nested = scalers.get("feature_scalers", scalers.get("scalers"))
        if isinstance(nested, Mapping):
            scalers = nested

    if isinstance(scalers, Mapping):
        missing = [name for name in feature_names if name not in scalers]
        if missing:
            raise KeyError(
                "Saved scaler mapping does not contain the exact CNN feature(s): "
                f"{missing}"
            )
        for j, name in enumerate(feature_names):
            scaler = scalers[name]
            values = x[:, :, j].reshape(-1, 1)
            x[:, :, j] = np.asarray(scaler.transform(values)).reshape(x.shape[0], x.shape[1])
        return x

    if isinstance(scalers, (list, tuple)):
        if len(scalers) != len(feature_names):
            raise ValueError(
                f"Expected {len(feature_names)} saved feature scalers, got {len(scalers)}."
            )
        for j, scaler in enumerate(scalers):
            values = x[:, :, j].reshape(-1, 1)
            x[:, :, j] = np.asarray(scaler.transform(values)).reshape(x.shape[0], x.shape[1])
        return x

    if hasattr(scalers, "transform"):
        flat = x.reshape(-1, x.shape[-1])
        transformed = np.asarray(scalers.transform(flat))
        if transformed.shape != flat.shape:
            raise ValueError(
                "Joint scaler returned an unexpected feature shape: "
                f"{transformed.shape}, expected {flat.shape}."
            )
        return transformed.reshape(x.shape)

    raise TypeError(
        "Unsupported scalers.joblib structure. Expected fitted scaler(s) with transform()."
    )


def _prepare_model_input(model: Any, x: np.ndarray) -> np.ndarray:
    """Adapt only the tensor rank expected by the saved model, not the data values."""
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        if len(shape) != 1:
            raise ValueError("Only single-input CNN models are supported.")
        shape = shape[0]

    if shape is None:
        return x

    rank = len(shape)
    if rank == 3:
        return x
    if rank == 4 and shape[-1] in (1, None):
        return x[..., np.newaxis]
    if rank == 2:
        return x.reshape(x.shape[0], -1)

    raise ValueError(f"Unsupported saved model input shape: {shape}")


def _predict_probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    pred = np.asarray(model.predict(x, verbose=0))

    if pred.ndim == 1:
        prob = pred
    elif pred.ndim == 2 and pred.shape[1] == 1:
        prob = pred[:, 0]
    elif pred.ndim == 2 and pred.shape[1] == 2:
        # Binary softmax: class 1 is wet.
        prob = pred[:, 1]
    else:
        raise ValueError(f"Unexpected CNN output shape: {pred.shape}")

    if not np.isfinite(prob).all():
        raise ValueError("CNN returned non-finite probabilities.")
    return prob.astype(np.float64, copy=False)


def predict_wet_dry(
    ds: xr.Dataset,
    model_dir: str | Path,
    *,
    signal_var: str = DEFAULT_SIGNAL_VAR,
    threshold: float = DEFAULT_THRESHOLD,
    time_dim: str = "time",
    cml_dim: str = "cml_id",
    sublink_dim: str = "sublink_id",
) -> xr.DataArray:
    """Run only the trained CNN and return the wet/dry classification flag.

    Parameters
    ----------
    ds
        Already aligned 15-minute CML+CPP dataset. For NL/PMIN_CPP it must
        contain either ``Pmin_normalized`` or the selected raw Pmin variable,
        plus ``cot``, ``cre``, ``cwp``, ``cdnc``, ``cgt`` and ``cph``.
    model_dir
        Directory containing ``model.keras`` and ``scalers.joblib``. An
        optional ``model_manifest.json`` or ``config.json`` is read when
        available.
    signal_var
        Raw Pmin variable used only when ``Pmin_normalized`` is not already in
        ``ds``. The current min/max workflow uses ``rsl_min``.
    threshold
        CNN probability threshold. Default 0.5 reproduces the current v0.13
        notebook setting.

    Returns
    -------
    xarray.DataArray
        Integer wet/dry flag with the same time/CML coordinates as the input:
        ``1 = wet``, ``0 = dry``, ``-1 = unclassifiable``. If the raw signal
        has a sublink dimension, it is preserved and each sublink is inferred
        independently using coordinate-label selection.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    model_dir = Path(model_dir)
    model_path = model_dir / "model.keras"
    scalers_path = model_dir / "scalers.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not scalers_path.is_file():
        raise FileNotFoundError(scalers_path)

    manifest = _load_manifest(model_dir)
    features = tuple(
        _manifest_value(
            manifest,
            ("features", "feature_names", "input_features"),
            DEFAULT_FEATURES,
        )
    )
    sequence_length = int(
        _manifest_value(manifest, ("sequence_length", "seq_len"), DEFAULT_SEQUENCE_LENGTH)
    )
    rolling_window = int(
        _manifest_value(
            manifest,
            ("normalization_window", "rolling_window", "rolling_window_samples"),
            DEFAULT_ROLLING_WINDOW,
        )
    )
    rolling_min_periods = int(
        _manifest_value(
            manifest,
            ("normalization_minimum", "rolling_min_periods", "min_periods"),
            DEFAULT_ROLLING_MIN_PERIODS,
        )
    )

    if features != DEFAULT_FEATURES:
        raise ValueError(
            "This standalone entry point is intentionally restricted to the latest "
            f"NL/PMIN_CPP CNN feature contract. Got: {features}"
        )
    if sequence_length != DEFAULT_SEQUENCE_LENGTH:
        raise ValueError(
            "This standalone entry point is intentionally restricted to the latest "
            f"NL/PMIN_CPP sequence length ({DEFAULT_SEQUENCE_LENGTH}). Got: "
            f"{sequence_length}"
        )

    _validate_dataset(
        ds,
        signal_var=signal_var,
        features=features,
        time_dim=time_dim,
        cml_dim=cml_dim,
    )

    model = _load_keras_model(model_path)
    scalers = joblib.load(scalers_path)
    time = pd.DatetimeIndex(pd.to_datetime(ds[time_dim].values))
    cml_labels = ds[cml_dim].values

    signal_da = ds[DEFAULT_NORMALIZED_SIGNAL] if DEFAULT_NORMALIZED_SIGNAL in ds else ds[signal_var]
    has_sublinks = sublink_dim in signal_da.dims
    sublink_labels = signal_da[sublink_dim].values if has_sublinks else np.asarray([None], dtype=object)

    if has_sublinks:
        output = np.full(
            (len(time), len(cml_labels), len(sublink_labels)),
            -1,
            dtype=np.int8,
        )
    else:
        output = np.full((len(time), len(cml_labels)), -1, dtype=np.int8)

    for cml_i, cml_label in enumerate(cml_labels):
        for sublink_i, sublink_label in enumerate(sublink_labels):
            matrix = _build_feature_matrix(
                ds,
                cml_label=cml_label,
                sublink_label=sublink_label,
                signal_var=signal_var,
                features=features,
                cml_dim=cml_dim,
                sublink_dim=sublink_dim,
                time_dim=time_dim,
                rolling_window=rolling_window,
                rolling_min_periods=rolling_min_periods,
            )
            sequences, end_indices = _build_sequences(
                matrix,
                time,
                sequence_length=sequence_length,
                cadence=DEFAULT_CADENCE,
            )
            if len(end_indices) == 0:
                continue

            sequences = _apply_scalers(sequences, scalers, features)
            model_input = _prepare_model_input(model, sequences)
            probabilities = _predict_probabilities(model, model_input)
            flags = (probabilities >= float(threshold)).astype(np.int8)

            if has_sublinks:
                output[end_indices, cml_i, sublink_i] = flags
            else:
                output[end_indices, cml_i] = flags

    if has_sublinks:
        result = xr.DataArray(
            output,
            dims=(time_dim, cml_dim, sublink_dim),
            coords={
                time_dim: ds[time_dim],
                cml_dim: ds[cml_dim],
                sublink_dim: signal_da[sublink_dim],
            },
            name="cnn_wet_dry",
        )
    else:
        result = xr.DataArray(
            output,
            dims=(time_dim, cml_dim),
            coords={time_dim: ds[time_dim], cml_dim: ds[cml_dim]},
            name="cnn_wet_dry",
        )

    result.attrs.update(
        {
            "long_name": "CNN wet/dry flag",
            "flag_values": np.asarray([-1, 0, 1], dtype=np.int8),
            "flag_meanings": "unclassifiable dry wet",
            "cnn_threshold": float(threshold),
            "training_region": "NL",
            "scenario": "PMIN_CPP",
            "sequence_length": DEFAULT_SEQUENCE_LENGTH,
            "cadence": "15min",
            "feature_order": ",".join(DEFAULT_FEATURES),
        }
    )
    return result
