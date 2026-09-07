# CNN wet/dry inference only

This is the minimal production entry point for the SEVIRI-CML **v0.13 NL/PMIN_CPP** wet/dry CNN.

It deliberately stops at the wet/dry flag. It does **not** run CML QC, interpolation, resampling, CPP extraction/download, Mode classification, RSTD, radar/gauge matching, QPE, WAA, rainfall retrieval, fusion, metrics, or plotting.

## Model contract

The current model is:

- training region: `NL`
- scenario: `PMIN_CPP`
- raw min/max signal: `rsl_min`
- sequence length: `4`
- cadence: `15min`
- CNN threshold used by the current v0.13 notebook: `0.5`
- feature order:

```text
Pmin_normalized
cot
cre
cwp
cdnc
cgt
cph
```

The four 15-minute samples form one 1-hour CNN sequence. No time mismatch is accepted when a sequence is formed.

The model directory must contain the trained artifacts from the existing v0.13 `published_v2` registry:

```text
model.keras
scalers.joblib
model_manifest.json   # when available
```

For the current registry this is normally the `TRAIN_NL/PMIN_CPP/test_01` model directory.

## Input

Pass an **already aligned 15-minute CML + CPP** `xarray.Dataset`.

Minimum variables:

```text
rsl_min
cot
cre
cwp
cdnc
cgt
cph
```

If `Pmin_normalized` is already present, it is used directly. Otherwise the CNN module constructs it from the selected raw Pmin signal using the model-required 24-hour trailing rolling median (96 samples, minimum 8 samples / 2 hours).

Typical dimensions are:

```text
time
cml_id
sublink_id   # optional; preserved when rsl_min is directional
```

CPP variables can be `(time, cml_id)` while `rsl_min` can be `(time, cml_id, sublink_id)`. Each sublink is classified independently. Sublinks are selected by coordinate label rather than positional index.

## Install CNN-only dependencies

```bash
pip install -r requirements-cnn.txt
```

## Minimal use for Inigo

```python
from pathlib import Path

import xarray as xr

from cnn_wet_dry_inference import predict_wet_dry

# This file must already contain aligned 15-min CML + CPP inputs.
ds = xr.open_dataset("/path/to/CML_CPP.nc")

model_dir = Path(
    "/path/to/data/aux/models/published_v2/"
    "TRAIN_NL/PMIN_CPP/test_01"
)

wet = predict_wet_dry(
    ds,
    model_dir=model_dir,
    signal_var="rsl_min",
    threshold=0.5,
)

# wet is the only product from this entry point.
#  1 = wet
#  0 = dry
# -1 = unclassifiable (missing CNN input, incomplete sequence, or time gap)
wet.to_netcdf("CNN_WET_DRY.nc")
```

## Function

```python
predict_wet_dry(
    ds,
    model_dir,
    signal_var="rsl_min",
    threshold=0.5,
)
```

Returns one `xarray.DataArray` named `cnn_wet_dry`.

For a directional min/max CML input the returned shape is normally:

```text
(time, cml_id, sublink_id)
```

For a non-directional signal it is:

```text
(time, cml_id)
```

## Important behavior

- The saved `scalers.joblib` is loaded and applied. Nothing is refit on the production data.
- The saved Keras `model.keras` is loaded with `compile=False`.
- Feature ordering is fixed to the model training order.
- Sequence length is fixed to four 15-minute samples for this entry point.
- A CNN sequence is skipped if its four timestamps are not exactly consecutive at 15-minute cadence.
- Missing/non-finite CNN inputs are not silently converted to dry; the corresponding output remains `-1`.
- No external wet/dry method is fused with the CNN.
