# Peg Insertion Analysis with SAM 3

Automatic, prompt-driven analysis of HDF5 peg-insertion demonstrations. The project uses SAM 3 to segment and track a peg and, when configured, the hand and holder. It supports one or two cameras, extracts planar insertion and slippage metrics, renders photographic task traces, and compares outcomes across runs, methods, and peg tolerances.

> [!IMPORTANT]
> This is a 2D image-plane analysis pipeline. It does not perform intrinsic or extrinsic camera calibration, stereo reconstruction, or 3D sensor fusion. Millimetre values rely on per-camera planar scale estimates and may be biased by perspective or foreshortening. Retain pixel measurements when those assumptions are weak.

## Features

- Text-prompted SAM 3 segmentation and bidirectional video propagation
- Main-camera insertion depth and peg angular error
- Optional hand-derived axial slip in the main camera
- Optional secondary-camera lateral slip
- Optional holder segmentation for validation and visualization
- Single-camera operation with only a main-camera peg prompt
- Automatic insertion-onset detection from a single timestamp discontinuity
- Reusable compressed mask archives, including metadata migration without rerunning SAM
- Prompt checks on pre-insertion frames before full segmentation
- Per-demonstration CSV summaries and compact time series
- Photographic task-evolution traces using original peg pixels
- RGB object extraction from saved masks or directly from selected HDF5 frames
- Cross-run merging and configuration-driven method comparisons
- Manual timestamp-gap insertion for datasets that lack an onset marker
- Demo-local failure handling so malformed demonstrations do not stop an entire batch

## Measurement model

### Required input

Each complete demonstration is expected below:

```text
/demos/demo_xxxxxx/cameras/<camera_serial>/
├── rgb
└── host_timestamp_ns
```

The `rgb` and `host_timestamp_ns` datasets must have matching frame counts. Demonstrations whose `complete` attribute is false are ignored.

### Insertion onset

For each configured camera, the pipeline expects exactly one unusually large positive step in `host_timestamp_ns`:

- `pre_gap_frame` is the frame immediately before the discontinuity.
- `insertion_start_frame` is the first frame after it.
- Frames before `insertion_start_frame` form the pre-insertion phase.
- Frames from `insertion_start_frame` onward form the insertion phase.

Gap detection uses the median positive timestamp step and its median absolute deviation, controlled by `onset.gap_mad_multiplier` and `onset.gap_nominal_multiplier`.

### Metrics

| Metric | Camera and definition |
|---|---|
| Insertion depth | Main camera. Reduction in the peg's visible principal-axis length relative to the median of the last configured valid pre-insertion frames. |
| Axial slip | Main camera, when a hand prompt is configured. Change from the pre-insertion baseline in peg length above the hand reference, measured along the peg axis. |
| Lateral slip | Secondary camera, when a hand prompt is configured. Change from the pre-insertion baseline in the Euclidean distance between the selected bottom hand-row point and the top-left corner of the lower-peg bounding box. |
| Peg angle | Signed image-plane angle of the peg principal axis relative to downward image vertical. |
| Angular error | Absolute peg angle. Initial error is measured at insertion start; final error is measured at the frame of maximum insertion depth. |

The hand reference is derived from the bottommost occupied hand-mask row. For the lateral metric, disconnected row segments are separated, clipped to the peg's full horizontal bounding box, and the eligible segment closest to the lower-peg corner is selected.

### Scale estimation

Scale is estimated independently for each configured camera from the first valid peg mask at or before `pre_gap_frame`:

- Main camera: `peg_length_mm / visible_length_px`
- Secondary camera: `peg_width_mm / visible_width_px`

Frame `0` is preferred. If its peg mask is invalid, the analyzer warns and uses the first valid pre-gap frame. Missing physical dimensions do not prevent pixel analysis, but the corresponding millimetre values will be unavailable.

## Requirements

- Linux
- Conda, Miniforge, or another compatible environment manager
- Git
- Python 3.12
- An NVIDIA CUDA-capable GPU with a compatible driver
- Access to the official gated SAM 3 checkpoint

The project metadata is versioned as `0.8.0`. The supplied compatibility files pin NumPy 1.26 and Setuptools 80.9.0. The Setuptools pin is required because the current SAM 3 runtime path imports `pkg_resources`.

## Installation

### 1. Create the environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate peg-sam3
```

### 2. Install a CUDA-enabled PyTorch build

Install a build compatible with your NVIDIA driver. For example, the original project configuration uses the CUDA 12.8 wheel index:

```bash
python -m pip install torch==2.10.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu128
```

If your system requires another CUDA build, use the corresponding PyTorch wheel index instead.

### 3. Install SAM 3

```bash
git clone https://github.com/facebookresearch/sam3.git "$HOME/sam3"
python -m pip install --no-deps -e "$HOME/sam3"
```

The `--no-deps` option keeps this project's environment and constraints authoritative.

### 4. Install this project

```bash
python -m pip install -e .
```

If local SAM assets are required, set `sam3.checkpoint_path` and `sam3.bpe_path` in `config.yaml`. Complete any checkpoint-access and authentication steps required by the checkpoint provider before running segmentation.

### Verify the runtime

```bash
python - <<'PY'
import einops
import psutil
import pycocotools
import torch
from sam3.model_builder import build_sam3_video_predictor

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("SAM 3 runtime imports: OK")
PY
```

### Repair an existing environment

```bash
conda activate peg-sam3
python -m pip install --upgrade pip
python -m pip install --force-reinstall "setuptools==80.9.0"
python -m pip install -c constraints.txt -r requirements-sam3-runtime.txt
python -m pip install --no-deps -e "$HOME/sam3"
python -m pip install -e .
```

## Configuration

Create a local configuration:

```bash
cp config.example.yaml config.yaml
```

`config.yaml` is ignored by Git because it commonly contains camera serial numbers and machine-specific checkpoint paths.

A minimal single-camera configuration is:

```yaml
main_camera_serial: "MAIN_SERIAL"
secondary_camera_serial: null

sam3:
  gpus: [0]
  checkpoint_path: null
  bpe_path: null
  output_probability_threshold: 0.45
  offload_video_to_cpu: true
  offload_state_to_cpu: true
  jpeg_quality: 95

text_prompts:
  main:
    peg: "the insertion peg"

dimensions:
  peg_length_mm: 75.0
  peg_width_mm: null

onset:
  pre_window_frames: 5
  gap_mad_multiplier: 8.0
  gap_nominal_multiplier: 3.0
```

For full two-camera slippage analysis, configure both cameras and their hand prompts:

```yaml
main_camera_serial: "MAIN_SERIAL"
secondary_camera_serial: "SECONDARY_SERIAL"

text_prompts:
  main:
    peg: "the insertion peg"
    holder: "the peg holder"            # optional
    hand: "the hand holding the peg"    # enables axial slip
  secondary:
    peg: "the insertion peg"
    holder: "the peg holder"            # optional
    hand: "the hand holding the peg"    # enables lateral slip

dimensions:
  peg_length_mm: 75.0
  peg_width_mm: 17.5
```

Only `main_camera_serial` and `text_prompts.main.peg` are always required. The holder prompt is optional because current metrics do not depend on the holder. Hand prompts are independently optional. If a hand prompt is absent, the associated slippage metric is left missing rather than causing the demonstration to fail.

## Dataset and output paths

Commands that read the HDF5 dataset directly require `--dataset-path`:

- `inspect`
- `check-prompts`
- `segment`
- direct-dataset mode of `extract-pixels`
- `export-objects`
- `insert-gap`

For `inspect`, `check-prompts`, and `segment`, `--output-dir` is optional. When omitted, a sibling directory named after the dataset without its extension is used. For example:

```text
/data/session_01.h5  ->  /data/session_01/
```

Segmentation metadata records the resolved dataset path. Commands that consume saved analysis artifacts use `--input-dir`:

- `analyze`
- `plot`
- `render-demo`
- saved-mask mode of `extract-pixels`

## Recommended workflow

### 1. Inspect timestamp gaps

```bash
peg-analysis -c config.yaml inspect \
  --dataset-path /path/to/dataset.h5
```

Every configured camera in every complete demonstration should report exactly one gap.

### 2. Check prompts

```bash
peg-analysis -c config.yaml check-prompts \
  --dataset-path /path/to/dataset.h5 \
  --max-demos 5
```

Results are written to `<output_dir>/prompt_check/`. Review the side-by-side overlays and `report.json`. A match means SAM 3 returned an object, not necessarily the intended object.

### 3. Segment and track

```bash
peg-analysis -c config.yaml segment \
  --dataset-path /path/to/dataset.h5
```

Useful options:

```bash
# Process one complete demonstration
peg-analysis -c config.yaml segment \
  --dataset-path /path/to/dataset.h5 \
  --demo demo_000002

# Replace existing masks
peg-analysis -c config.yaml segment \
  --dataset-path /path/to/dataset.h5 \
  --overwrite
```

Without `--overwrite`, existing mask archives are reused. Legacy metadata is supplemented with the resolved dataset path and missing onset fields without rerunning SAM.

### 4. Analyze saved masks

```bash
peg-analysis -c config.yaml analyze \
  --input-dir /path/to/dataset
```

To analyze one demonstration:

```bash
peg-analysis -c config.yaml analyze \
  --input-dir /path/to/dataset \
  --demo demo_000002
```

`analyze` reads the source HDF5 path from mask metadata. If older metadata lacks `dataset_path`, run `segment` again without `--overwrite` to repair it.

Segmentation and analysis treat demonstration-specific errors as local failures, issue warnings, and continue with later demonstrations. Analysis records failures in `skipped_demos.csv`.

### 5. Plot outcomes

```bash
peg-analysis plot --input-dir /path/to/dataset
```

Mark trials below a maximum insertion-depth threshold with `x` markers:

```bash
peg-analysis plot \
  --input-dir /path/to/dataset \
  --insertion-depth-threshold 20
```

## Additional commands

### Render a photographic task trace

```bash
peg-analysis -c config.yaml render-demo \
  --input-dir /path/to/dataset \
  --demo demo_000002
```

The last pre-insertion RGB frame is used as the background. Saved peg masks act only as alpha mattes for original RGB peg pixels sampled from insertion onset to maximum measured depth. No artificial mask colors, contours, or heatmaps are drawn.

Options include:

```bash
peg-analysis -c config.yaml render-demo \
  --input-dir /path/to/dataset \
  --demo demo_000002 \
  --role main \
  --traces 20 \
  --alpha 0.7 \
  --no-crop \
  -o task_trace.png
```

The default output is `<input_dir>/renders/<demo>_task_trace.png`.

### Extract prompted RGB pixels

Use saved masks without rerunning SAM:

```bash
peg-analysis -c config.yaml extract-pixels \
  --input-dir /path/to/dataset \
  --demo demo_000002 \
  --frames 100 250 400
```

Or prompt only the requested HDF5 frames directly:

```bash
peg-analysis -c config.yaml extract-pixels \
  --dataset-path /path/to/dataset.h5 \
  --demo demo_000002 \
  --frames 100 250 400 \
  -o /path/to/pixels
```

If `--frames` is omitted, frame `0`, the last pre-gap frame, and the final frame are selected independently for each configured camera. If `--demo` is omitted, every complete demonstration is processed.

For each selected camera frame, the command writes one transparent RGBA PNG containing the union of all configured object masks while preserving the original RGB pixels. `manifest.json` records the output path, frame index, total union pixel count, and per-object pixel counts.

### Export arbitrary objects from key frames

Prompts are supplied on the command line and are not read from `text_prompts` in the YAML:

```bash
peg-analysis -c config.yaml export-objects \
  --dataset-path /path/to/dataset.h5 \
  --prompts peg="the red cylinder" holder="the white cylinder"
```

The command processes every complete demonstration and every camera by default. Restrict it with `--demo` and repeated `--camera-serial` options:

```bash
peg-analysis -c config.yaml export-objects \
  --dataset-path /path/to/dataset.h5 \
  --prompts peg="the red cylinder" \
  --demo 2 \
  --camera-serial MAIN_SERIAL \
  -o /path/to/object_export
```

Each prompt is run independently on the first, last pre-gap, and final frame. One transparent PNG per camera key frame contains the union of matched objects. `manifest.json` records prompts, match status, SAM metadata, per-object pixel counts, and union pixel counts.

### Merge multiple analyzed runs

```bash
peg-analysis merge-plot run_a run_b \
  -o merged_plots \
  --insertion-depth-threshold 20
```


### Compare methods across tolerances

Copy the supplied specification:

```bash
cp cumulative.example.json cumulative.json
peg-analysis cumulative-plot \
  --spec cumulative.json \
  -o cumulative_results
```

Each dataset entry links one method and tolerance to one or more analyzed directories:

```json
{
  "method": "method_a",
  "tolerance": 0.5,
  "data_dirs": [
    "/data/method_a_tol_0_5_part_1",
    "/data/method_a_tol_0_5_part_2"
  ]
}
```

Relative paths are resolved from the JSON specification directory. Important top-level options include:

```json
{
  "insertion_depth_threshold": 20.0,
  "normalization_type": "zscore",
  "normalization_scope": "group",
  "linear_analysis": {
    "enabled": true,
    "show_fit": true,
    "show_statistics": true,
    "confidence_band": false,
    "alpha": 0.05,
    "line_width": 1.8
  }
}
```

Supported angle normalizations are:

- `none`: raw initial angular error in degrees
- `max`: divide by the maximum absolute initial angle
- `minmax`: map the minimum and maximum to 0 and 1
- `zscore`: subtract the mean and divide by the population standard deviation

Supported normalization scopes are:

- `group`: each method and tolerance pair independently
- `tolerance`: pool methods within each tolerance
- `global`: pool all selected trials


Linear analysis is performed separately for each method/tolerance group and for both scatter outcomes: maximum insertion depth and maximum absolute axial slip versus initial angular error. Groups with fewer than three finite pairs or constant inputs or outputs are reported as unavailable rather than assigned misleading statistics.

Cumulative outputs include:

```text
cumulative_results/
├── cumulative_outcomes.png
├── initial_angle_vs_outcomes_by_tolerance.png
├── cumulative_trials.csv
├── plotted_trials.csv
└── linear_correlations.csv
```

### Insert a timestamp gap

Create a copy with a gap inserted before a selected frame:

```bash
peg-analysis insert-gap \
  --dataset-path /path/to/dataset.h5 \
  --demo 2 \
  --frame 300 \
  --gap-ms 500 \
  -o /path/to/dataset_with_gap.h5
```

By default, all cameras in the demonstration are shifted from `--frame` onward to preserve synchronization. Use `--camera-serial` to target one camera, or `--in-place` to modify the source dataset directly. In-place editing cannot be combined with `--output-path`.

> [!CAUTION]
> Prefer the default copy behavior. Use `--in-place` only when you have a separate backup of the source HDF5 file.

## Outputs

After segmentation and analysis:

```text
<output_dir>/
├── masks/
│   ├── demo_xxxxxx_main.npz
│   ├── demo_xxxxxx_main_meta.json
│   ├── demo_xxxxxx_secondary.npz
│   └── demo_xxxxxx_secondary_meta.json
├── demo_xxxxxx_timeseries.csv
├── summary.csv
├── skipped_demos.csv          # only when analysis skips demos
├── prompt_check/              # when check-prompts is run
├── plots/                     # when plot is run
├── renders/                   # when render-demo is run
└── pixels/                    # default saved-mask extraction destination
```

`summary.csv` contains one row per successfully analyzed demonstration, including:

- maximum insertion depth
- frame of maximum depth
- maximum absolute axial and lateral slip, when available
- initial and final angular error
- angular-error change
- main and secondary validity fractions

Each `demo_xxxxxx_timeseries.csv` is intentionally compact and contains frame metadata, validity flags, final pixel metrics, and angular measurements. Per-frame millimetre conversions and intermediate geometry helpers are not exported.

Mask metadata includes the source dataset path, demonstration, camera role and serial, frame count, pre-gap and insertion-start indices, timestamp gap, prompts, and selected SAM object IDs.

## Validation checklist

Before trusting the reported measurements:

- Inspect prompt-check overlays for every configured object and camera.
- Verify that the selected SAM instance is the intended object, not merely a valid match.
- Inspect the scale frame, especially when frame `0` is invalid and fallback is used.
- Confirm that the peg dimension configured for each camera matches the visible dimension used for scale.
- Inspect masks near the pre-gap frame, insertion start, contact, maximum slip, and maximum depth.
- Check for perspective, foreshortening, occlusion, and out-of-plane motion.
- Confirm that the hand mask's bottom row is a stable anatomical or task reference throughout the sequence.
- Prefer pixel outputs when the planar millimetre conversion is unreliable.
- Review `skipped_demos.csv` and validity fractions rather than treating missing values as zeros.

## Tests

Run the unit test suite from the repository root:

```bash
PYTHONPATH=. pytest -q
```

The tests cover timestamp-gap handling, configuration paths, scale fallback, optional camera and hand inputs, geometry, plotting, cumulative normalization and regression, extraction, rendering, and failure resilience.

## Troubleshooting

### `pkg_resources` import error

```bash
python -m pip install --force-reinstall "setuptools==80.9.0"
python -m pip install -c constraints.txt -r requirements-sam3-runtime.txt
```

### Missing `einops`, `psutil`, or `pycocotools`

```bash
python -m pip install -c constraints.txt -r requirements-sam3-runtime.txt
```

### PyTorch cannot see CUDA

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is false, install a PyTorch wheel compatible with the installed NVIDIA driver and CUDA support. The verification script reports the detected PyTorch and CUDA state.


## Cumulative method comparison

Use a JSON specification to compare multiple methods across tolerance levels. Set top-level `font_name` and `font_size` values to control the Matplotlib font family and base font size for all cumulative plots. Each figure row is one tolerance. The three columns show insertion depth, axial slip, and lateral slip using the existing distribution style: method-colored mean bars, 95% confidence intervals, individual trial markers, and short trial labels. The complete figure uses one shared y-axis range so all outcomes and tolerance rows are directly comparable.

Copy and edit the included example:
If `torch.cuda.is_available()` is false, install a PyTorch wheel compatible with the installed NVIDIA driver and required CUDA runtime.

### No or multiple timestamp gaps

Inspect the dataset:

```bash
peg-analysis -c config.yaml inspect --dataset-path /path/to/dataset.h5
```

The pipeline requires one detected gap per configured camera. If the dataset legitimately lacks the pause/resume discontinuity, use `insert-gap` on a copy at the known insertion-start frame.

### Existing masks cannot be analyzed

If the error says mask metadata does not contain `dataset_path`, repair metadata without rerunning SAM:

```bash
peg-analysis -c config.yaml segment \
  --dataset-path /path/to/dataset.h5
```

Do not add `--overwrite` for metadata-only migration.

### Missing millimetre outputs

Set the relevant physical dimension:

- `dimensions.peg_length_mm` for main-camera millimetre outputs
- `dimensions.peg_width_mm` for secondary-camera millimetre outputs

Pixel outputs remain available when a physical dimension is absent.

## Repository layout

```text
.
├── peg_analysis/
│   ├── analyze.py
│   ├── cli.py
│   ├── core.py
│   ├── cumulative.py
│   ├── export_objects.py
│   ├── extract_pixels.py
│   ├── plots.py
│   ├── prompt_check.py
│   ├── render.py
│   ├── sam3_runner.py
│   ├── segment.py
│   └── timestamp_gap.py
├── tests/
├── config.example.yaml
├── cumulative.example.json
├── environment.yml
├── constraints.txt
├── requirements-sam3-runtime.txt
├── pyproject.toml
└── README.md
```
```json
"normalization_type": "zscore",
"normalization_scope": "group"
```

`normalization_scope` controls which selected trials provide the normalization statistics: `"group"` uses each method/tolerance pair independently, `"tolerance"` pools methods within each tolerance, and `"global"` pools all methods and tolerances. The default is `"group"`.

Supported normalization values are:

- `"none"`: raw initial angular error in degrees.
- `"max"`: divide by the maximum absolute initial angle within each method and tolerance pair.
- `"minmax"`: map the minimum and maximum initial angles within each method and tolerance pair to 0 and 1.
- `"zscore"`: subtract the group mean and divide by the population standard deviation within each method and tolerance pair.



Configure per-group linear analysis for both angle-outcome scatter rows:

```json
"linear_analysis": {
  "enabled": true,
  "show_fit": true,
  "show_statistics": true,
  "confidence_band": false,
  "alpha": 0.05,
  "line_width": 1.8
}
```

For every method/tolerance group, the scatter figure draws an optional least-squares fit. Set `line_width` to control the fitted-line thickness. It reports `n`, Pearson's `r`, the two-sided p-value, and `R²`. The command also writes `linear_correlations.csv` with the slope, intercept, standard errors, normalization settings, and a cautious interpretation. Groups with fewer than three finite pairs or constant input/output values are recorded as unavailable rather than assigned misleading statistics. Set `linear_analysis` to `false` to hide fits and annotations while still exporting the numerical analysis.

Set the optional top-level failure threshold in millimetres:

```json
"insertion_depth_threshold": 20.0
```

A trial whose `max_insertion_depth_mm` is below this threshold is drawn with an `x` marker in every outcome panel. Trials meeting the threshold retain circular markers. Omit the setting or use `null` to disable failure markers. Within each tolerance row, only methods that actually have data are shown on the x-axis; if one method is present, that row has one method label.

## Optional secondary camera and hand tracking

Only `main_camera_serial` and `text_prompts.main.peg` are required. To run on a
single-camera dataset, omit `secondary_camera_serial` (or set it to `null`) and
the `text_prompts.secondary` section may also be omitted.

The `hand` prompt is optional independently for each configured camera. When it
is absent, segmentation and analysis still produce peg-based insertion depth
and angular metrics. Hand-derived outputs are left missing rather than causing
the demonstration to fail:

- main hand absent: axial-slip values and validity are missing/false;
- secondary hand absent: lateral-slip values and validity are missing/false;
- secondary camera absent: all secondary-camera and lateral-slip summary values
  are missing, while main-camera analysis continues normally.

The `holder` prompt is also optional because the current metrics do not depend
on it. Existing configurations that provide both cameras and all three prompts
retain their previous behavior.

### Render cumulative trial video galleries

After `cumulative-plot`, render exactly the plotted physical trials as one MP4 per tolerance:

```bash
peg-analysis cumulative-gallery \
  --spec cumulative.json \
  -o cumulative_results
```

The command reads `cumulative_results/plotted_trials.csv` by default.
Gallery layout is method-grouped. With multiple methods, every method occupies a 5-row by 2-column block; when a tolerance contains only one method, that group is rotated to 2 rows by 5 columns. Videos begin at shallow engagement and run to each trial's final recorded frame. A trial's `highlight` value in its original `summary.csv` is interpreted as the global frame index at which its success/failure border first appears. Missing `highlight` values, or values earlier than shallow engagement, are replaced by the trial's final frame. The border remains visible thereafter: green when `max_insertion_depth_mm >= insertion_depth_threshold`, red otherwise.

Optional JSON settings are:

```json
"video_gallery": {
  "shallow_engagement_depth_mm": 2.0,
  "fps": 30.0,
  "cell_width": 320,
  "cell_height": 240,
  "trials_per_method": 10,
  "seed": 0,
  "truncate_at_max_depth": false,
  "group_by_tolerance": true,
  "highlight_early_seconds": 0.0
}
```

`shallow_engagement_depth_mm` defaults to 2 mm. The frame is inferred from each compact main-camera time series using the scale implied by the recorded maximum-depth frame and `max_insertion_depth_mm`; if that cannot be recovered, insertion onset is used as the conservative fallback. A per-dataset/method gallery is always rendered for every configured dataset entry. With the supplied six dataset entries this means six separate method videos, each using the 2-row by 5-column layout. `video_gallery.group_by_tolerance` is additive: when `true`, the renderer also creates aggregate tolerance videos that combine all methods available at each tolerance; when `false`, only the six per-dataset/method videos are produced. The default is `true`.

Set `video_gallery.highlight_early_seconds` to a non-negative duration to make a valid border appear that many seconds before its `highlight` frame. The shift uses the original HDF5 `host_timestamp_ns`, rather than the MP4 playback FPS, so the requested duration is measured in actual recorded time and is clamped to shallow engagement. The gallery always refreshes `highlight` from each trial's original `source_summary`/`summary.csv`, so an older `plotted_trials.csv` does not silently lose later manual highlight edits. Missing, pre-engagement, or later-than-displayed `highlight` values first resolve to the final displayed source frame. `highlight_early_seconds` is then applied to that fallback using the original HDF5 `host_timestamp_ns`, so the fallback border also appears the requested host-clock duration before the end.

Set `video_gallery.truncate_at_max_depth` to `true` to end each trial at its `max_depth_frame` instead of the final recorded frame. The default is `false`. When truncation is enabled, `highlight` is still interpreted globally but is clamped to `max_depth_frame`; therefore a missing, pre-engagement, or later-than-maximum-depth highlight appears on that trial's final displayed frame. Shorter trials hold their final displayed frame while longer trials finish, so the grid remains synchronized and stable.

Gallery success classification can also reject implausibly large insertion depths. For a standalone directory gallery use `--anomalous-depth MM`; a trial at or above that maximum insertion depth is colored as a failure even if it passes `--insertion-depth-threshold`. For cumulative galleries, set `video_gallery.anomalous_depth_mm` to the same upper failure boundary.

Both gallery workflows support first/last selection. The standalone CLI uses mutually exclusive `--first-n N` and `--last-n N`. The cumulative spec accepts mutually exclusive `video_gallery.first_n` and `video_gallery.last_n`; selection is applied in plotted-trial CSV order within each method before any deterministic filling to `trials_per_method`. Use `null` for both to disable selection.


Both gallery CLIs can use the same JSON spec. For a standalone directory gallery, pass `--spec cumulative.json`; values under `video_gallery` become its defaults, including `fps`, `shallow_engagement_depth_mm`, `highlight_early_seconds`, `truncate_at_max_depth`, `anomalous_depth_mm`, `first_n`, `last_n`, `cell_width`, and `cell_height`. The standalone gallery also uses the spec's top-level `insertion_depth_threshold` unless `--insertion-depth-threshold` overrides it. Explicit standalone CLI options always take precedence over the spec. For boolean max-depth truncation, use `--truncate-at-max-depth` or `--no-truncate-at-max-depth` to explicitly override the spec.

```bash
peg-analysis gallery --input-dir /path/to/analyzed_dataset --spec cumulative.json
peg-analysis gallery --input-dir /path/to/analyzed_dataset --spec cumulative.json --last-n 5 --anomalous-depth 35
peg-analysis gallery --input-dir /path/to/analyzed_dataset --tolerance 1.0
```

Cumulative gallery videos label their tolerance in the header. The standalone `gallery` command also labels the tolerance when `--tolerance MM` is supplied.

For gallery timing, set both `video_gallery.dataset_fps` (the known recording FPS) and `video_gallery.fps` (the exported MP4 FPS). The renderer uses `dataset_fps` to map recorded frames into export frames without changing the recording's real-time duration. Every exported gallery MP4 contains its own progress strip at the bottom with a completion bar plus `Elapsed` and `Remaining` playback time. The standalone `gallery` command reads both values from the same spec and may override them with `--dataset-fps` and `--fps`.
