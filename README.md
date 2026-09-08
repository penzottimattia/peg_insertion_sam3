# Peg Insertion Analysis with SAM 3

Automatic two-camera geometry analysis for HDF5 peg-insertion demonstrations. The pipeline uses SAM 3 text prompts to segment and track the peg, holder, and hand without manual clicks.

## What the project measures

- **Main camera:** insertion depth and axial peg-to-hand slippage
- **Secondary camera:** lateral peg-to-hand slippage
- **Both cameras:** peg position relative to a fixed holder origin and an independent approximate mm/px scale

The analysis is planar. It does not perform camera calibration, stereo reconstruction, or 3D fusion.

## Geometry assumptions

- `t0` is the last frame before the single pause/resume timestamp gap.
- Insertion begins on the first frame after that gap.
- Image `+y` is the insertion axis and image `+x` is the lateral axis.
- The holder-mask centroid at `t0` is the fixed origin.
- The holder and cameras remain fixed during each demonstration.
- Scale is estimated separately for each view from the peg mask in frame `0`, where the complete peg should be visible.

## Repository layout

```text
.
├── peg_analysis/              Python package, CLI, and prompt checker
├── tests/                     Unit tests
├── config.example.yaml        Configuration template
├── environment.yml            Conda environment
├── constraints.txt            Compatibility pins
├── requirements-sam3-runtime.txt
└── pyproject.toml
```

## Requirements

- Linux
- Conda or Miniforge
- Git
- An NVIDIA CUDA-capable GPU and compatible driver
- Access to the official gated SAM 3 checkpoint
- Python 3.12, installed by the environment file

The default installer uses the CUDA 12.8 PyTorch wheel index. Override `PYTORCH_INDEX_URL` when your system needs another build.

## Installation

Create and activate the Conda environment from the project root:

```bash
conda env create -f environment.yml
conda activate peg-sam3
```

Install a CUDA-enabled PyTorch build compatible with your NVIDIA driver. The following command uses the CUDA 12.8 wheel index supplied by the original project configuration:

```bash
python -m pip install torch==2.10.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu128
```

Clone and install SAM 3. The editable SAM 3 installation uses `--no-deps` so that this project’s Conda environment and constraints remain authoritative:

```bash
git clone https://github.com/facebookresearch/sam3.git "$HOME/sam3"
python -m pip install --no-deps -e "$HOME/sam3"
```

Install this project in editable mode:

```bash
python -m pip install -e .
```

If your SAM 3 setup requires authentication for a gated checkpoint, authenticate with the relevant provider and request checkpoint access before running analysis. Set `sam3.checkpoint_path` and `sam3.bpe_path` in `config.yaml` when local assets must be supplied explicitly.

### Verify the installation

```bash
python - <<'PY'
import pkg_resources
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

### Update or repair an existing environment

From the project root:

```bash
conda activate peg-sam3
python -m pip install --upgrade pip
python -m pip install --force-reinstall "setuptools==80.9.0"
python -m pip install -c constraints.txt -r requirements-sam3-runtime.txt
python -m pip install --no-deps -e "$HOME/sam3"
python -m pip install -e .
```

Setuptools is pinned because the current SAM 3 runtime imports `pkg_resources`. The runtime requirements also provide `einops`, `psutil`, and `pycocotools`.

## Configuration

Create a local configuration file:

```bash
cp config.example.yaml config.yaml
```

Set:

- both camera serials
- text prompts for peg, holder, and hand
- at least one physical peg dimension

Example:

```yaml
dimensions:
  peg_length_mm: 75.0
  peg_width_mm: 17.5
```

At frame `0`, OpenCV `minAreaRect` is fitted to the peg mask. The longer rectangle side maps to `peg_length_mm`, and the shorter side maps to `peg_width_mm`. If both dimensions are supplied, the median of the two mm/px estimates is used. Perspective and foreshortening can bias this planar scale, so pixel outputs remain the authoritative measurements when perspective error is significant.

`config.yaml` is intentionally excluded from the distributable archive because it commonly contains machine-specific paths and camera serial numbers.

## Dataset and output paths

The dataset path is a CLI argument and is not read from `config.yaml`. Dataset-reading commands (`inspect`, `check-prompts`, and `segment`) require `--dataset-path`. Segmentation records the resolved dataset path in its mask metadata. Downstream commands (`analyze` and `plot`) consume the resulting analysis directory through `--input-dir`. Use `--output-dir` with dataset-reading commands to choose the output location. If it is omitted, the output directory is created next to the dataset and uses the dataset filename without its extension.

For example, `/data/session_01.h5` defaults to `/data/session_01/`.

## Commands

Inspect the dataset and verify that every complete demonstration has exactly one timestamp gap per camera:

```bash
peg-analysis -c config.yaml inspect --dataset-path /path/to/dataset.h5
```

Run segmentation, tracking, geometry extraction, and CSV export:

```bash
peg-analysis -c config.yaml analyze --input-dir /path/to/dataset
```

For legacy outputs whose mask metadata predates `dataset_path`, migrate the existing metadata without rerunning SAM. Existing mask archives are reused and only their JSON metadata is updated:

```bash
peg-analysis -c config.yaml segment --dataset-path /path/to/dataset.h5
```

Create figures from the generated summary:

```bash
peg-analysis plot --input-dir /path/to/dataset
```

Render a single photographic task-evolution trace for one demonstration. The last pre-insertion image is used as the background, while saved SAM masks are used only as alpha mattes to extract real RGB peg pixels from sampled insertion frames:

```bash
peg-analysis -c config.yaml render-demo \
  --input-dir /path/to/dataset \
  --demo demo_000002
```

By default, 12 peg appearances are overlaid for both cameras and the figure is cropped around the task. Use `--role main`, `--role secondary`, `--traces 20`, `--alpha 0.7`, `--no-crop`, or `-o figure.png` to customize the output. The default file is `<input-dir>/renders/<demo>_task_trace.png`. No masks, heatmaps, or artificial contour colours are drawn.

Draw trials below an insertion depth threshold with `x` markers instead of circles:

```bash
peg-analysis plot --input-dir /path/to/dataset --insertion-depth-threshold 20
```

Analysis treats malformed demonstrations as demo-local failures. It emits a warning, records the failure in `skipped_demos.csv`, and continues with later demonstrations.

Quickly test all six object prompts on `t0` frames without propagating masks through the videos:

```bash
peg-analysis -c config.yaml check-prompts --dataset-path /path/to/dataset.h5
```

By default, this checks the first complete demonstration. Sample more demonstrations with:

```bash
peg-analysis -c config.yaml check-prompts --dataset-path /path/to/dataset.h5 --max-demos 5
```

The command writes side-by-side mask overlays and a machine-readable report to `<output_dir>/prompt_check/`. Review these results before running the full analysis. A returned match only means that SAM 3 found an object, so visually confirm that each mask covers the intended peg, holder, or hand.

Merge summaries and optionally plot only the trials with the greatest maximum insertion depth:

```bash
peg-analysis merge-plot run_a run_b --nmax-trials 10
```

The same insertion depth threshold can be applied to merged plots:

```bash
peg-analysis merge-plot run_a run_b --insertion-depth-threshold 20
```

`merged_summary.csv` always retains every trial. `plotted_summary.csv`, the figures, and `trial_key.csv` contain only the selected trials when `--nmax-trials` is used.

The depth, axial-slip, and lateral-slip outcome panels use the same y-axis range for direct visual comparison.

The CLI supports `inspect`, `check-prompts`, `segment`, `analyze`, `plot`, `render-demo`, `merge-plot`, and `insert-gap`.

## Metrics

| Aspect | Method |
|---|---|
| Insertion onset | The first frame after the single detected timestamp gap. |
| Baseline | Median of the last five valid pre-insertion frames. |
| Insertion depth | Main-camera reduction in the peg's visible length relative to the pre-insertion baseline. |
| Axial slip | Main-camera change in peg length above the hand reference, measured along the estimated peg axis. |
| Lateral slip | Secondary-camera change from the pre-insertion baseline in the Euclidean distance between the selected thumb-tip median point and the top-left corner of the axis-aligned bounding box of peg pixels strictly below the thumb row. Disconnected hand-row segments are separated, clipped to the full peg horizontal bounding box, and the eligible segment closest to the lower-peg corner is selected. |
| Hand reference | Bottommost row of the hand mask, using the median horizontal coordinate of pixels on that row. |
| Peg geometry | Principal-axis analysis of the peg mask, with visible length and width obtained from projection extents. |
| Holder | The holder mask is validated but is not used in the current metric calculations. |
| Main-camera scale | Peg length in millimetres divided by visible peg length in frame 0. |
| Secondary-camera scale | Peg width in millimetres divided by visible peg width in frame 0. |
| Millimetre conversion | Each pixel metric is multiplied by the independently estimated scale for its camera view. |
| Peg angle | Signed image-plane angle relative to downward image vertical. |
| Angular error | Absolute value of the peg angle. |
| Initial angular metrics | Values at the insertion-start frame. |
| Final angular metrics | Values at the frame of maximum measured insertion depth. |
| Summary extrema | Maximum depth and axial/lateral slip extrema are computed over valid insertion-phase frames. |
| Pre-insertion noise | Count, sample standard deviation, and median absolute deviation over all valid pre-insertion values. |

## Outputs

```text
<output_dir>/
├── masks/
│   ├── demo_xxxxxx_main.npz
│   ├── demo_xxxxxx_secondary.npz
│   └── *_meta.json
├── demo_xxxxxx_timeseries.csv
└── summary.csv
```

The aggregate summary contains only the primary outcomes, angular-error endpoints, and camera-validity fractions. Timeseries files contain essential frame metadata and final pixel metrics only. Millimetre conversions and geometry helper quantities are not exported per frame. Analysis writes one aggregate `summary.csv`, including when `--demo` is used.

Metadata includes prompts, selected object IDs, `t0`, `scale_frame_index`, holder centroid, peg rectangle dimensions, and scale.

## Validation checklist

- Confirm that the largest text-matched instances are the intended peg, holder, and hand.
- Carefully inspect the frame `0` peg mask because it determines scale.
- Tune prompts and `output_probability_threshold` when objects are confused.
- Inspect masks around `t0`, contact, maximum slippage, and maximum depth.
- Peg-centroid depth may be biased when the holder occludes the peg.
- The hand reference assumes stable segmentation and limited hand rotation.
- Prefer pixel measurements when planar scale assumptions are weak.

## Tests

```bash
PYTHONPATH=. pytest -q
```

## Troubleshooting

### `pkg_resources` import error

The project pins `setuptools==80.9.0`, which retains `pkg_resources` for the current SAM 3 runtime path:

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

Use a JSON specification to compare multiple methods across tolerance levels. Each figure row is one tolerance. The three columns show insertion depth, axial slip, and lateral slip using the existing distribution style: method-colored mean bars, 95% confidence intervals, individual trial markers, and short trial labels. The complete figure uses one shared y-axis range so all outcomes and tolerance rows are directly comparable.

Copy and edit the included example:

```bash
cp cumulative.example.json cumulative.json
peg-analysis cumulative-plot --spec cumulative.json -o cumulative_results --nmax-trials 10
```

Each `datasets` entry identifies a method and tolerance. `data_dirs` accepts one or more analyzed-data directories containing `summary.csv`, so split datasets can be combined without manually merging files:

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

Paths may be absolute or relative to the JSON specification. The command writes `cumulative_outcomes.png`, `initial_angle_vs_outcomes_by_tolerance.png`, `cumulative_trials.csv`, and `plotted_trials.csv`. The complete CSV always retains every configured trial. The plotted CSV and figure contain only the selected trials when `nmax_trials` is used. Trial labels in the figure correspond to the `trial_label` column in these CSV files.


The angle-outcomes figure contains one column per tolerance and two rows: maximum insertion depth versus initial angular error, followed by maximum absolute axial slip versus initial angular error. It uses the same method colors and `Txx` labels as the bar plots and uses `x` markers for failures. If a failure threshold is configured, it is shown as a horizontal dashed line in every insertion-depth scatter panel and every left-hand insertion-depth bar panel. All scatter panels share both x- and y-axis limits across rows and columns for direct comparison.


Choose the scatter-plot angle normalization at the top level:

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

Normalization uses only trials selected after applying per-method, per-tolerance `nmax_trials`. For constant groups, the scale safely falls back to `1.0`, so min-max values and z-scores are zero. `plotted_trials.csv` records `angle_normalization_type`, `initial_angular_error_group_mean_deg`, `initial_angular_error_group_min_deg`, `initial_angular_error_group_max_deg`, `initial_angular_error_group_scale_deg`, and `normalized_initial_angular_error`. Override the JSON for one run with `--normalization-type none`, `max`, `minmax`, or `zscore`. The legacy `normalized_angle: true` setting is still accepted and maps to `"max"`.


Configure per-group linear analysis for both angle-outcome scatter rows:

```json
"linear_analysis": {
  "enabled": true,
  "show_fit": true,
  "show_statistics": true,
  "confidence_band": false,
  "alpha": 0.05
}
```

For every method/tolerance group, the scatter figure draws an optional least-squares fit and reports `n`, Pearson's `r`, the two-sided p-value, and `R²`. The command also writes `linear_correlations.csv` with the slope, intercept, standard errors, normalization settings, and a cautious interpretation. Groups with fewer than three finite pairs or constant input/output values are recorded as unavailable rather than assigned misleading statistics. Set `linear_analysis` to `false` to hide fits and annotations while still exporting the numerical analysis.

Set the optional maximum number of plotted trials at the top level:

```json
"nmax_trials": 10
```

For every method at every tolerance, this independently selects the N trials with the greatest `max_insertion_depth_mm`. For example, with two methods, three tolerances, and `nmax_trials: 10`, the figure can include up to 60 trials. The CLI option `--nmax-trials` overrides the JSON value for one run. Omit the setting or use `null` to plot all trials.

Set the optional top-level failure threshold in millimetres:

```json
"insertion_depth_threshold": 20.0
```

A trial whose `max_insertion_depth_mm` is below this threshold is drawn with an `x` marker in every outcome panel. Trials meeting the threshold retain circular markers. Omit the setting or use `null` to disable failure markers. Within each tolerance row, only methods that actually have data are shown on the x-axis; if one method is present, that row has one method label.
