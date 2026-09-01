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

`merged_summary.csv` always retains every trial. `plotted_summary.csv`, the figures, and `trial_key.csv` contain only the selected trials when `--nmax-trials` is used.

The depth, axial-slip, and lateral-slip outcome panels use the same y-axis range for direct visual comparison.

The CLI supports `inspect`, `check-prompts`, `segment`, `analyze`, `plot`, and `merge-plot`.

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
