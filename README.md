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

- `dataset_path`
- `output_dir`
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

## Commands

Inspect the dataset and verify that every complete demonstration has exactly one timestamp gap per camera:

```bash
peg-analysis -c config.yaml inspect
```

Run segmentation, tracking, geometry extraction, and CSV export:

```bash
peg-analysis -c config.yaml analyze
```

Quickly test all six object prompts on `t0` frames without propagating masks through the videos:

```bash
peg-analysis -c config.yaml check-prompts
```

By default, this checks the first complete demonstration. Sample more demonstrations with:

```bash
peg-analysis -c config.yaml check-prompts --max-demos 5
```

The command writes side-by-side mask overlays and a machine-readable report to `<output_dir>/prompt_check/`. Review these results before running the full analysis. A returned match only means that SAM 3 found an object, so visually confirm that each mask covers the intended peg, holder, or hand.

The CLI supports `inspect`, `check-prompts`, and `analyze`.

## Metrics

For peg centroid `(x_p, y_p)`, fixed holder centroid `(x_h, y_h)`, and hand lower-left bounding-box reference `(x_g, y_g)`:

```text
peg_u = x_p - x_h
peg_v = y_p - y_h

lateral_slip = [(x_p - x_g) at t] - median_pre(x_p - x_g)
axial_slip   = [(y_p - y_g) at t] - median_pre(y_p - y_g)
insertion_depth = peg_v(t) - median_pre(peg_v)
```

Recommended fields:

- `secondary / lateral_slip_px` for lateral failure analysis
- `main / axial_slip_px` for main-view slippage
- `main / insertion_depth_px` for insertion depth
- corresponding `_mm` fields for approximate view-specific physical units

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
