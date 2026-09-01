import argparse
import json
from pathlib import Path
from .core import config


def _add_dataset_io(parser):
    parser.add_argument(
        '--dataset-path', required=True,
        help='Input HDF5 dataset. This value is not read from config.yaml.',
    )
    parser.add_argument(
        '--output-dir',
        help='Analysis output directory. Defaults to a directory named after the dataset, next to it.',
    )


def _load_config(args):
    return config(args.config, dataset_path=args.dataset_path, output_dir=args.output_dir)


def _add_input_dir(parser, help_text):
    parser.add_argument('--input-dir', required=True, help=help_text)


def _load_analysis_config(args):
    input_dir = Path(args.input_dir).expanduser().resolve()
    masks_dir = input_dir / "masks"
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    metadata_paths = sorted(masks_dir.glob("*_meta.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No mask metadata found in: {masks_dir}")

    dataset_paths = set()
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        dataset_path = metadata.get("dataset_path")
        if dataset_path:
            dataset_paths.add(str(Path(dataset_path).expanduser().resolve()))

    if not dataset_paths:
        raise ValueError(
            "Mask metadata does not contain dataset_path. Run segment again without "
            "--overwrite to repair existing metadata without rerunning SAM."
        )
    if len(dataset_paths) != 1:
        raise ValueError(
            "Input directory references multiple datasets: "
            + ", ".join(sorted(dataset_paths))
        )

    return config(
        args.config,
        dataset_path=next(iter(dataset_paths)),
        output_dir=input_dir,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', default='config.yaml',
                   help='YAML containing camera, SAM, dimension, onset, and visualization settings')
    sub = p.add_subparsers(dest='command', required=True)

    inspect = sub.add_parser('inspect')
    _add_dataset_io(inspect)

    check = sub.add_parser('check-prompts')
    _add_dataset_io(check)
    check.add_argument('--max-demos', type=int, default=1)

    seg = sub.add_parser('segment', help='Run SAM and save masks')
    _add_dataset_io(seg)
    seg.add_argument('--demo')
    seg.add_argument('--overwrite', action='store_true')

    ana = sub.add_parser('analyze', help='Compute metrics from saved masks')
    _add_input_dir(
        ana,
        'Analysis directory containing masks and segmentation metadata',
    )
    ana.add_argument('--demo')

    plot = sub.add_parser('plot', help='Create figures from a summary CSV')
    _add_input_dir(plot, 'Analysis directory containing summary.csv')
    plot.add_argument('--summary', help='Optional summary CSV; defaults to <input-dir>/summary.csv')

    merge = sub.add_parser(
        'merge-plot',
        help='Merge and plot summary.csv files from multiple output directories; no config or dataset required',
    )
    merge.add_argument('output_dirs', nargs='+', help='Analysis output directories containing summary.csv')
    merge.add_argument('-o', '--output-dir', default='merged_plots',
                       help='Destination directory (default: ./merged_plots)')
    merge.add_argument(
        '--nmax-trials', type=int,
        help='Plot only the N trials with the greatest maximum insertion depth',
    )

    a = p.parse_args()

    if a.command == 'merge-plot':
        if len(a.output_dirs) < 2:
            merge.error('provide at least two output directories')
        if a.nmax_trials is not None and a.nmax_trials < 1:
            merge.error('--nmax-trials must be at least 1')
        from .plots import merge_plots
        merge_plots(a.output_dirs, a.output_dir, a.nmax_trials)
        return

    if a.command == 'analyze':
        c = _load_analysis_config(a)
    elif a.command == 'plot':
        c = {'output_dir': str(Path(a.input_dir).expanduser().resolve())}
    else:
        c = _load_config(a)

    if a.command == 'inspect':
        import h5py
        from .core import demos, group, detect_gap
        with h5py.File(c['dataset_path'], 'r') as h:
            for d in demos(h):
                print(d)
                for role, serial in [('main', c['main_camera_serial']), ('secondary', c['secondary_camera_serial'])]:
                    g = group(h, d, serial)
                    print(role, len(g['rgb']), detect_gap(
                        g['host_timestamp_ns'][:], c['onset']['gap_mad_multiplier'],
                        c['onset']['gap_nominal_multiplier']))
    elif a.command == 'check-prompts':
        if a.max_demos < 1:
            p.error('--max-demos must be at least 1')
        from .prompt_check import check_prompts
        check_prompts(c, a.max_demos)
    elif a.command == 'segment':
        from .segment import segment
        segment(c, a.demo, a.overwrite)
    elif a.command == 'analyze':
        from .analyze import analyze
        analyze(c, a.demo)
    else:
        from .plots import make_plots
        make_plots(c, a.summary)
