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

    pixels = sub.add_parser(
        'extract-pixels',
        help='Extract RGB pixels for peg, holder, and hand from requested frames in both cameras',
    )
    source = pixels.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset-path', help='HDF5 dataset; run prompts only on requested frames')
    source.add_argument('--input-dir', help='Already segmented analysis directory containing masks')
    pixels.add_argument('--demo', help='Demo name or numeric index; omit to process all complete demos')
    pixels.add_argument('--frames', nargs='+', type=int, metavar='N', help='Frame indices to extract; omit for first, handover (last pre-gap), and last')
    pixels.add_argument('-o', '--output-dir', help='Destination directory (default: <analysis>/pixels or <dataset>_pixels)')

    ana = sub.add_parser('analyze', help='Compute metrics from saved masks')
    _add_input_dir(
        ana,
        'Analysis directory containing masks and segmentation metadata',
    )
    ana.add_argument('--demo')

    plot = sub.add_parser('plot', help='Create figures from a summary CSV')
    _add_input_dir(plot, 'Analysis directory containing summary.csv')
    plot.add_argument('--summary', help='Optional summary CSV; defaults to <input-dir>/summary.csv')
    plot.add_argument(
        '--insertion-depth-threshold', type=float,
        help='Draw trials below this maximum insertion depth (mm) with x markers',
    )

    render = sub.add_parser(
        'render-demo',
        help='Overlay real RGB peg appearances as one task-evolution trace',
    )
    _add_input_dir(render, 'Analysis directory containing masks and segmentation metadata')
    render.add_argument('--demo', required=True, help='Demo name, for example demo_000002')
    render.add_argument(
        '--role', choices=('main', 'secondary', 'both'), default='both',
        help='Camera view to render (default: both)',
    )
    render.add_argument(
        '--traces', type=int, default=12,
        help='Number of sampled peg appearances per camera (default: 12)',
    )
    render.add_argument(
        '--alpha', type=float, default=0.82,
        help='Opacity of non-final peg appearances in (0, 1] (default: 0.82)',
    )
    render.add_argument(
        '--no-crop', action='store_true',
        help='Keep the full camera frame instead of cropping around the task',
    )
    render.add_argument('-o', '--output', help='Output PNG path')

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
    merge.add_argument(
        '--insertion-depth-threshold', type=float,
        help='Draw trials below this maximum insertion depth (mm) with x markers',
    )

    cumulative = sub.add_parser(
        'cumulative-plot',
        help='Compare methods with cumulative outcome curves, arranged by tolerance',
    )
    cumulative.add_argument(
        '--spec', required=True,
        help='JSON mapping methods and tolerances to one or more analyzed-data directories',
    )
    cumulative.add_argument(
        '-o', '--output-dir',
        help='Destination directory (default: cumulative_plots next to the JSON spec)',
    )
    cumulative.add_argument(
        '--nmax-trials', type=int,
        help='Plot only the N trials with the greatest maximum insertion depth; overrides JSON',
    )
    cumulative.add_argument(
        '--normalization-type', choices=('none', 'max', 'minmax', 'zscore'),
        help='Scatter angle normalization; overrides normalization_type in JSON',
    )

    gap = sub.add_parser(
        'insert-gap',
        help='Manually insert a timestamp gap before a frame in one demo',
    )
    gap.add_argument('--dataset-path', required=True, help='Input HDF5 dataset')
    gap.add_argument('--demo', required=True,
                     help='Demo name or numeric index, for example demo_000002 or 2')
    gap.add_argument('--frame', required=True, type=int,
                     help='First frame after the inserted gap; must be at least 1')
    gap.add_argument('--gap-ms', required=True, type=float,
                     help='Positive duration to add to timestamps from --frame onward')
    gap.add_argument('--camera-serial',
                     help='Update one camera only; by default all cameras in the demo are updated')
    destination = gap.add_mutually_exclusive_group()
    destination.add_argument('-o', '--output-path',
                             help='Output HDF5 path; defaults to <dataset>_with_gap.h5')
    destination.add_argument('--in-place', action='store_true',
                             help='Modify the source dataset instead of creating a copy')

    a = p.parse_args()

    if a.command == 'extract-pixels':
        from .extract_pixels import extract_pixels
        if a.input_dir:
            c = _load_analysis_config(a)
            source_kind = 'segmented'
            pixel_output = a.output_dir
        else:
            # Keep extraction artifacts separate from the normal analysis directory.
            default_pixels = str(Path(a.dataset_path).expanduser().resolve().with_suffix('')) + '_pixels'
            c = config(a.config, dataset_path=a.dataset_path, output_dir=default_pixels)
            source_kind = 'hdf5'
            pixel_output = a.output_dir or default_pixels
        extract_pixels(c, a.demo, a.frames, source=source_kind, output_dir=pixel_output)
        return

    if a.command == 'cumulative-plot':
        if a.nmax_trials is not None and a.nmax_trials < 1:
            cumulative.error('--nmax-trials must be at least 1')
        from .cumulative import cumulative_plots
        cumulative_plots(
            a.spec, a.output_dir, a.nmax_trials, a.normalization_type
        )
        return

    if a.command == 'insert-gap':
        if a.gap_ms <= 0:
            gap.error('--gap-ms must be greater than zero')
        gap_ns = round(a.gap_ms * 1_000_000)
        if gap_ns <= 0:
            gap.error('--gap-ms is too small to represent in nanoseconds')
        from .timestamp_gap import insert_timestamp_gap
        output_path, results = insert_timestamp_gap(
            a.dataset_path,
            a.demo,
            a.frame,
            gap_ns,
            output_path=a.output_path,
            in_place=a.in_place,
            camera_serial=a.camera_serial,
        )
        for result in results:
            print(
                f"Updated {result['demo']}/camera {result['camera_serial']}: "
                f"added {result['added_gap_ns']} ns before frame {result['frame']} "
                f"(resulting timestamp step {result['resulting_step_ns']} ns)"
            )
        print(f"Saved dataset: {output_path}")
        return

    if a.command == 'merge-plot':
        if len(a.output_dirs) < 2:
            merge.error('provide at least two output directories')
        if a.nmax_trials is not None and a.nmax_trials < 1:
            merge.error('--nmax-trials must be at least 1')
        from .plots import merge_plots
        merge_plots(
            a.output_dirs, a.output_dir, a.nmax_trials,
            insertion_depth_threshold=a.insertion_depth_threshold,
        )
        return

    if a.command in ('analyze', 'render-demo'):
        c = _load_analysis_config(a)
    elif a.command == 'plot':
        c = {'output_dir': str(Path(a.input_dir).expanduser().resolve())}
    else:
        c = _load_config(a)

    if a.command == 'inspect':
        import h5py
        from .core import camera_roles, demos, group, detect_gap
        with h5py.File(c['dataset_path'], 'r') as h:
            for d in demos(h):
                print(d)
                for role, serial in camera_roles(c):
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
    elif a.command == 'render-demo':
        if a.traces < 2:
            render.error('--traces must be at least 2')
        if not 0 < a.alpha <= 1:
            render.error('--alpha must be in (0, 1]')
        from .render import render_demo
        from .core import camera_roles
        available_roles = tuple(role for role, _ in camera_roles(c))
        roles = available_roles if a.role == 'both' else (a.role,)
        if any(role not in available_roles for role in roles):
            render.error('secondary camera is not configured')
        render_demo(
            c, a.demo, roles=roles, traces=a.traces, alpha=a.alpha,
            output_path=a.output, crop=not a.no_crop,
        )
    else:
        from .plots import make_plots
        make_plots(
            c, a.summary,
            insertion_depth_threshold=a.insertion_depth_threshold,
        )
