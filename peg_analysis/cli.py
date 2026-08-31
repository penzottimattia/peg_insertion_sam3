import argparse
from .core import config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', default='config.yaml')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect')
    check = sub.add_parser('check-prompts')
    check.add_argument('--max-demos', type=int, default=1)
    seg = sub.add_parser('segment', help='Run SAM and save masks')
    seg.add_argument('--demo')
    seg.add_argument('--overwrite', action='store_true')
    ana = sub.add_parser('analyze', help='Compute metrics from saved masks')
    ana.add_argument('--demo')
    plot = sub.add_parser('plot', help='Create figures from a summary CSV')
    plot.add_argument('--summary', help='Optional summary CSV; defaults to <output_dir>/summary.csv')
    a = p.parse_args()
    c = config(a.config)
    if a.command == 'inspect':
        import h5py
        from .core import demos, group, detect_gap
        with h5py.File(c['dataset_path'], 'r') as h:
            for d in demos(h):
                print(d)
                for role, serial in [('main', c['main_camera_serial']), ('secondary', c['secondary_camera_serial'])]:
                    g = group(h, d, serial)
                    print(role, len(g['rgb']), detect_gap(g['host_timestamp_ns'][:], c['onset']['gap_mad_multiplier'], c['onset']['gap_nominal_multiplier']))
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
