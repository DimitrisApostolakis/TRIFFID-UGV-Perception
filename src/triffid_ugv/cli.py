"""Command line: `python -m triffid_ugv <command>`.

  run      start the live ROS 2 node (subscribes to the camera, status, fix and heading topics)
  check    validate the config, model files and support bank without loading a model or ROS
  version  print the package version
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from . import __version__
from .config import load_config


def _config_options(parser):
    parser.add_argument('--config', default='configs/default.yaml', help='YAML config (default: %(default)s)')
    parser.add_argument('--root', help='Folder that relative config paths resolve against (default: CWD)')
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='SECTION.KEY=VALUE',
                        help='Override a config value, e.g. --set detector.backend=sam3 (repeatable)')


def build_parser():
    parser = argparse.ArgumentParser(prog='triffid_ugv', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true')
    commands = parser.add_subparsers(dest='command', required=True)
    _config_options(commands.add_parser('run', help='Run the live node'))
    _config_options(commands.add_parser('check', help='Validate config, model files and supports'))
    commands.add_parser('version', help='Print the version')
    return parser


def check(config) -> list[str]:
    """Human-readable problems that would stop or degrade a run (empty = ready)."""
    problems = []
    det = config.detector
    if det.backend == 'yoloe' and not Path(det.yoloe.checkpoint).is_file():
        problems.append(f'YOLOE checkpoint missing: {det.yoloe.checkpoint}')
    if det.backend == 'sam3':
        for path in (det.sam3.checkpoint, det.sam3.bpe, Path(det.sam3.repo) / 'sam3/model_builder.py'):
            if not Path(path).is_file():
                problems.append(f'SAM3 file missing: {path}')
    needs_bank = det.backend == 'yoloe' or (det.backend == 'sam3' and (det.sam3.prompt != 'text'))
    if needs_bank or det.classes in ('auto', '', None):
        from .supports import SupportBank
        try:
            bank = SupportBank(config.paths.supports)
            summary = bank.summary()
            print(f'Support bank {bank.path}: {len(summary)} classes, {sum(summary.values())} examples')
            for name, count in summary.items():
                print(f'  {name:22s} {count}')
            if not summary:
                problems.append('Support bank has no usable examples')
        except (FileNotFoundError, ValueError) as error:
            if needs_bank:
                problems.append(f'Support bank: {error}')
    if config.tracker.backend == 'masa':
        if not Path(config.tracker.masa.checkpoint).is_file():
            problems.append(f'MASA checkpoint missing: {config.tracker.masa.checkpoint}')
        if not Path(config.tracker.masa.repo, 'masa').is_dir():
            problems.append(f'MASA source missing at {config.tracker.masa.repo} (build the `masa` Docker target)')
    if not any(s in config.position.sources for s in ('status', 'fix', 'api')) and config.position.fixed_lat is None:
        problems.append('No position source configured')
    return problems


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    if args.command == 'version':
        print(__version__)
        return 0
    config = load_config(args.config, args.overrides, root=args.root)
    if args.command == 'check':
        print(json.dumps({k: config.to_dict()[k] for k in ('detector', 'tracker', 'topics')}, indent=1))
        problems = check(config)
        for problem in problems:
            print(f'PROBLEM: {problem}', file=sys.stderr)
        print('OK' if not problems else f'{len(problems)} problem(s)')
        return 1 if problems else 0
    if args.command == 'run':
        problems = check(config)
        if problems:
            for problem in problems:
                logging.error(problem)
            return 1
        from .app import build_pipeline
        from .ros.node import spin
        spin(config, build_pipeline(config))
        return 0
    return 2
