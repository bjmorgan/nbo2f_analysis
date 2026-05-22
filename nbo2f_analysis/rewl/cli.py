"""argparse CLI for the REWL driver: `rewl run|resume|postprocess`."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

from nbo2f_analysis.rewl.config import RewlConfig, load_yaml


def _maybe_chdir(out_dir: str | None) -> None:
    if out_dir is not None:
        target = Path(out_dir).resolve()
        target.mkdir(parents=True, exist_ok=True)
        os.chdir(target)
    print(f"CWD: {Path.cwd()}")


def _load_and_override(args) -> RewlConfig:
    cfg = load_yaml(args.config)
    if getattr(args, "seed", None) is not None:
        cfg = replace(cfg, random_seed=int(args.seed))
    return cfg


def _cmd_run(args) -> int:
    _maybe_chdir(args.out_dir)
    cfg = _load_and_override(args)
    from nbo2f_analysis.rewl.run import run
    run(cfg, force=args.force)
    return 0


def _cmd_resume(args) -> int:
    _maybe_chdir(args.out_dir)
    cfg = _load_and_override(args)
    from nbo2f_analysis.rewl.run import resume
    resume(cfg, extra_cycles=args.extra_cycles)
    return 0


def _cmd_postprocess(args) -> int:
    _maybe_chdir(args.out_dir)
    cfg = _load_and_override(args)
    from nbo2f_analysis.rewl.postprocess import postprocess
    postprocess(cfg)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rewl",
        description="REWL driver for NbO2F finite-size scaling.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Start a fresh REWL run.")
    run_p.add_argument("config", type=Path, help="Path to YAML config.")
    run_p.add_argument("--seed", type=int, default=None,
                       help="Override random_seed from the YAML.")
    run_p.add_argument("--out-dir", default=None,
                       help="Run directory (default: CWD).")
    run_p.add_argument("--force", action="store_true",
                       help="Overwrite an existing checkpoint.")
    run_p.set_defaults(func=_cmd_run)

    res_p = sub.add_parser("resume", help="Continue a checkpointed run.")
    res_p.add_argument("config", type=Path, help="Path to YAML config.")
    res_p.add_argument("--extra-cycles", type=int, default=None,
                       help="Run N more cycles beyond what is in the checkpoint.")
    res_p.add_argument("--out-dir", default=None,
                       help="Run directory (default: CWD).")
    res_p.set_defaults(func=_cmd_resume)

    pp_p = sub.add_parser(
        "postprocess",
        help="Re-derive CSVs + figure from the checkpoint, no WL run.",
    )
    pp_p.add_argument("config", type=Path, help="Path to YAML config.")
    pp_p.add_argument("--out-dir", default=None,
                      help="Run directory (default: CWD).")
    pp_p.set_defaults(func=_cmd_postprocess)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
