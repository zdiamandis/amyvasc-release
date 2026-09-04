#!/usr/bin/env python3
"""Run the manuscript Rapidtide configuration and summarize its delay maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amyvasc_release.rest_lags import (  # noqa: E402
    load_targets,
    prepare_runs,
    read_subjects,
    run_rapidtide_jobs,
    select_rest_cohort,
    summarize_rapidtide,
    write_run_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser(
        "select-cohort",
        help="select participants with all four resting-state BOLD and motion files",
    )
    select.add_argument("--subjects", type=Path, required=True)
    select.add_argument("--hcp-root", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--audit-output", type=Path, required=True)

    for name in ("plan", "run"):
        subparser = commands.add_parser(
            name,
            help=(
                "write an executable per-run command file"
                if name == "plan"
                else "execute the four per-participant runs sequentially"
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        subparser.add_argument("--subjects", type=Path, required=True)
        subparser.add_argument("--hcp-root", type=Path, required=True)
        subparser.add_argument("--rapidtide-root", type=Path, required=True)
        subparser.add_argument("--executable", default="rapidtide")
        subparser.add_argument("--n-processes", type=int, default=4)
        if name == "plan":
            subparser.add_argument("--output", type=Path, required=True)
        else:
            subparser.add_argument("--overwrite", action="store_true")

    summarize = commands.add_parser(
        "summarize",
        help="build consensus maps and participant-first lag/gradient summaries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    summarize.add_argument("--subjects", type=Path, required=True)
    summarize.add_argument("--rapidtide-root", type=Path, required=True)
    summarize.add_argument(
        "--amygdala-mask",
        type=Path,
        required=True,
        help="Binary CIT168 pAmy >= 0.50 mask from 01_prepare_masks.py.",
    )
    summarize.add_argument("--targets", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    summarize.add_argument(
        "--existing-consensus-only",
        action="store_true",
        help="Use saved consensus maps and omit optional run-level reproducibility tables.",
    )
    summarize.add_argument("--min-coverage", type=int, default=2)
    summarize.add_argument("--n-bins", type=int, default=6)
    summarize.add_argument("--min-bin-voxels", type=int, default=2)
    summarize.add_argument("--min-valid-bins", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    subjects = read_subjects(args.subjects)
    if args.command == "select-cohort":
        audit = select_rest_cohort(subjects, args.hcp_root)
        retained = audit.loc[audit["included"], ["subject_id"]]
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(args.audit_output, sep="\t", index=False)
        if retained.empty:
            raise ValueError(
                f"No participants have all four resting-state runs; see {args.audit_output}"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        retained.to_csv(args.output, sep="\t", index=False)
        print(
            f"Selected {len(retained)} of {len(audit)} participants; audit: {args.audit_output}"
        )
        return 0

    if args.command in {"plan", "run"}:
        jobs = prepare_runs(subjects, args.hcp_root, args.rapidtide_root)
        if args.command == "plan":
            write_run_plan(
                jobs,
                args.output,
                executable=args.executable,
                n_processes=args.n_processes,
            )
            print(f"Wrote {len(jobs)} commands to {args.output}")
        else:
            run_rapidtide_jobs(
                jobs,
                executable=args.executable,
                n_processes=args.n_processes,
                overwrite=args.overwrite,
            )
            print(f"Completed {len(jobs)} Rapidtide jobs")
        return 0

    targets = load_targets(args.targets)
    tables = summarize_rapidtide(
        rapidtide_root=args.rapidtide_root,
        subjects=subjects,
        amygdala_mask_path=args.amygdala_mask,
        targets=targets,
        output_dir=args.output_dir,
        build_missing_consensus=not args.existing_consensus_only,
        include_run_reproducibility=not args.existing_consensus_only,
        min_coverage=args.min_coverage,
        n_bins=args.n_bins,
        min_bin_voxels=args.min_bin_voxels,
        min_valid_bins=args.min_valid_bins,
        strict=True,
    )
    print(
        f"Summarized {len(tables['consensus_qc'])} participants and "
        f"{len(targets)} targets in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
