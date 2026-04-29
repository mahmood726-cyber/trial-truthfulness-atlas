"""Command-line entry point. Subcommands: preflight, build, sweep, verify-one."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from tta import config, pipeline, preflight
from tta.dashboards import atlas_dashboard, verification_ui
from tta.judge.ollama_client import OllamaClient


def _resolve_fixtures_dir() -> Path:
    return Path(__file__).parent.parent.parent / "tests" / "fixtures"


def _resolve_out_dir() -> Path:
    return config.OUTPUTS_DIR


def _make_ollama_client() -> OllamaClient:
    return OllamaClient()


def _resolve_snapshot_date(args: argparse.Namespace) -> date:
    raw = getattr(args, "snapshot_date", None) or config.SNAPSHOT_DATE
    return date.fromisoformat(raw)


def cmd_preflight(args: argparse.Namespace) -> int:
    checks = preflight.run_checks()
    print(preflight.format_action_list(checks))
    return 0 if all(c.ok for c in checks) else 1


def cmd_build(args: argparse.Namespace) -> int:
    out_dir = _resolve_out_dir()
    snapshot = _resolve_snapshot_date(args)
    if args.fixture_mode:
        atlas, rollup = pipeline.run_5trial_fixture(
            fixtures_dir=_resolve_fixtures_dir(),
            out_dir=out_dir,
            snapshot_date=snapshot,
            ollama_client=_make_ollama_client(),
        )
    else:
        # Full sweep is v0.2.0 work; keep v0.1.x honest by routing to fixture.
        print("Full Pairwise70 sweep is deferred to v0.2.0. Use --fixture-mode for v0.1.x.")
        return 2
    atlas_dashboard.write(atlas, out_dir / "dashboard.html",
                          title=f"Trial Truthfulness Atlas v{__import__('tta').__version__}")
    verification_ui.write(atlas, out_dir / "verification.html",
                          title="TTA verification")
    print(f"Wrote {out_dir / 'atlas.csv'} ({len(atlas)} trials).")
    print(f"Wrote {out_dir / 'ma_rollup.csv'} ({len(rollup)} MAs).")
    print(f"Wrote {out_dir / 'dashboard.html'} and {out_dir / 'verification.html'}.")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    return cmd_build(args)


def cmd_verify_one(args: argparse.Namespace) -> int:
    if not args.fixture_mode:
        print("Real-data verify-one is deferred to v0.2.0. Pass --fixture-mode "
              "to inspect a fixture trial.")
        return 2
    out_dir = _resolve_out_dir()
    atlas, _ = pipeline.run_5trial_fixture(
        fixtures_dir=_resolve_fixtures_dir(),
        out_dir=out_dir,
        snapshot_date=_resolve_snapshot_date(args),
        ollama_client=_make_ollama_client(),
    )
    target = atlas[atlas["nct_id"] == args.nct]
    if target.empty:
        print(f"NCT {args.nct} not found in atlas.")
        return 1
    row = target.iloc[0]
    for k in ["nct_id", "Study", "review_doi", "bridge_method",
              "outcome_drift", "n_drift", "direction_concordance",
              "results_posting"]:
        print(f"  {k}: {row.get(k)}")
    return 0


def _add_snapshot_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--snapshot-date", dest="snapshot_date", default=None,
                   help="ISO date (YYYY-MM-DD) of the AACT snapshot; "
                        "defaults to TTA_SNAPSHOT_DATE env or config.SNAPSHOT_DATE")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tta", description="Trial Truthfulness Atlas")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("preflight", help="check external prereqs")

    b = sub.add_parser("build", help="build atlas + dashboards (fixture mode in v0.1.x)")
    b.add_argument("--fixture-mode", action="store_true",
                   help="run on the 5-trial test fixture instead of real data")
    _add_snapshot_arg(b)

    s = sub.add_parser("sweep", help="alias for build (v0.1.x)")
    s.add_argument("--fixture-mode", action="store_true")
    _add_snapshot_arg(s)

    v = sub.add_parser("verify-one", help="print all flags for one NCT")
    v.add_argument("--nct", required=True, help="CT.gov NCT identifier")
    v.add_argument("--fixture-mode", action="store_true")
    _add_snapshot_arg(v)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "preflight":
        return cmd_preflight(args)
    if args.cmd == "build":
        return cmd_build(args)
    if args.cmd == "sweep":
        return cmd_sweep(args)
    if args.cmd == "verify-one":
        return cmd_verify_one(args)
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
