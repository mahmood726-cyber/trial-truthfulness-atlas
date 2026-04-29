"""Command-line entry point. Subcommands: preflight, build, sweep, verify-one."""

from __future__ import annotations

import argparse
import sys

from tta import preflight


def cmd_preflight(args: argparse.Namespace) -> int:
    checks = preflight.run_checks()
    print(preflight.format_action_list(checks))
    return 0 if all(c.ok for c in checks) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tta", description="Trial Truthfulness Atlas")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight", help="check external prereqs")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "preflight":
        return cmd_preflight(args)
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
