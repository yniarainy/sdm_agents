from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from agents.orchestrator import SDMOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDM multi-agent production pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run non-interactively using config defaults",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=0,
        metavar="N",
        help="Keep only the last N runs; delete older workspace directories (0 = keep all)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete ALL previous workspace runs before starting",
    )
    return parser.parse_args()


def clean_workspace(output_dir: str, keep: int = 0, clean_all: bool = False) -> int:
    """Remove old run directories from the workspace.

    Args:
        output_dir: Path to workspace directory.
        keep: If > 0, keep only the most recent N runs.
        clean_all: If True, delete all runs.

    Returns:
        Number of directories removed.
    """
    ws = Path(output_dir)
    if not ws.exists():
        return 0

    dirs = sorted(
        [d for d in ws.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda d: d.stat().st_mtime,
    )

    if clean_all:
        to_remove = dirs
    elif keep > 0 and len(dirs) > keep:
        to_remove = dirs[:-keep]
    else:
        return 0

    removed = 0
    for d in to_remove:
        try:
            shutil.rmtree(d)
            print(f"  Removed: {d.name}")
            removed += 1
        except Exception as exc:
            print(f"  Failed to remove {d.name}: {exc}")

    return removed


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    # ── Cleanup ──
    if args.clean or args.keep > 0:
        # Read output_dir from config
        import yaml
        config_path = Path(args.config)
        output_dir = "./workspace"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            output_dir = data.get("output_dir", output_dir)

        removed = clean_workspace(output_dir, keep=args.keep, clean_all=args.clean)
        print(f"Cleaned {removed} old run(s) from {output_dir}")

    # ── Run ──
    orchestrator = SDMOrchestrator(
        config_path=args.config,
        interactive=not args.auto,
    )

    try:
        state = orchestrator.run()
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        return 1

    print("\n=== SDM Pipeline Done ===")
    print(f"Output directory: {state.run_dir}")
    print("Step status:")
    for step, status in state.step_status.items():
        print(f"  - {step}: {status}")

    if state.error_events:
        print(f"Detected {len(state.error_events)} error event(s). "
              f"See: {state.artifacts.get('errors', 'errors.json')}")

    print("Key artifacts:")
    for key, path in state.artifacts.items():
        print(f"  - {key}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
