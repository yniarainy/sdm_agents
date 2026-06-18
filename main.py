from __future__ import annotations

import argparse
import sys

from agents.orchestrator import SDMOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDM multi-agent production pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run non-interactively using config defaults",
    )
    return parser.parse_args()


def main() -> int:
	if hasattr(sys.stdout, "reconfigure"):
		sys.stdout.reconfigure(encoding="utf-8", errors="replace")
	if hasattr(sys.stderr, "reconfigure"):
		sys.stderr.reconfigure(encoding="utf-8", errors="replace")

	args = parse_args()

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
		print(f"- {step}: {status}")

	if state.error_events:
		print(f"Detected {len(state.error_events)} error event(s). See: {state.artifacts.get('errors', 'errors.json')}")

	print("Key artifacts:")
	for key, path in state.artifacts.items():
		print(f"- {key}: {path}")

	return 0


if __name__ == "__main__":
    sys.exit(main())