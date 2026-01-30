from __future__ import annotations

import argparse
import logging
from typing import Any, Dict

from .compare import format_comparison, run_comparison, save_comparison
from .core.certificate import dump_certificate
from .core.config import load_config
from .core.grid import build_grid
from .core.output import create_output_dir, generate_plots, write_raw_responses, write_summary
from .core.prompt import render_prompts
from .core.runner import AlignmentSimRunner
from .games.compile_game import compile_game
from .games.game_runner import run_game
from .games.run_suite import run_suite
from .games.summarize_game import summarize_game

logger = logging.getLogger(__name__)


def _run_dry(cfg: Dict[str, Any]) -> None:
    grid = build_grid(cfg)
    for i, cell in enumerate(grid, start=1):
        prompts = render_prompts(cfg, cell)
        print(f"Cell {i}: {cell}")
        if prompts["system"]:
            print("System:")
            print(prompts["system"])
        if prompts.get("history"):
            print("History:")
            for turn in prompts["history"]:
                print(f"  [{turn['role']}]: {turn['content'][:80]}{'...' if len(turn['content']) > 80 else ''}")
        if prompts["user"]:
            print("User:")
            print(prompts["user"])
        print("-" * 40)


def _setup_logging(verbosity: int) -> None:
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    if verbosity >= 2:
        fmt = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"

    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    logger.debug("Logging initialized at level %s", logging.getLevelName(level))


def main() -> None:
    parser = argparse.ArgumentParser(prog="alignmentsim")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v for INFO, -vv for DEBUG)")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run an AlignmentSim config.")
    run_parser.add_argument("config", help="Path to YAML config.")
    run_parser.add_argument("--output", help="Write certificate YAML to this path (legacy, prefer --output-dir).")
    run_parser.add_argument("--output-dir", help="Base directory for outputs (default: outputs/). Creates timestamped subdir with certificate, summary, and plots.")
    run_parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    run_parser.add_argument("--concurrency", "-c", type=int, default=10, help="Max concurrent API requests (default: 10).")
    run_parser.add_argument("--dry-run", action="store_true", help="Print grid/prompt rendering only.")

    compare_parser = sub.add_parser("compare", help="Run config across multiple models and compare results.")
    compare_parser.add_argument("config", help="Path to YAML config.")
    compare_parser.add_argument("--models", "-m", required=True, help="Comma-separated list of models to compare.")
    compare_parser.add_argument("--output-dir", help="Base directory for outputs (default: outputs/).")
    compare_parser.add_argument("--concurrency", "-c", type=int, default=10, help="Max concurrent API requests (default: 10).")
    compare_parser.add_argument("--format", choices=["table", "csv", "markdown"], default="table", help="Output format (default: table).")

    compile_parser = sub.add_parser("compile-game", help="Compile a GameSpec into a suite of configs.")
    compile_parser.add_argument("gamespec", help="Path to GameSpec YAML.")
    compile_parser.add_argument("--output", help="Output directory for suite (default: suites/<game_name>/).")
    compile_parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if not empty.")

    run_suite_parser = sub.add_parser("run-suite", help="Run a compiled suite manifest.")
    run_suite_parser.add_argument("suite", help="Path to suite.yaml or suite directory.")
    run_suite_parser.add_argument("--output", help="Override results_dir from suite.yaml.")
    run_suite_parser.add_argument("--concurrency", "-c", type=int, default=10, help="Max concurrent API requests (default: 10).")
    run_suite_parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")

    run_game_parser = sub.add_parser("run-game", help="Run a multi-agent GameSpec directly.")
    run_game_parser.add_argument("gamespec", help="Path to GameSpec YAML.")
    run_game_parser.add_argument("--output-dir", help="Base output directory for results (default: outputs/).")
    run_game_parser.add_argument("--concurrency", "-c", type=int, default=10, help="Max concurrent samples (default: 10).")
    run_game_parser.add_argument("--no-plots", action="store_true", help="Skip plot generation (if supported).")
    run_game_parser.add_argument("--n-samples", type=int, help="Override samples per mechanism (default: suite.n_per_cell).")

    summarize_parser = sub.add_parser("summarize-game", help="Summarize a suite run into equilibrium summary.")
    summarize_parser.add_argument("suite", help="Path to suite.yaml or suite directory.")
    summarize_parser.add_argument("--output", help="Output path for equilibrium_summary.yaml.")

    aggregate_parser = sub.add_parser("aggregate-inspect", help="Aggregate results from an Inspect eval log.")
    aggregate_parser.add_argument("log", help="Path to Inspect eval log (JSON or directory).")
    aggregate_parser.add_argument("--output", help="Output path for summary YAML.")
    aggregate_parser.add_argument("--format", choices=["yaml", "json"], default="yaml", help="Output format (default: yaml).")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "run":
        logger.info("Running AlignmentSim")
        cfg = load_config(args.config)
        if args.dry_run:
            logger.info("Dry run mode - printing prompts only")
            _run_dry(cfg)
            return

        runner = AlignmentSimRunner(cfg, concurrency=args.concurrency)
        cert = runner.run()

        # Extract internal data for output generation
        results = cert.pop("_results", [])
        config = cert.pop("_config", cfg)

        # Determine output mode
        if args.output_dir is not None or args.output is None:
            # Use output directory mode (new default)
            base_dir = args.output_dir if args.output_dir else "outputs"
            output_dir = create_output_dir(config, base_dir)

            # Write certificate
            cert_path = f"{output_dir}/certificate.yaml"
            dump_certificate(cert, cert_path)

            # Write summary
            write_summary(output_dir, config, results, cert["fit"], cert["predictions"])

            # Write raw responses
            write_raw_responses(output_dir, results, config)

            # Generate plots
            if not args.no_plots:
                generate_plots(output_dir, config, results, cert["fit"])

            logger.info("All outputs written to: %s", output_dir)
            print(f"\nOutputs written to: {output_dir}/")
            print(f"  - certificate.yaml")
            print(f"  - summary.txt")
            print(f"  - raw_responses.json")
            if not args.no_plots:
                print(f"  - plots/")
        else:
            # Legacy single-file mode
            dump_certificate(cert, args.output)
            logger.info("Certificate written to: %s", args.output)

    elif args.command == "compare":
        logger.info("Running AlignmentSim comparison")
        cfg = load_config(args.config)
        models = [m.strip() for m in args.models.split(",")]
        logger.info("Comparing %d models: %s", len(models), ", ".join(models))

        comparison = run_comparison(cfg, models, args.concurrency)

        # Output formatted comparison
        output = format_comparison(comparison, cfg, args.format)
        print(output)

        # Save if output_dir specified
        if args.output_dir:
            cert_path = save_comparison(comparison, cfg, args.output_dir)
            print(f"Comparison saved to: {cert_path}")

    elif args.command == "compile-game":
        out_dir = compile_game(args.gamespec, output_dir=args.output, overwrite=args.overwrite)
        print(f"Suite compiled to: {out_dir}")

    elif args.command == "run-suite":
        out_dir = run_suite(args.suite, output_dir=args.output, concurrency=args.concurrency, no_plots=args.no_plots)
        print(f"Suite results written to: {out_dir}")

    elif args.command == "run-game":
        out_dir = run_game(
            args.gamespec,
            output_dir=args.output_dir,
            concurrency=args.concurrency,
            n_samples=args.n_samples,
            no_plots=args.no_plots,
        )
        print(f"Game results written to: {out_dir}")

    elif args.command == "summarize-game":
        out_path = summarize_game(args.suite, output_path=args.output)
        print(f"Equilibrium summary written to: {out_path}")

    elif args.command == "aggregate-inspect":
        from .inspect_compatibility.aggregate import aggregate_results, compare_mechanisms, write_summary as write_agg_summary
        import json
        import yaml

        summary = aggregate_results(args.log)

        if args.output:
            write_agg_summary(summary, args.output, format=args.format)
            print(f"Summary written to: {args.output}")
        else:
            # Print to stdout
            if args.format == "json":
                print(json.dumps({"game_summary": summary}, indent=2))
            else:
                print(yaml.safe_dump({"game_summary": summary}, sort_keys=False, default_flow_style=False))

        # Also print comparison if multiple mechanisms
        if len(summary) > 1:
            comparison = compare_mechanisms(summary)
            print("\n--- Mechanism Comparison ---")
            if args.format == "json":
                print(json.dumps(comparison, indent=2))
            else:
                print(yaml.safe_dump(comparison, sort_keys=False, default_flow_style=False))


if __name__ == "__main__":
    main()
