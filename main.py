#!/usr/bin/env python3
"""
WDM Optical Link OSNR/BER Performance Analyzer
===============================================
CLI entry point for analyzing multi-span WDM optical fiber links.

Supports:
  - Single link analysis with formatted report
  - Plotting (OSNR vs spans, power sweep, BER curve, WDM channels)
  - Max reach computation
  - Flask web dashboard launch
  - Configuration profile loading

Usage:
  python main.py --spans 10 --span-length 80 --launch-power 0
  python main.py --plot all
  python main.py --web
  python main.py --config submarine

Author: WDM Analyzer Team
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.link import OpticalLink
from core.metrics import evaluate_performance, get_osnr_threshold
from core.wdm import WDMSystem
from analysis.optimizer import find_optimal_launch_power, find_max_reach


def load_config(name: str) -> dict:
    """Load a configuration profile from the configs directory."""
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    config_path = os.path.join(config_dir, f"{name}.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def print_report(link_result: dict, perf: dict, reach: dict, opt: dict,
                 nf_db: float, wdm_summary: dict = None) -> None:
    """Print a formatted CLI performance report."""
    status_icon = "[PASS]" if perf["pass"] else "[FAIL]"
    status_color_start = ""
    status_color_end = ""

    print()
    print("=" * 54)
    print("   WDM OPTICAL LINK PERFORMANCE REPORT")
    print("=" * 54)
    print(f"  Total Length      : {link_result['total_length_km']:.0f} km")
    print(f"  Number of Spans   : {link_result['num_spans']}")
    print(f"  Launch Power      : {link_result['launch_power_dbm']:.1f} dBm")
    print(f"  EDFA NF           : {nf_db:.1f} dB")
    print(f"  Bitrate           : {perf['bitrate_gbps']:.0f} Gbps")
    print(f"  Modulation        : {perf['modulation']}")
    print(f"  Nonlinear Effects : {'ON' if link_result.get('include_nonlinear', False) else 'OFF'}")
    print("-" * 54)
    print(f"  Signal Power (Rx) : {link_result['signal_power_at_rx_dbm']:.2f} dBm")
    print(f"  ASE Noise Total   : {link_result['total_ase_dbm']:.2f} dBm")

    if link_result.get("include_nonlinear", False):
        print(f"  NLI Noise Total   : {link_result['total_nli_dbm']:.2f} dBm")
        print(f"  Total Noise       : {link_result['total_noise_dbm']:.2f} dBm")

    print(f"  OSNR              : {perf['osnr_db']:.2f} dB")
    print(f"  OSNR Threshold    : {perf['threshold_db']:.1f} dB")
    print(f"  Q-Factor          : {perf['q_factor']:.2f}")
    print(f"  BER               : {perf['ber_scientific']}")
    print(f"  OSNR Margin       : {perf['margin_db']:.2f} dB")
    print("-" * 54)
    print(f"  Max Reach         : {reach['max_spans']} spans ({reach['max_distance_km']:.0f} km)")
    print(f"  Optimal Power     : {opt['optimal_power_dbm']:.1f} dBm (OSNR: {opt['achieved_osnr_db']:.1f} dB)")
    print("-" * 54)
    print(f"  Status            : {status_icon}")
    print("=" * 54)

    if wdm_summary:
        print()
        print("-" * 54)
        print("   WDM CHANNEL SUMMARY")
        print("-" * 54)
        print(f"  Channels          : {wdm_summary['num_channels']}")
        print(f"  Best OSNR         : {wdm_summary['best_osnr_db']:.2f} dB (Ch {wdm_summary['best_channel']})")
        print(f"  Worst OSNR        : {wdm_summary['worst_osnr_db']:.2f} dB (Ch {wdm_summary['worst_channel']})")
        print(f"  Avg OSNR          : {wdm_summary['average_osnr_db']:.2f} dB")
        print(f"  Pass / Fail       : {wdm_summary['channels_pass']} / {wdm_summary['channels_fail']}")
        print("=" * 54)
    print()


def generate_plots(args, plot_type: str) -> None:
    """Generate and save/show plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualization.plotter import (
        plot_osnr_vs_spans, plot_launch_power_sweep,
        plot_ber_vs_osnr, plot_wdm_channel_osnr,
    )

    plots_to_generate = []
    if plot_type == "all":
        plots_to_generate = ["osnr_vs_spans", "power_sweep", "ber_curve", "wdm"]
    else:
        plots_to_generate = [plot_type]

    for ptype in plots_to_generate:
        print(f"  Generating plot: {ptype}...")

        if ptype == "osnr_vs_spans":
            fig = plot_osnr_vs_spans(
                span_length_km=args.span_length,
                launch_power_dbm=args.launch_power,
                max_spans=max(args.spans + 10, 30),
                bitrate_gbps=args.bitrate,
                nf_db=args.nf,
                include_nonlinear=not args.no_nonlinear,
            )
        elif ptype == "power_sweep":
            fig = plot_launch_power_sweep(
                num_spans=args.spans,
                span_length_km=args.span_length,
                bitrate_gbps=args.bitrate,
                nf_db=args.nf,
                include_nonlinear=not args.no_nonlinear,
            )
        elif ptype == "ber_curve":
            fig = plot_ber_vs_osnr(bitrate_gbps=args.bitrate)
        elif ptype == "wdm":
            wdm = WDMSystem(num_channels=max(args.wdm_channels, 2))
            results = wdm.analyze_all_channels({
                "num_spans": args.spans,
                "span_length_km": args.span_length,
                "launch_power_dbm": args.launch_power,
                "edfa_noise_figure_db": args.nf,
                "bitrate_gbps": args.bitrate,
                "modulation": args.modulation,
                "include_nonlinear": not args.no_nonlinear,
            })
            fig = plot_wdm_channel_osnr(results, bitrate_gbps=args.bitrate)
        else:
            print(f"  Unknown plot type: {ptype}")
            continue

        filename = f"plot_{ptype}.png"
        fig.savefig(filename, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved: {filename}")

    print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WDM Optical Link OSNR/BER Performance Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --spans 10 --span-length 80 --launch-power 0
  python main.py --spans 30 --span-length 80 --bitrate 10
  python main.py --plot all
  python main.py --plot osnr_vs_spans --spans 20
  python main.py --max-reach --launch-power 2
  python main.py --web
  python main.py --config submarine
        """,
    )

    # Link parameters
    parser.add_argument("--spans", type=int, default=10,
                        help="Number of fiber spans (default: 10)")
    parser.add_argument("--span-length", type=float, default=80.0,
                        help="Span length in km (default: 80)")
    parser.add_argument("--launch-power", type=float, default=0.0,
                        help="Launch power in dBm (default: 0)")
    parser.add_argument("--nf", type=float, default=5.0,
                        help="EDFA noise figure in dB (default: 5.0)")

    # Signal parameters
    parser.add_argument("--bitrate", type=float, default=10.0,
                        help="Bitrate in Gbps (default: 10)")
    parser.add_argument("--modulation", type=str, default="OOK",
                        choices=["OOK", "QPSK", "16QAM"],
                        help="Modulation format (default: OOK)")
    parser.add_argument("--wdm-channels", type=int, default=1,
                        help="Number of WDM channels (default: 1)")

    # Actions
    parser.add_argument("--plot", type=str, default=None,
                        choices=["osnr_vs_spans", "power_sweep", "ber_curve", "wdm", "all"],
                        help="Generate plot(s)")
    parser.add_argument("--web", action="store_true",
                        help="Launch Flask web dashboard on port 5000")
    parser.add_argument("--max-reach", action="store_true",
                        help="Compute and print max reach")
    parser.add_argument("--config", type=str, default=None,
                        help="Load configuration profile (default_system, submarine, metro, long_haul)")
    parser.add_argument("--no-nonlinear", action="store_true",
                        help="Disable nonlinear effects (ASE-only mode)")

    args = parser.parse_args()

    # Load config profile if specified
    if args.config:
        cfg = load_config(args.config)
        if cfg:
            print(f"Loaded profile: {args.config}")
            args.spans = cfg.get("num_spans", args.spans)
            args.span_length = cfg.get("span_length_km", args.span_length)
            args.launch_power = cfg.get("launch_power_dbm", args.launch_power)
            args.nf = cfg.get("edfa_noise_figure_db", args.nf)
            args.bitrate = cfg.get("bitrate_gbps", args.bitrate)
            args.modulation = cfg.get("modulation", args.modulation)
            args.wdm_channels = cfg.get("num_wdm_channels", args.wdm_channels)
        else:
            print(f"Warning: Config '{args.config}' not found, using defaults.")

    # Launch web dashboard
    if args.web:
        from web.app import app
        print("Starting WDM Analyzer Web Dashboard...")
        print("Open http://localhost:5000 in your browser")
        app.run(debug=True, port=5000, use_reloader=False)
        return

    # Run link analysis
    include_nl = not args.no_nonlinear
    link = OpticalLink(
        num_spans=args.spans,
        span_length_km=args.span_length,
        launch_power_dbm=args.launch_power,
        edfa_noise_figure_db=args.nf,
        include_nonlinear=include_nl,
    )
    link_result = link.analyze()

    # Performance evaluation
    perf = evaluate_performance(
        osnr_db=link_result["osnr_db"],
        bitrate_gbps=args.bitrate,
        modulation=args.modulation,
    )

    # Max reach
    threshold = get_osnr_threshold(args.bitrate)
    reach = find_max_reach(
        launch_power_dbm=args.launch_power,
        span_length_km=args.span_length,
        target_osnr_db=threshold,
        nf_db=args.nf,
        include_nonlinear=include_nl,
    )

    # Optimal power
    opt = find_optimal_launch_power(
        num_spans=args.spans,
        span_length_km=args.span_length,
        nf_db=args.nf,
        bitrate_gbps=args.bitrate,
        include_nonlinear=include_nl,
    )

    # WDM analysis
    wdm_summary = None
    if args.wdm_channels > 1:
        wdm = WDMSystem(num_channels=args.wdm_channels)
        wdm.analyze_all_channels({
            "num_spans": args.spans,
            "span_length_km": args.span_length,
            "launch_power_dbm": args.launch_power,
            "edfa_noise_figure_db": args.nf,
            "bitrate_gbps": args.bitrate,
            "modulation": args.modulation,
            "include_nonlinear": include_nl,
        })
        wdm_summary = wdm.channel_summary()

    # Print report
    print_report(link_result, perf, reach, opt, args.nf, wdm_summary)

    # Max reach flag
    if args.max_reach:
        print(f"Max Reach: {reach['max_spans']} spans = {reach['max_distance_km']:.0f} km")
        print(f"  (OSNR threshold: {threshold:.1f} dB for {args.bitrate:.0f} Gbps)")
        print()

    # Generate plots
    if args.plot:
        generate_plots(args, args.plot)

    # Exit code: 0 for PASS, 1 for FAIL
    sys.exit(0 if perf["pass"] else 1)


if __name__ == "__main__":
    main()
