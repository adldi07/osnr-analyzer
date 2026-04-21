"""
Flask Web Dashboard
===================
Single-page web application for interactive WDM optical link analysis.

Routes:
  GET  /              → Renders dashboard with default configuration
  POST /analyze       → Full link analysis → JSON response
  POST /plot/<type>   → Generates matplotlib plot → base64 PNG
  GET  /max_reach     → Computes maximum transmission reach → JSON

The dashboard auto-updates on input changes (debounced 500ms)
and displays OSNR, Q-factor, BER, and per-channel WDM results.
"""

import io
import base64
import json
import os

from flask import Flask, render_template, request, jsonify

# Configure matplotlib for non-interactive backend BEFORE any other imports
import matplotlib
matplotlib.use("Agg")

from core.link import OpticalLink
from core.metrics import evaluate_performance, get_osnr_threshold
from core.wdm import WDMSystem
from analysis.sweep import osnr_vs_spans, osnr_vs_launch_power
from analysis.optimizer import find_optimal_launch_power, find_max_reach
from visualization.plotter import (
    plot_osnr_vs_spans,
    plot_launch_power_sweep,
    plot_ber_vs_osnr,
    plot_wdm_channel_osnr,
)

# ─── Flask App Configuration ─────────────────────────────────────────────────
app = Flask(__name__)

# Load default configuration
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")


def _load_config(name: str = "default_system") -> dict:
    """Load a configuration profile from the configs directory."""
    config_path = os.path.join(CONFIG_DIR, f"{name}.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def _get_available_configs() -> list[str]:
    """List all available configuration profiles."""
    configs = []
    if os.path.isdir(CONFIG_DIR):
        for fname in os.listdir(CONFIG_DIR):
            if fname.endswith(".json"):
                configs.append(fname.replace(".json", ""))
    return sorted(configs)


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    matplotlib.pyplot.close(fig)
    return img_b64


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main dashboard page."""
    config = _load_config("default_system")
    configs = _get_available_configs()
    return render_template("index.html", config=config, configs=configs)


@app.route("/load_config/<name>")
def load_config(name: str):
    """Load a specific configuration profile."""
    config = _load_config(name)
    if not config:
        return jsonify({"error": f"Config '{name}' not found"}), 404
    return jsonify(config)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Perform full link analysis.

    Accepts JSON body:
      {spans, span_length, launch_power, nf, bitrate, modulation,
       wdm_channels, include_nonlinear}

    Returns JSON with link analysis, performance metrics, and
    optimal power / max reach data.
    """
    data = request.get_json()

    num_spans = int(data.get("spans", 10))
    span_length = float(data.get("span_length", 80))
    launch_power = float(data.get("launch_power", 0))
    nf = float(data.get("nf", 5.0))
    bitrate = float(data.get("bitrate", 10))
    modulation = data.get("modulation", "OOK")
    wdm_channels = int(data.get("wdm_channels", 1))
    include_nl = data.get("include_nonlinear", True)

    # ── Link Analysis ─────────────────────────────────────────────
    link = OpticalLink(
        num_spans=num_spans,
        span_length_km=span_length,
        launch_power_dbm=launch_power,
        edfa_noise_figure_db=nf,
        include_nonlinear=include_nl,
    )
    link_result = link.analyze()

    # ── Performance Metrics ───────────────────────────────────────
    perf = evaluate_performance(
        osnr_db=link_result["osnr_db"],
        bitrate_gbps=bitrate,
        modulation=modulation,
    )

    # ── Max Reach ─────────────────────────────────────────────────
    threshold = get_osnr_threshold(bitrate)
    reach = find_max_reach(
        launch_power_dbm=launch_power,
        span_length_km=span_length,
        target_osnr_db=threshold,
        nf_db=nf,
        include_nonlinear=include_nl,
    )

    # ── Optimal Power ─────────────────────────────────────────────
    opt_power = find_optimal_launch_power(
        num_spans=num_spans,
        span_length_km=span_length,
        nf_db=nf,
        bitrate_gbps=bitrate,
        include_nonlinear=include_nl,
    )

    # ── WDM Analysis (if multi-channel) ───────────────────────────
    wdm_summary = None
    if wdm_channels > 1:
        wdm = WDMSystem(num_channels=wdm_channels)
        wdm.analyze_all_channels({
            "num_spans": num_spans,
            "span_length_km": span_length,
            "launch_power_dbm": launch_power,
            "edfa_noise_figure_db": nf,
            "bitrate_gbps": bitrate,
            "modulation": modulation,
            "include_nonlinear": include_nl,
        })
        wdm_summary = wdm.channel_summary()

    return jsonify({
        "link": link_result,
        "performance": perf,
        "max_reach": reach,
        "optimal_power": opt_power,
        "wdm_summary": wdm_summary,
    })


@app.route("/plot/<plot_type>", methods=["POST"])
def generate_plot(plot_type: str):
    """
    Generate a plot and return as base64 PNG.

    Supported plot_type values:
      - osnr_vs_spans
      - power_sweep
      - ber_curve
      - wdm
    """
    data = request.get_json() or {}

    num_spans = int(data.get("spans", 10))
    span_length = float(data.get("span_length", 80))
    launch_power = float(data.get("launch_power", 0))
    nf = float(data.get("nf", 5.0))
    bitrate = float(data.get("bitrate", 10))
    modulation = data.get("modulation", "OOK")
    wdm_channels = int(data.get("wdm_channels", 1))
    include_nl = data.get("include_nonlinear", True)

    try:
        if plot_type == "osnr_vs_spans":
            fig = plot_osnr_vs_spans(
                span_length_km=span_length,
                launch_power_dbm=launch_power,
                max_spans=max(num_spans + 10, 30),
                bitrate_gbps=bitrate,
                nf_db=nf,
                include_nonlinear=include_nl,
            )
        elif plot_type == "power_sweep":
            fig = plot_launch_power_sweep(
                num_spans=num_spans,
                span_length_km=span_length,
                bitrate_gbps=bitrate,
                nf_db=nf,
                include_nonlinear=include_nl,
            )
        elif plot_type == "ber_curve":
            fig = plot_ber_vs_osnr(bitrate_gbps=bitrate)
        elif plot_type == "wdm":
            wdm = WDMSystem(num_channels=max(wdm_channels, 2))
            results = wdm.analyze_all_channels({
                "num_spans": num_spans,
                "span_length_km": span_length,
                "launch_power_dbm": launch_power,
                "edfa_noise_figure_db": nf,
                "bitrate_gbps": bitrate,
                "modulation": modulation,
                "include_nonlinear": include_nl,
            })
            fig = plot_wdm_channel_osnr(results, bitrate_gbps=bitrate)
        else:
            return jsonify({"error": f"Unknown plot type: {plot_type}"}), 400

        img_b64 = _fig_to_base64(fig)
        return jsonify({"image": img_b64, "plot_type": plot_type})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/max_reach")
def max_reach_route():
    """
    Compute maximum transmission reach.

    Query params: power, span_length, target_osnr, nf, include_nonlinear
    """
    power = float(request.args.get("power", 0))
    span_length = float(request.args.get("span_length", 80))
    target_osnr = float(request.args.get("target_osnr", 15.6))
    nf = float(request.args.get("nf", 5.0))
    include_nl = request.args.get("include_nonlinear", "true").lower() == "true"

    result = find_max_reach(
        launch_power_dbm=power,
        span_length_km=span_length,
        target_osnr_db=target_osnr,
        nf_db=nf,
        include_nonlinear=include_nl,
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
