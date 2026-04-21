"""
Plotting Module
===============
Generates publication-quality matplotlib figures for optical link analysis.

Four plot types:
  1. OSNR vs Spans — reach limitation visualization
  2. Launch Power Sweep — optimal power identification
  3. BER vs OSNR — waterfall curve
  4. WDM Channel OSNR — per-channel bar chart

All plots use consistent styling:
  - figsize=(10, 6), fontsize=12, grid=True, tight_layout
  - Title includes key configuration parameters
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for web/headless rendering
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from analysis.sweep import osnr_vs_spans, osnr_vs_launch_power
from analysis.optimizer import find_max_reach
from core.metrics import (
    evaluate_performance,
    get_osnr_threshold,
    osnr_to_q_factor,
    q_to_ber,
    OSNR_THRESHOLDS_DB,
)

# ─── Plot Style Constants ────────────────────────────────────────────────────
PRIMARY_BLUE = "#124191"
ACCENT_GREEN = "#00c853"
ACCENT_RED = "#ff1744"
BG_DARK = "#0d1117"
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#21262d"
FONT_SIZE = 12


def _apply_dark_theme(fig: Figure, ax) -> None:
    """Apply consistent dark theme to figure and axes."""
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor("#161b22")
    ax.tick_params(colors=TEXT_COLOR, labelsize=FONT_SIZE - 1)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.spines["top"].set_color(GRID_COLOR)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["right"].set_color(GRID_COLOR)
    ax.grid(True, alpha=0.3, color=GRID_COLOR)


def plot_osnr_vs_spans(
    span_length_km: float = 80.0,
    launch_power_dbm: float = 0.0,
    max_spans: int = 30,
    bitrate_gbps: float = 10.0,
    nf_db: float = 5.0,
    include_nonlinear: bool = True,
) -> Figure:
    """
    Plot OSNR vs number of spans with threshold and max reach annotations.

    Features:
      - Blue line: OSNR curve
      - Red dashed: OSNR threshold for target BER
      - Green vertical: max reach point
      - Distance annotation on max reach

    Parameters
    ----------
    span_length_km : float
        Span length in km.
    launch_power_dbm : float
        Launch power in dBm.
    max_spans : int
        Maximum spans to plot.
    bitrate_gbps : float
        Bitrate in Gbps.
    nf_db : float
        EDFA noise figure in dB.
    include_nonlinear : bool
        Include NLI noise.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    # Run sweep
    data = osnr_vs_spans(
        span_length_km=span_length_km,
        launch_power_dbm=launch_power_dbm,
        max_spans=max_spans,
        bitrate_gbps=bitrate_gbps,
        nf_db=nf_db,
        include_nonlinear=include_nonlinear,
    )

    # Get threshold and max reach
    threshold = get_osnr_threshold(bitrate_gbps)
    reach = find_max_reach(
        launch_power_dbm=launch_power_dbm,
        span_length_km=span_length_km,
        target_osnr_db=threshold,
        nf_db=nf_db,
        include_nonlinear=include_nonlinear,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_theme(fig, ax)

    # OSNR curve
    ax.plot(data["spans"], data["osnr"], color=PRIMARY_BLUE, linewidth=2.5,
            marker="o", markersize=4, label="OSNR (dB)")

    # Threshold line
    ax.axhline(y=threshold, color=ACCENT_RED, linestyle="--", linewidth=1.5,
               label=f"Threshold ({threshold} dB @ {bitrate_gbps}G)")

    # Max reach annotation
    if reach["max_spans"] > 0:
        ax.axvline(x=reach["max_spans"], color=ACCENT_GREEN, linestyle="--",
                   linewidth=1.5, alpha=0.8, label=f"Max Reach: {reach['max_spans']} spans")
        ax.annotate(
            f"Max Reach\n{reach['max_distance_km']:.0f} km",
            xy=(reach["max_spans"], threshold),
            xytext=(reach["max_spans"] + max_spans * 0.05, threshold + 2),
            fontsize=FONT_SIZE - 1,
            color=ACCENT_GREEN,
            arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22",
                      edgecolor=ACCENT_GREEN, alpha=0.9),
        )

    ax.set_xlabel("Number of Spans", fontsize=FONT_SIZE)
    ax.set_ylabel("OSNR (dB)", fontsize=FONT_SIZE)
    nli_label = " [ASE+NLI]" if include_nonlinear else " [ASE only]"
    ax.set_title(
        f"OSNR vs Spans — {span_length_km} km spans, "
        f"{launch_power_dbm} dBm, NF={nf_db} dB{nli_label}",
        fontsize=FONT_SIZE + 1, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=FONT_SIZE - 1,
              facecolor="#161b22", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    fig.tight_layout()
    return fig


def plot_launch_power_sweep(
    num_spans: int = 10,
    span_length_km: float = 80.0,
    bitrate_gbps: float = 10.0,
    nf_db: float = 5.0,
    include_nonlinear: bool = True,
) -> Figure:
    """
    Plot OSNR vs launch power with nonlinear regime shading.

    Features:
      - Blue line: OSNR curve (inverted-U with NLI)
      - Red shaded region: nonlinear regime (> +4 dBm)
      - Green marker: optimal launch power
      - Red dashed: OSNR threshold

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    data = osnr_vs_launch_power(
        num_spans=num_spans,
        span_length_km=span_length_km,
        nf_db=nf_db,
        include_nonlinear=include_nonlinear,
    )

    # Find optimal power point
    max_osnr_idx = int(np.argmax(data["osnr"]))
    optimal_power = data["powers"][max_osnr_idx]
    optimal_osnr = data["osnr"][max_osnr_idx]

    threshold = get_osnr_threshold(bitrate_gbps)

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_theme(fig, ax)

    # OSNR curve
    ax.plot(data["powers"], data["osnr"], color=PRIMARY_BLUE, linewidth=2.5,
            label="OSNR (dB)")

    # Nonlinear regime shading (> +4 dBm)
    ax.axvspan(4.0, max(data["powers"]), alpha=0.15, color=ACCENT_RED,
               label="Nonlinear Regime (> +4 dBm)")

    # Threshold line
    ax.axhline(y=threshold, color=ACCENT_RED, linestyle="--", linewidth=1.5,
               label=f"Threshold ({threshold} dB)")

    # Optimal power marker
    ax.plot(optimal_power, optimal_osnr, "o", color=ACCENT_GREEN,
            markersize=10, zorder=5)
    ax.annotate(
        f"Optimal: {optimal_power:.1f} dBm\nOSNR: {optimal_osnr:.1f} dB",
        xy=(optimal_power, optimal_osnr),
        xytext=(optimal_power - 3, optimal_osnr + 1.5),
        fontsize=FONT_SIZE - 1,
        color=ACCENT_GREEN,
        arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22",
                  edgecolor=ACCENT_GREEN, alpha=0.9),
    )

    ax.set_xlabel("Launch Power (dBm)", fontsize=FONT_SIZE)
    ax.set_ylabel("OSNR (dB)", fontsize=FONT_SIZE)
    nli_label = " [ASE+NLI]" if include_nonlinear else " [ASE only]"
    ax.set_title(
        f"OSNR vs Launch Power — {num_spans}×{span_length_km} km, "
        f"NF={nf_db} dB{nli_label}",
        fontsize=FONT_SIZE + 1, fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=FONT_SIZE - 1,
              facecolor="#161b22", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    fig.tight_layout()
    return fig


def plot_ber_vs_osnr(
    bitrate_gbps: float = 10.0,
    osnr_range: tuple[float, float] = (5.0, 30.0),
    steps: int = 200,
) -> Figure:
    """
    Plot BER vs OSNR waterfall curve (semilogy).

    Features:
      - Blue curve: BER vs OSNR
      - Red dashed: BER = 1e-9 threshold
      - Clearly labeled axes with scientific notation

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    osnr_db_values = np.linspace(osnr_range[0], osnr_range[1], steps)
    ber_values = []

    for osnr_db in osnr_db_values:
        osnr_linear = 10.0 ** (osnr_db / 10.0)
        q = osnr_to_q_factor(osnr_linear, bitrate_gbps=bitrate_gbps)
        ber = q_to_ber(q)
        ber_values.append(max(ber, 1e-40))  # Floor for log scale

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_theme(fig, ax)

    # BER curve (semilogy)
    ax.semilogy(osnr_db_values, ber_values, color=PRIMARY_BLUE, linewidth=2.5,
                label=f"BER @ {bitrate_gbps} Gbps")

    # BER = 1e-9 threshold
    ax.axhline(y=1e-9, color=ACCENT_RED, linestyle="--", linewidth=1.5,
               label="BER = 1×10⁻⁹ threshold")

    # Find OSNR at BER = 1e-9
    ber_array = np.array(ber_values)
    threshold_idx = np.argmin(np.abs(ber_array - 1e-9))
    threshold_osnr = osnr_db_values[threshold_idx]
    ax.axvline(x=threshold_osnr, color=ACCENT_GREEN, linestyle=":", linewidth=1,
               alpha=0.7)
    ax.annotate(
        f"OSNR = {threshold_osnr:.1f} dB\nat BER = 10⁻⁹",
        xy=(threshold_osnr, 1e-9),
        xytext=(threshold_osnr + 3, 1e-6),
        fontsize=FONT_SIZE - 1,
        color=ACCENT_GREEN,
        arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22",
                  edgecolor=ACCENT_GREEN, alpha=0.9),
    )

    ax.set_xlabel("OSNR (dB)", fontsize=FONT_SIZE)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=FONT_SIZE)
    ax.set_title(
        f"BER vs OSNR Waterfall — {bitrate_gbps} Gbps",
        fontsize=FONT_SIZE + 1, fontweight="bold",
    )
    ax.set_ylim(1e-15, 1)
    ax.legend(loc="upper right", fontsize=FONT_SIZE - 1,
              facecolor="#161b22", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    fig.tight_layout()
    return fig


def plot_wdm_channel_osnr(
    wdm_results: list[dict],
    threshold_db: float = None,
    bitrate_gbps: float = 10.0,
) -> Figure:
    """
    Plot per-channel OSNR as a bar chart for all WDM channels.

    Features:
      - Green bars: channels above threshold (PASS)
      - Red bars: channels below threshold (FAIL)
      - Red dashed: OSNR threshold line
      - Wavelength labels on x-axis

    Parameters
    ----------
    wdm_results : list[dict]
        Per-channel results from WDMSystem.analyze_all_channels().
    threshold_db : float, optional
        OSNR threshold. If None, derived from bitrate.
    bitrate_gbps : float, optional
        Bitrate in Gbps (default: 10).

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    if threshold_db is None:
        threshold_db = get_osnr_threshold(bitrate_gbps)

    channels = [r["channel_index"] for r in wdm_results]
    osnr_values = [r["osnr_db"] for r in wdm_results]
    wavelengths = [r.get("wavelength_nm", 0) for r in wdm_results]

    # Color bars based on pass/fail
    colors = [ACCENT_GREEN if o >= threshold_db else ACCENT_RED for o in osnr_values]

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_theme(fig, ax)

    ax.bar(channels, osnr_values, color=colors, alpha=0.85, edgecolor="none",
           width=0.8)

    # Threshold line
    ax.axhline(y=threshold_db, color=ACCENT_RED, linestyle="--", linewidth=1.5,
               label=f"Threshold ({threshold_db} dB)")

    # X-axis labels: show wavelength for every Nth channel
    num_ch = len(channels)
    if num_ch <= 20:
        tick_step = 1
    elif num_ch <= 50:
        tick_step = 5
    else:
        tick_step = 10

    tick_positions = list(range(0, num_ch, tick_step))
    tick_labels = [f"{wavelengths[i]:.1f}" for i in tick_positions]
    ax.set_xticks([channels[i] for i in tick_positions])
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=FONT_SIZE - 2)

    ax.set_xlabel("Channel Wavelength (nm)", fontsize=FONT_SIZE)
    ax.set_ylabel("OSNR (dB)", fontsize=FONT_SIZE)
    ax.set_title(
        f"WDM Channel OSNR — {num_ch} Channels",
        fontsize=FONT_SIZE + 1, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=FONT_SIZE - 1,
              facecolor="#161b22", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    fig.tight_layout()
    return fig
