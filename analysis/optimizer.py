"""
Optimization Module
===================
Find optimal operating parameters for multi-span optical links.

Two key optimizations:
  1. Optimal launch power — balance ASE noise (low power) vs NLI (high power)
  2. Maximum reach — maximum number of spans before OSNR drops below threshold

Uses scipy.optimize for efficient parameter search.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from core.link import OpticalLink
from core.metrics import get_osnr_threshold


def find_optimal_launch_power(
    num_spans: int,
    span_length_km: float,
    target_osnr_db: float = None,
    nf_db: float = 5.0,
    bitrate_gbps: float = 10.0,
    include_nonlinear: bool = True,
    power_bounds: tuple[float, float] = (-10.0, 10.0),
) -> dict:
    """
    Find the launch power that maximizes OSNR.

    With NLI noise enabled, there exists an optimal launch power
    that balances ASE noise (decreases with power) and NLI noise
    (increases as P³).

    In ASE-only mode, OSNR increases monotonically with power,
    so the optimal is the upper bound.

    Parameters
    ----------
    num_spans : int
        Number of fiber spans.
    span_length_km : float
        Length of each span in km.
    target_osnr_db : float, optional
        Target OSNR in dB. If None, uses bitrate threshold.
    nf_db : float, optional
        EDFA noise figure in dB (default: 5.0).
    bitrate_gbps : float, optional
        Bitrate in Gbps (default: 10).
    include_nonlinear : bool, optional
        Include NLI noise (default: True).
    power_bounds : tuple[float, float], optional
        Search range in dBm (default: -10 to +10).

    Returns
    -------
    dict
        {optimal_power_dbm, achieved_osnr_db, target_osnr_db, margin_db}
    """
    if target_osnr_db is None:
        target_osnr_db = get_osnr_threshold(bitrate_gbps)

    def neg_osnr(power_dbm: float) -> float:
        """Negative OSNR — minimize this to maximize OSNR."""
        link = OpticalLink(
            num_spans=num_spans,
            span_length_km=span_length_km,
            launch_power_dbm=float(power_dbm),
            edfa_noise_figure_db=nf_db,
            include_nonlinear=include_nonlinear,
        )
        result = link.analyze()
        return -result["osnr_db"]

    # Find power that maximizes OSNR (minimizes negative OSNR)
    opt_result = minimize_scalar(
        neg_osnr,
        bounds=power_bounds,
        method="bounded",
    )

    optimal_power = float(opt_result.x)
    achieved_osnr = -float(opt_result.fun)

    return {
        "optimal_power_dbm": round(optimal_power, 2),
        "achieved_osnr_db": round(achieved_osnr, 2),
        "target_osnr_db": target_osnr_db,
        "margin_db": round(achieved_osnr - target_osnr_db, 2),
    }


def find_max_reach(
    launch_power_dbm: float,
    span_length_km: float,
    target_osnr_db: float = 15.6,
    nf_db: float = 5.0,
    max_spans_search: int = 200,
    include_nonlinear: bool = True,
) -> dict:
    """
    Find the maximum number of spans before OSNR drops below threshold.

    Iterates span count from 1 until OSNR falls below the target.

    Parameters
    ----------
    launch_power_dbm : float
        Per-channel launch power in dBm.
    span_length_km : float
        Length of each span in km.
    target_osnr_db : float, optional
        Minimum acceptable OSNR in dB (default: 15.6 for 10G).
    nf_db : float, optional
        EDFA noise figure in dB (default: 5.0).
    max_spans_search : int, optional
        Maximum spans to check (default: 200).
    include_nonlinear : bool, optional
        Include NLI noise (default: True).

    Returns
    -------
    dict
        {max_spans, max_distance_km, final_osnr_db, target_osnr_db}
    """
    max_spans = 0
    final_osnr = 0.0

    for n in range(1, max_spans_search + 1):
        link = OpticalLink(
            num_spans=n,
            span_length_km=span_length_km,
            launch_power_dbm=launch_power_dbm,
            edfa_noise_figure_db=nf_db,
            include_nonlinear=include_nonlinear,
        )
        result = link.analyze()
        if result["osnr_db"] >= target_osnr_db:
            max_spans = n
            final_osnr = result["osnr_db"]
        else:
            break

    return {
        "max_spans": max_spans,
        "max_distance_km": max_spans * span_length_km,
        "final_osnr_db": round(final_osnr, 2),
        "target_osnr_db": target_osnr_db,
    }
