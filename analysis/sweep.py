"""
Parameter Sweep Module
======================
Performs systematic parameter sweeps to characterize link performance
across different operating conditions.

Three sweep types:
  1. OSNR vs Number of Spans — shows reach limitation
  2. OSNR vs Launch Power — shows optimal power point (with NLI)
  3. OSNR vs Span Length — shows impact of span engineering

Each function returns a dict with arrays suitable for plotting.
"""

from core.link import OpticalLink
from core.metrics import evaluate_performance


def osnr_vs_spans(
    span_length_km: float,
    launch_power_dbm: float,
    max_spans: int = 30,
    bitrate_gbps: float = 10.0,
    modulation: str = "OOK",
    nf_db: float = 5.0,
    include_nonlinear: bool = True,
) -> dict:
    """
    Sweep OSNR as a function of the number of spans.

    Shows how OSNR degrades with increasing transmission distance
    as ASE (and NLI) noise accumulates.

    Parameters
    ----------
    span_length_km : float
        Length of each fiber span in km.
    launch_power_dbm : float
        Per-channel launch power in dBm.
    max_spans : int, optional
        Maximum number of spans to simulate (default: 30).
    bitrate_gbps : float, optional
        Bitrate in Gbps (default: 10).
    modulation : str, optional
        Modulation format (default: "OOK").
    nf_db : float, optional
        EDFA noise figure in dB (default: 5.0).
    include_nonlinear : bool, optional
        Include GN-model NLI noise (default: True).

    Returns
    -------
    dict
        {spans: list[int], osnr: list[float], distances: list[float],
         performances: list[dict]}
    """
    spans_list = list(range(1, max_spans + 1))
    osnr_list = []
    distances = []
    performances = []

    for n in spans_list:
        link = OpticalLink(
            num_spans=n,
            span_length_km=span_length_km,
            launch_power_dbm=launch_power_dbm,
            edfa_noise_figure_db=nf_db,
            include_nonlinear=include_nonlinear,
        )
        result = link.analyze()
        perf = evaluate_performance(
            osnr_db=result["osnr_db"],
            bitrate_gbps=bitrate_gbps,
            modulation=modulation,
        )
        osnr_list.append(result["osnr_db"])
        distances.append(n * span_length_km)
        performances.append(perf)

    return {
        "spans": spans_list,
        "osnr": osnr_list,
        "distances": distances,
        "performances": performances,
    }


def osnr_vs_launch_power(
    num_spans: int,
    span_length_km: float,
    power_range: tuple[float, float] = (-5.0, 8.0),
    steps: int = 50,
    nf_db: float = 5.0,
    include_nonlinear: bool = True,
) -> dict:
    """
    Sweep OSNR as a function of launch power.

    In the ASE-limited regime (low power), OSNR increases linearly
    with launch power. With NLI enabled, OSNR peaks at an optimal
    power and then decreases (inverted-U curve).

    Parameters
    ----------
    num_spans : int
        Number of fiber spans.
    span_length_km : float
        Length of each span in km.
    power_range : tuple[float, float], optional
        (min_dBm, max_dBm) range to sweep (default: -5 to +8).
    steps : int, optional
        Number of power steps (default: 50).
    nf_db : float, optional
        EDFA noise figure in dB (default: 5.0).
    include_nonlinear : bool, optional
        Include GN-model NLI noise (default: True).

    Returns
    -------
    dict
        {powers: list[float], osnr: list[float]}
    """
    import numpy as np

    powers = list(np.linspace(power_range[0], power_range[1], steps))
    osnr_list = []

    for p in powers:
        link = OpticalLink(
            num_spans=num_spans,
            span_length_km=span_length_km,
            launch_power_dbm=p,
            edfa_noise_figure_db=nf_db,
            include_nonlinear=include_nonlinear,
        )
        result = link.analyze()
        osnr_list.append(result["osnr_db"])

    return {
        "powers": powers,
        "osnr": osnr_list,
    }


def osnr_vs_span_length(
    num_spans: int,
    launch_power_dbm: float,
    length_range: tuple[float, float] = (40.0, 120.0),
    steps: int = 20,
    nf_db: float = 5.0,
    include_nonlinear: bool = True,
) -> dict:
    """
    Sweep OSNR as a function of span length.

    Longer spans have higher loss, requiring higher EDFA gain,
    which produces more ASE noise per amplifier.

    Parameters
    ----------
    num_spans : int
        Number of fiber spans.
    launch_power_dbm : float
        Per-channel launch power in dBm.
    length_range : tuple[float, float], optional
        (min_km, max_km) range to sweep (default: 40 to 120).
    steps : int, optional
        Number of length steps (default: 20).
    nf_db : float, optional
        EDFA noise figure in dB (default: 5.0).
    include_nonlinear : bool, optional
        Include GN-model NLI noise (default: True).

    Returns
    -------
    dict
        {lengths: list[float], osnr: list[float]}
    """
    import numpy as np

    lengths = list(np.linspace(length_range[0], length_range[1], steps))
    osnr_list = []

    for length in lengths:
        link = OpticalLink(
            num_spans=num_spans,
            span_length_km=length,
            launch_power_dbm=launch_power_dbm,
            edfa_noise_figure_db=nf_db,
            include_nonlinear=include_nonlinear,
        )
        result = link.analyze()
        osnr_list.append(result["osnr_db"])

    return {
        "lengths": lengths,
        "osnr": osnr_list,
    }
