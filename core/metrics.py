"""
Performance Metrics Module
==========================
Computes Q-factor, BER, and OSNR margin for optical links.

Converts between OSNR (optical domain) and BER (electrical domain)
using the standard Q-factor relationship:

  Q = √(2 × OSNR × B_o / B_e)
  BER = 0.5 × erfc(Q / √2)

where:
  - B_o = optical reference bandwidth (12.5 GHz = 0.1 nm)
  - B_e = electrical bandwidth = bitrate / 2  (for NRZ modulation)

Modulation format penalties are applied as OSNR offsets.

References:
  - ITU-T G.Sup39 (Optical system design)
  - Winzer & Essiambre, "Advanced Modulation Formats" (JLT 2006)
"""

import numpy as np
from scipy.special import erfc

# ─── Modulation Format Definitions ───────────────────────────────────────────
# Each format has bits_per_symbol and an OSNR penalty (dB) relative to OOK
MODULATION_FORMATS: dict[str, dict] = {
    "OOK": {
        "bits_per_symbol": 1,
        "osnr_penalty_db": 0.0,
        "description": "On-Off Keying (direct detection)",
    },
    "QPSK": {
        "bits_per_symbol": 2,
        "osnr_penalty_db": 1.0,
        "description": "Quadrature Phase Shift Keying (coherent)",
    },
    "16QAM": {
        "bits_per_symbol": 4,
        "osnr_penalty_db": 3.5,
        "description": "16-Quadrature Amplitude Modulation (coherent)",
    },
}

# ─── OSNR Thresholds by Bitrate ─────────────────────────────────────────────
# Minimum required OSNR (dB) for BER < 1e-9 (pre-FEC) at each line rate
OSNR_THRESHOLDS_DB: dict[float, float] = {
    10.0: 15.6,    # 10 Gbps (OOK, direct detection)
    40.0: 18.0,    # 40 Gbps (DPSK/DQPSK)
    100.0: 20.0,   # 100 Gbps (DP-QPSK, coherent)
}

# ─── Default Parameters ─────────────────────────────────────────────────────
DEFAULT_OPTICAL_BW_HZ = 12.5e9      # 0.1 nm reference bandwidth at 1550 nm
DEFAULT_BITRATE_GBPS = 10.0          # 10 Gbps base rate


def osnr_to_q_factor(
    osnr_linear: float,
    optical_bw_hz: float = DEFAULT_OPTICAL_BW_HZ,
    electrical_bw_hz: float = None,
    bitrate_gbps: float = DEFAULT_BITRATE_GBPS,
) -> float:
    """
    Convert optical OSNR to electrical Q-factor.

    Formula: Q = √(2 × OSNR × B_o / B_e)

    This assumes matched filtering and NRZ modulation.

    Parameters
    ----------
    osnr_linear : float
        OSNR in linear scale (not dB).
    optical_bw_hz : float, optional
        Optical reference bandwidth in Hz (default: 12.5 GHz).
    electrical_bw_hz : float, optional
        Electrical bandwidth in Hz. If None, derived from bitrate.
    bitrate_gbps : float, optional
        Bitrate in Gbps (default: 10). Used if electrical_bw_hz is None.

    Returns
    -------
    float
        Q-factor (dimensionless).
    """
    if electrical_bw_hz is None:
        electrical_bw_hz = get_electrical_bandwidth(bitrate_gbps)

    if electrical_bw_hz <= 0 or osnr_linear <= 0:
        return 0.0

    # Q = √(2 × OSNR × B_o / B_e)
    q = np.sqrt(2.0 * osnr_linear * optical_bw_hz / electrical_bw_hz)
    return float(q)


def q_to_ber(q_factor: float) -> float:
    """
    Convert Q-factor to Bit Error Rate (BER).

    Formula: BER = 0.5 × erfc(Q / √2)

    Parameters
    ----------
    q_factor : float
        Q-factor (dimensionless).

    Returns
    -------
    float
        Bit error rate (0 to 0.5).
    """
    if q_factor <= 0:
        return 0.5  # Worst case: random guessing
    return float(0.5 * erfc(q_factor / np.sqrt(2.0)))


def get_electrical_bandwidth(bitrate_gbps: float) -> float:
    """
    Electrical receiver bandwidth for NRZ modulation.

    Formula: B_e = bitrate / 2  (Nyquist bandwidth)

    Parameters
    ----------
    bitrate_gbps : float
        Data bitrate in Gbps.

    Returns
    -------
    float
        Electrical bandwidth in Hz.
    """
    return bitrate_gbps * 1e9 / 2.0


def get_osnr_threshold(bitrate_gbps: float) -> float:
    """
    Get the minimum required OSNR (dB) for the given bitrate.

    Uses predefined thresholds for standard line rates.
    For non-standard rates, interpolates between nearest values.

    Parameters
    ----------
    bitrate_gbps : float
        Data bitrate in Gbps.

    Returns
    -------
    float
        Required OSNR threshold in dB.
    """
    if bitrate_gbps in OSNR_THRESHOLDS_DB:
        return OSNR_THRESHOLDS_DB[bitrate_gbps]

    # Interpolate for non-standard bitrates
    rates = sorted(OSNR_THRESHOLDS_DB.keys())
    thresholds = [OSNR_THRESHOLDS_DB[r] for r in rates]

    if bitrate_gbps < rates[0]:
        return thresholds[0]
    if bitrate_gbps > rates[-1]:
        return thresholds[-1]

    return float(np.interp(bitrate_gbps, rates, thresholds))


def evaluate_performance(
    osnr_db: float,
    bitrate_gbps: float = DEFAULT_BITRATE_GBPS,
    modulation: str = "OOK",
    optical_bw_hz: float = DEFAULT_OPTICAL_BW_HZ,
) -> dict:
    """
    Evaluate link performance from OSNR.

    Computes Q-factor, BER, OSNR margin, and pass/fail status
    for the specified modulation format and bitrate.

    Parameters
    ----------
    osnr_db : float
        Optical signal-to-noise ratio in dB.
    bitrate_gbps : float, optional
        Data bitrate in Gbps (default: 10).
    modulation : str, optional
        Modulation format: "OOK", "QPSK", "16QAM" (default: "OOK").
    optical_bw_hz : float, optional
        Optical reference bandwidth in Hz (default: 12.5 GHz).

    Returns
    -------
    dict
        Performance report with keys:
        osnr_db, q_factor, ber, ber_scientific, margin_db,
        pass, bitrate_gbps, modulation
    """
    # Validate modulation format
    if modulation not in MODULATION_FORMATS:
        raise ValueError(
            f"Unknown modulation '{modulation}'. "
            f"Supported: {list(MODULATION_FORMATS.keys())}"
        )

    mod_info = MODULATION_FORMATS[modulation]

    # Apply modulation penalty to effective OSNR
    effective_osnr_db = osnr_db - mod_info["osnr_penalty_db"]
    effective_osnr_linear = 10.0 ** (effective_osnr_db / 10.0)

    # Compute Q-factor and BER
    B_e = get_electrical_bandwidth(bitrate_gbps)
    q_factor = osnr_to_q_factor(effective_osnr_linear, optical_bw_hz, B_e)
    ber = q_to_ber(q_factor)

    # OSNR margin = actual OSNR − required threshold
    threshold_db = get_osnr_threshold(bitrate_gbps)
    margin_db = osnr_db - threshold_db

    # Format BER in scientific notation
    if ber > 0:
        exponent = int(np.floor(np.log10(ber))) if ber > 0 else 0
        mantissa = ber / (10.0 ** exponent) if exponent != 0 else ber
        ber_scientific = f"{mantissa:.2f}e{exponent}"
    else:
        ber_scientific = "0.00e+00"

    # Pass/fail: BER < 1e-9 AND positive OSNR margin
    is_pass = (ber < 1e-9) and (margin_db >= 0)

    return {
        "osnr_db": round(osnr_db, 2),
        "q_factor": round(q_factor, 2),
        "ber": float(ber),
        "ber_scientific": ber_scientific,
        "margin_db": round(margin_db, 2),
        "pass": is_pass,
        "bitrate_gbps": bitrate_gbps,
        "modulation": modulation,
        "threshold_db": threshold_db,
    }
