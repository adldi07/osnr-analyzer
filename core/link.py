"""
Optical Link Model
==================
Models a complete multi-span amplified optical fiber link:

    TX → [FiberSpan → EDFA] × N → RX

Each EDFA operates in loss-compensating mode (gain = span loss),
so the signal power at each amplifier output equals the launch power.

Noise sources modeled:
  1. ASE noise — accumulates linearly with the number of EDFAs
  2. NLI noise — nonlinear interference from Kerr effect (GN model),
     grows as P_launch³, creating an optimal launch power

OSNR = P_signal / (P_ASE_total + P_NLI_total)

References:
  - Agrawal, "Fiber-Optic Communication Systems" (Wiley, 5th ed.)
  - Poggiolini, JLT 2014 (GN model)
"""

import numpy as np

from core.fiber import FiberSpan
from core.amplifier import EDFA


# ─── Conversion Utilities ────────────────────────────────────────────────────

def dbm_to_watts(power_dbm: float) -> float:
    """Convert power from dBm to watts: P_W = 10^((P_dBm - 30) / 10)."""
    return 10.0 ** ((power_dbm - 30.0) / 10.0)


def watts_to_dbm(power_watts: float) -> float:
    """Convert power from watts to dBm: P_dBm = 10·log₁₀(P_W) + 30."""
    if power_watts <= 0:
        return -float("inf")
    return 10.0 * np.log10(power_watts) + 30.0


class OpticalLink:
    """
    Model a complete multi-span amplified fiber link.

    Architecture: TX → [FiberSpan → EDFA]×N → RX

    Each EDFA gain is set equal to the preceding span loss
    (loss-compensating operation), so signal power is restored
    at each amplifier output.

    Parameters
    ----------
    num_spans : int
        Number of fiber spans (and EDFAs) in the link.
    span_length_km : float
        Length of each fiber span in km.
    launch_power_dbm : float
        Per-channel transmitter launch power in dBm.
    attenuation_db_per_km : float, optional
        Fiber attenuation in dB/km (default: 0.2).
    edfa_noise_figure_db : float, optional
        EDFA noise figure in dB (default: 5.0).
    optical_bw_hz : float, optional
        Optical reference bandwidth in Hz (default: 12.5 GHz = 0.1 nm).
    include_nonlinear : bool, optional
        Whether to include GN-model nonlinear noise (default: True).
    nonlinear_coefficient : float, optional
        Fiber nonlinear coefficient γ in 1/(W·km) (default: 1.3).
    """

    def __init__(
        self,
        num_spans: int,
        span_length_km: float,
        launch_power_dbm: float,
        attenuation_db_per_km: float = 0.2,
        edfa_noise_figure_db: float = 5.0,
        optical_bw_hz: float = 12.5e9,
        include_nonlinear: bool = True,
        nonlinear_coefficient: float = 1.3,
    ) -> None:
        if num_spans < 1:
            raise ValueError(f"Number of spans must be ≥ 1, got {num_spans}")
        if span_length_km <= 0:
            raise ValueError(f"Span length must be positive, got {span_length_km} km")

        self.num_spans = num_spans
        self.span_length_km = span_length_km
        self.launch_power_dbm = launch_power_dbm
        self.optical_bw_hz = optical_bw_hz
        self.include_nonlinear = include_nonlinear

        # Create fiber span and EDFA models
        self.fiber = FiberSpan(
            length_km=span_length_km,
            attenuation_db_per_km=attenuation_db_per_km,
            nonlinear_coefficient=nonlinear_coefficient,
        )

        # EDFA gain = span loss (loss-compensating mode)
        span_loss_db = self.fiber.total_loss_db()
        self.edfa = EDFA(
            gain_db=span_loss_db,
            noise_figure_db=edfa_noise_figure_db,
        )

    def analyze(self) -> dict:
        """
        Perform complete link analysis.

        Computes signal power, ASE noise, NLI noise, and OSNR
        for the entire multi-span link.

        Returns
        -------
        dict
            Analysis results with keys:
            - num_spans, total_length_km, launch_power_dbm
            - signal_power_at_rx_dbm
            - ase_per_edfa_dbm, total_ase_dbm
            - nli_per_span_dbm, total_nli_dbm (if nonlinear enabled)
            - total_noise_dbm
            - osnr_linear, osnr_db
        """
        N = self.num_spans

        # ── Signal Power ──────────────────────────────────────────────
        # In loss-compensating mode, signal power at receiver = launch power
        signal_power_dbm = self.launch_power_dbm
        signal_power_w = dbm_to_watts(signal_power_dbm)

        # ── ASE Noise Accumulation ───────────────────────────────────
        # Each EDFA contributes identical ASE noise
        ase_per_edfa_w = self.edfa.ase_power_watts(self.optical_bw_hz)
        total_ase_w = N * ase_per_edfa_w

        # ── NLI Noise (GN Model) ─────────────────────────────────────
        # Nonlinear noise grows as P³ and accumulates per span
        if self.include_nonlinear:
            nli_per_span_w = self.fiber.nonlinear_noise_power_w(
                launch_power_w=signal_power_w,
                channel_bandwidth_hz=self.optical_bw_hz,
            )
            total_nli_w = N * nli_per_span_w
        else:
            nli_per_span_w = 0.0
            total_nli_w = 0.0

        # ── Total Noise & OSNR ────────────────────────────────────────
        total_noise_w = total_ase_w + total_nli_w
        osnr_linear = signal_power_w / total_noise_w if total_noise_w > 0 else float("inf")
        osnr_db = 10.0 * np.log10(osnr_linear) if osnr_linear > 0 else -float("inf")

        return {
            "num_spans": N,
            "total_length_km": N * self.span_length_km,
            "launch_power_dbm": self.launch_power_dbm,
            "signal_power_at_rx_dbm": signal_power_dbm,
            "ase_per_edfa_dbm": watts_to_dbm(ase_per_edfa_w),
            "total_ase_dbm": watts_to_dbm(total_ase_w),
            "nli_per_span_dbm": watts_to_dbm(nli_per_span_w),
            "total_nli_dbm": watts_to_dbm(total_nli_w),
            "total_noise_dbm": watts_to_dbm(total_noise_w),
            "osnr_linear": float(osnr_linear),
            "osnr_db": float(osnr_db),
            "include_nonlinear": self.include_nonlinear,
        }

    def __repr__(self) -> str:
        return (
            f"OpticalLink(spans={self.num_spans}, "
            f"span_length={self.span_length_km} km, "
            f"launch_power={self.launch_power_dbm} dBm, "
            f"NLI={'ON' if self.include_nonlinear else 'OFF'})"
        )
