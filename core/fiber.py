"""
Fiber Span Model
================
Models a single optical fiber span with attenuation, chromatic dispersion,
and nonlinear interference (Kerr effect via simplified GN model).

Typical SMF-28 fiber at 1550 nm:
  - Attenuation:  0.2  dB/km
  - Dispersion:   17.0 ps/(nm·km)
  - Nonlinear coefficient (gamma): 1.3 /W/km

References:
  - ITU-T G.652 (Standard Single-Mode Fiber)
  - Poggiolini, "The GN Model of Non-Linear Propagation" (JLT 2014)
"""

import numpy as np

# ─── Physical Constants ──────────────────────────────────────────────────────
SPEED_OF_LIGHT_M_S = 3.0e8          # Speed of light in vacuum (m/s)
REFERENCE_WAVELENGTH_M = 1550e-9     # C-band center wavelength (m)

# ─── Default Fiber Parameters ────────────────────────────────────────────────
DEFAULT_ATTENUATION_DB_PER_KM = 0.2          # Standard SMF-28 loss at 1550 nm
DEFAULT_DISPERSION_PS_PER_NM_KM = 17.0       # Chromatic dispersion for SMF-28
DEFAULT_NONLINEAR_COEFF_PER_W_PER_KM = 1.3   # Kerr nonlinear coefficient (γ)


class FiberSpan:
    """
    Model a single fiber span with attenuation, dispersion, and nonlinear effects.

    Represents a segment of optical fiber between two amplifier sites.
    Computes signal power loss and nonlinear interference noise
    using the simplified Gaussian Noise (GN) model.

    Parameters
    ----------
    length_km : float
        Span length in kilometers.
    attenuation_db_per_km : float, optional
        Fiber loss coefficient in dB/km (default: 0.2 for SMF-28).
    dispersion_ps_per_nm_km : float, optional
        Chromatic dispersion in ps/(nm·km) (default: 17.0 for SMF-28).
    nonlinear_coefficient : float, optional
        Kerr nonlinear coefficient γ in 1/(W·km) (default: 1.3).
    """

    def __init__(
        self,
        length_km: float,
        attenuation_db_per_km: float = DEFAULT_ATTENUATION_DB_PER_KM,
        dispersion_ps_per_nm_km: float = DEFAULT_DISPERSION_PS_PER_NM_KM,
        nonlinear_coefficient: float = DEFAULT_NONLINEAR_COEFF_PER_W_PER_KM,
    ) -> None:
        if length_km <= 0:
            raise ValueError(f"Span length must be positive, got {length_km} km")
        if attenuation_db_per_km < 0:
            raise ValueError(f"Attenuation cannot be negative, got {attenuation_db_per_km} dB/km")

        self.length_km = length_km
        self.attenuation_db_per_km = attenuation_db_per_km
        self.dispersion_ps_per_nm_km = dispersion_ps_per_nm_km
        self.nonlinear_coefficient = nonlinear_coefficient  # γ in 1/(W·km)

    # ── Loss Methods ──────────────────────────────────────────────────────

    def total_loss_db(self) -> float:
        """
        Total span attenuation in dB.

        Formula: L_dB = α(dB/km) × d(km)
        """
        return self.attenuation_db_per_km * self.length_km

    def total_loss_linear(self) -> float:
        """
        Total span attenuation as a linear ratio (dimensionless).

        Formula: L_linear = 10^(L_dB / 10)
        """
        return 10.0 ** (self.total_loss_db() / 10.0)

    def output_power_dbm(self, input_power_dbm: float) -> float:
        """
        Signal power at the span output in dBm.

        Simply subtracts the total span loss from input power.

        Parameters
        ----------
        input_power_dbm : float
            Input launch power in dBm.

        Returns
        -------
        float
            Output power in dBm.
        """
        return input_power_dbm - self.total_loss_db()

    # ── Dispersion Methods ────────────────────────────────────────────────

    def accumulated_dispersion(self, wavelength_nm: float = 1550.0) -> float:
        """
        Total accumulated chromatic dispersion across the span.

        Formula: D_total = D(ps/nm/km) × L(km)

        Parameters
        ----------
        wavelength_nm : float, optional
            Operating wavelength in nm (for future wavelength-dependent use).

        Returns
        -------
        float
            Accumulated dispersion in ps/nm.
        """
        return self.dispersion_ps_per_nm_km * self.length_km

    # ── Nonlinear Effects (GN Model) ──────────────────────────────────────

    @property
    def _alpha_neper_per_km(self) -> float:
        """Attenuation in Neper/km (for exponential decay calculations)."""
        # α(Np/km) = α(dB/km) × ln(10) / 10
        return self.attenuation_db_per_km * np.log(10) / 10.0

    @property
    def effective_length_km(self) -> float:
        """
        Effective nonlinear interaction length in km.

        For long spans (α·L >> 1), L_eff ≈ 1/α.

        Formula: L_eff = (1 - exp(-α·L)) / α
        """
        alpha = self._alpha_neper_per_km
        if alpha < 1e-12:
            return self.length_km  # Lossless fiber: L_eff = L
        return (1.0 - np.exp(-alpha * self.length_km)) / alpha

    @property
    def beta2_s2_per_km(self) -> float:
        """
        Group velocity dispersion parameter β₂ in s²/km.

        Derived from the dispersion coefficient D:
        β₂ = -D × λ² / (2π·c)

        Note: D in s/(m·m), λ in m, c in m/s → result in s²/m, then ×1e3 for s²/km.
        """
        # Convert D from ps/(nm·km) to SI: s/(m·m)
        # 1 ps/(nm·km) = 1e-12 s / (1e-9 m × 1e3 m) = 1e-6 s/m²
        D_si = self.dispersion_ps_per_nm_km * 1e-6  # s/m²
        lam = REFERENCE_WAVELENGTH_M  # m
        c = SPEED_OF_LIGHT_M_S  # m/s

        # β₂ in s²/m
        beta2_s2_per_m = -D_si * lam ** 2 / (2.0 * np.pi * c)
        # Convert to s²/km (multiply by 1e3 m/km)
        return beta2_s2_per_m * 1e3

    def nonlinear_noise_power_w(
        self,
        launch_power_w: float,
        channel_bandwidth_hz: float = 12.5e9,
    ) -> float:
        """
        Nonlinear interference (NLI) noise power per span using the
        simplified incoherent GN model (single-channel approximation).

        The GN model predicts NLI power spectral density as:
          G_NLI = (8/27) × γ² × L_eff × (P/B)³ × arcsinh(arg) / (π × |β₂|)
          arg   = (π²/2) × |β₂| × L_eff × B²

        Then: P_NLI = G_NLI × B_ref

        All internal calculations are done in SI units (m, s, W).

        Parameters
        ----------
        launch_power_w : float
            Per-channel launch power in watts.
        channel_bandwidth_hz : float, optional
            Channel bandwidth / reference bandwidth in Hz (default: 12.5 GHz).

        Returns
        -------
        float
            NLI noise power in watts for this span.
        """
        if launch_power_w <= 0:
            return 0.0

        # Convert all parameters to SI (m, s, W)
        gamma_per_w_per_m = self.nonlinear_coefficient * 1e-3     # /W/km → /W/m
        L_eff_m = self.effective_length_km * 1e3                   # km → m
        beta2_s2_per_m = self.beta2_s2_per_km * 1e-3               # s²/km → s²/m
        abs_beta2 = abs(beta2_s2_per_m)
        B = channel_bandwidth_hz                                   # Hz

        if abs_beta2 < 1e-35:
            return 0.0  # Zero dispersion: NLI model not applicable

        # Signal power spectral density (W/Hz)
        G_sig = launch_power_w / B

        # Arcsinh argument
        arg = (np.pi ** 2 / 2.0) * abs_beta2 * L_eff_m * B ** 2

        # NLI PSD (W/Hz) — simplified GN model
        G_NLI = (
            (8.0 / 27.0)
            * gamma_per_w_per_m ** 2
            * L_eff_m
            * G_sig ** 3
            * np.arcsinh(arg)
            / (np.pi * abs_beta2)
        )

        # NLI power in reference bandwidth
        P_NLI = G_NLI * B
        return float(P_NLI)

    def __repr__(self) -> str:
        return (
            f"FiberSpan(length_km={self.length_km}, "
            f"attenuation={self.attenuation_db_per_km} dB/km, "
            f"dispersion={self.dispersion_ps_per_nm_km} ps/nm/km, "
            f"γ={self.nonlinear_coefficient} /W/km)"
        )
