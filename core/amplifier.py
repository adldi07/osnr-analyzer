"""
EDFA Amplifier Model
====================
Models an Erbium-Doped Fiber Amplifier (EDFA) with gain and
Amplified Spontaneous Emission (ASE) noise generation.

An EDFA compensates fiber span loss but adds broadband ASE noise
that degrades the optical signal-to-noise ratio (OSNR).

Key formula:
  P_ASE = 2 × n_sp × (G - 1) × h × f × B_o

where:
  - n_sp  = spontaneous emission factor (related to noise figure)
  - G     = amplifier gain (linear)
  - h     = Planck's constant
  - f     = optical center frequency
  - B_o   = optical reference bandwidth (typically 12.5 GHz = 0.1 nm)
  - Factor of 2 accounts for both polarization states of ASE

References:
  - Desurvire, "Erbium-Doped Fiber Amplifiers" (Wiley, 2002)
  - ITU-T G.662 (Optical amplifier parameters)
"""

import numpy as np

# ─── Physical Constants ──────────────────────────────────────────────────────
PLANCK_CONSTANT = 6.62607015e-34     # Planck's constant h (J·s) — exact SI value
SPEED_OF_LIGHT = 3.0e8               # Speed of light c (m/s)

# ─── Default EDFA Parameters ─────────────────────────────────────────────────
DEFAULT_NOISE_FIGURE_DB = 5.0        # Typical commercial EDFA noise figure
DEFAULT_CENTER_FREQUENCY_HZ = 193.4e12  # C-band center ≈ 1550 nm
DEFAULT_OPTICAL_BW_HZ = 12.5e9      # 0.1 nm reference bandwidth at 1550 nm


class EDFA:
    """
    Model an Erbium-Doped Fiber Amplifier with ASE noise generation.

    The EDFA provides optical gain to compensate fiber span loss.
    It also generates ASE noise, which is the dominant noise source
    in long-haul optical links.

    Parameters
    ----------
    gain_db : float
        Amplifier gain in dB. Typically set equal to the preceding
        span loss for loss-compensating operation.
    noise_figure_db : float, optional
        Noise figure in dB (default: 5.0). The quantum limit is 3 dB.
    center_frequency_hz : float, optional
        Center optical frequency in Hz (default: 193.4 THz for 1550 nm).
    """

    def __init__(
        self,
        gain_db: float,
        noise_figure_db: float = DEFAULT_NOISE_FIGURE_DB,
        center_frequency_hz: float = DEFAULT_CENTER_FREQUENCY_HZ,
    ) -> None:
        if gain_db <= 0:
            raise ValueError(f"EDFA gain must be positive, got {gain_db} dB")
        if noise_figure_db < 0:
            raise ValueError(f"Noise figure cannot be negative, got {noise_figure_db} dB")

        self.gain_db = gain_db
        self.noise_figure_db = noise_figure_db
        self.center_frequency_hz = center_frequency_hz

    # ── Gain & Noise Figure (Linear) ─────────────────────────────────────

    @property
    def gain_linear(self) -> float:
        """Amplifier gain as a linear ratio: G = 10^(G_dB / 10)."""
        return 10.0 ** (self.gain_db / 10.0)

    @property
    def nf_linear(self) -> float:
        """Noise figure as a linear ratio: NF = 10^(NF_dB / 10)."""
        return 10.0 ** (self.noise_figure_db / 10.0)

    @property
    def n_sp(self) -> float:
        """
        Spontaneous emission factor (population inversion parameter).

        Derived from the noise figure:
          n_sp = NF_linear / (2 × (1 - 1/G))
               = NF_linear × G / (2 × (G - 1))

        For an ideal amplifier (NF = 3 dB, n_sp = 1), this represents
        complete population inversion.

        The high-gain approximation (G >> 1) gives n_sp ≈ NF/2.
        """
        G = self.gain_linear
        if G <= 1.0:
            # Guard clause for very low gain; avoid division by zero
            return self.nf_linear / 2.0
        return (self.nf_linear * G) / (2.0 * (G - 1.0))

    # ── ASE Noise ─────────────────────────────────────────────────────────

    def ase_power_watts(self, optical_bandwidth_hz: float = DEFAULT_OPTICAL_BW_HZ) -> float:
        """
        ASE noise power in watts within the specified optical bandwidth.

        Includes both polarization states of ASE noise:
          P_ASE = 2 × n_sp × (G - 1) × h × f × B_o

        Parameters
        ----------
        optical_bandwidth_hz : float, optional
            Optical reference bandwidth in Hz (default: 12.5 GHz = 0.1 nm).

        Returns
        -------
        float
            ASE noise power in watts.
        """
        G = self.gain_linear
        h = PLANCK_CONSTANT
        f = self.center_frequency_hz
        B_o = optical_bandwidth_hz

        # P_ASE = 2 × n_sp × (G-1) × h × f × B_o
        # Factor of 2: ASE is unpolarized → noise in both polarization states
        p_ase = 2.0 * self.n_sp * (G - 1.0) * h * f * B_o
        return p_ase

    def ase_power_dbm(self, optical_bandwidth_hz: float = DEFAULT_OPTICAL_BW_HZ) -> float:
        """
        ASE noise power in dBm within the specified optical bandwidth.

        Conversion: P_dBm = 10 × log₁₀(P_watts) + 30
        (The +30 converts from dBW to dBm since 1 mW = 10⁻³ W)

        Parameters
        ----------
        optical_bandwidth_hz : float, optional
            Optical reference bandwidth in Hz (default: 12.5 GHz).

        Returns
        -------
        float
            ASE noise power in dBm.
        """
        p_ase_w = self.ase_power_watts(optical_bandwidth_hz)
        if p_ase_w <= 0:
            return -float("inf")
        # dBm = 10·log₁₀(P_watts / 1e-3) = 10·log₁₀(P_watts) + 30
        return 10.0 * np.log10(p_ase_w) + 30.0

    # ── Output Power ──────────────────────────────────────────────────────

    def output_power_dbm(self, input_power_dbm: float) -> float:
        """
        Amplified signal power at the EDFA output in dBm.

        In dB domain: P_out(dBm) = P_in(dBm) + G(dB)

        Parameters
        ----------
        input_power_dbm : float
            Input signal power in dBm.

        Returns
        -------
        float
            Output signal power in dBm.
        """
        return input_power_dbm + self.gain_db

    def __repr__(self) -> str:
        return (
            f"EDFA(gain={self.gain_db} dB, "
            f"NF={self.noise_figure_db} dB, "
            f"freq={self.center_frequency_hz / 1e12:.1f} THz)"
        )
