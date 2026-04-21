"""
WDM System Model
================
Models a multi-channel Wavelength Division Multiplexing (WDM) system
across the C-band (1530–1565 nm).

Simulates EDFA gain tilt (non-uniform gain across channels) and
evaluates per-channel performance. Edge channels typically see
slightly worse OSNR due to gain non-uniformity.

Typical C-band WDM parameters:
  - 40 channels at 100 GHz spacing
  - 80 channels at 50 GHz spacing
  - Center wavelength: 1550 nm

References:
  - ITU-T G.694.1 (DWDM frequency grid)
  - Desurvire, "Erbium-Doped Fiber Amplifiers" (Wiley, 2002)
"""

from core.link import OpticalLink
from core.metrics import evaluate_performance, get_osnr_threshold

# ─── Physical Constants ──────────────────────────────────────────────────────
SPEED_OF_LIGHT_M_S = 3.0e8          # Speed of light (m/s)

# ─── Default WDM Parameters ─────────────────────────────────────────────────
DEFAULT_NUM_CHANNELS = 40
DEFAULT_CHANNEL_SPACING_GHZ = 100.0
DEFAULT_CENTER_WAVELENGTH_NM = 1550.0
DEFAULT_GAIN_TILT_DB_PER_CH = 0.05   # OSNR penalty per channel from center


class WDMSystem:
    """
    Multi-channel WDM system analysis.

    Evaluates performance across all WDM channels, accounting for
    EDFA gain tilt that degrades edge channel performance.

    Parameters
    ----------
    num_channels : int, optional
        Number of WDM channels (default: 40).
    channel_spacing_ghz : float, optional
        Channel spacing in GHz (default: 100).
    center_wavelength_nm : float, optional
        Center wavelength of the WDM grid in nm (default: 1550).
    edfa_gain_tilt_db_per_channel : float, optional
        OSNR penalty per channel position from center (default: 0.05 dB).
        Simulates EDFA gain non-uniformity across the C-band.
    """

    def __init__(
        self,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        channel_spacing_ghz: float = DEFAULT_CHANNEL_SPACING_GHZ,
        center_wavelength_nm: float = DEFAULT_CENTER_WAVELENGTH_NM,
        edfa_gain_tilt_db_per_channel: float = DEFAULT_GAIN_TILT_DB_PER_CH,
    ) -> None:
        if num_channels < 1:
            raise ValueError(f"Number of channels must be ≥ 1, got {num_channels}")

        self.num_channels = num_channels
        self.channel_spacing_ghz = channel_spacing_ghz
        self.center_wavelength_nm = center_wavelength_nm
        self.edfa_gain_tilt_db_per_channel = edfa_gain_tilt_db_per_channel

        # Store results after analysis
        self._results: list[dict] = []

    def get_channel_wavelengths(self) -> list[float]:
        """
        Compute wavelength for each WDM channel.

        Channels are symmetrically distributed around the center wavelength.
        Spacing is converted from GHz to nm using:
          Δλ = λ² × Δf / c

        Returns
        -------
        list[float]
            List of channel wavelengths in nm.
        """
        # Convert channel spacing from GHz to nm at center wavelength
        # Δλ = λ² × Δf / c  (nm)
        lambda_m = self.center_wavelength_nm * 1e-9  # nm → m
        delta_f = self.channel_spacing_ghz * 1e9      # GHz → Hz
        delta_lambda_nm = (lambda_m ** 2 * delta_f / SPEED_OF_LIGHT_M_S) * 1e9  # m → nm

        # Center the grid: channels distributed symmetrically
        center_idx = (self.num_channels - 1) / 2.0
        wavelengths = []
        for i in range(self.num_channels):
            offset = (i - center_idx) * delta_lambda_nm
            wavelengths.append(round(self.center_wavelength_nm + offset, 3))

        return wavelengths

    def analyze_all_channels(self, link_params: dict) -> list[dict]:
        """
        Run per-channel analysis across all WDM channels.

        For each channel, the link OSNR is adjusted by the gain tilt
        penalty based on that channel's distance from the center.

        Parameters
        ----------
        link_params : dict
            Parameters for OpticalLink:
            {num_spans, span_length_km, launch_power_dbm,
             attenuation_db_per_km, edfa_noise_figure_db,
             optical_bw_hz, bitrate_gbps, modulation}

        Returns
        -------
        list[dict]
            Per-channel performance results.
        """
        # Create the base link and get baseline OSNR
        link = OpticalLink(
            num_spans=link_params.get("num_spans", 10),
            span_length_km=link_params.get("span_length_km", 80),
            launch_power_dbm=link_params.get("launch_power_dbm", 0),
            attenuation_db_per_km=link_params.get("attenuation_db_per_km", 0.2),
            edfa_noise_figure_db=link_params.get("edfa_noise_figure_db", 5.0),
            optical_bw_hz=link_params.get("optical_bw_hz", 12.5e9),
            include_nonlinear=link_params.get("include_nonlinear", True),
        )
        link_analysis = link.analyze()
        base_osnr_db = link_analysis["osnr_db"]

        wavelengths = self.get_channel_wavelengths()
        center_idx = (self.num_channels - 1) / 2.0

        bitrate = link_params.get("bitrate_gbps", 10.0)
        modulation = link_params.get("modulation", "OOK")
        optical_bw = link_params.get("optical_bw_hz", 12.5e9)

        results = []
        for i, wavelength in enumerate(wavelengths):
            # Gain tilt penalty: increases linearly from center
            distance_from_center = abs(i - center_idx)
            tilt_penalty_db = distance_from_center * self.edfa_gain_tilt_db_per_channel

            # Per-channel OSNR after gain tilt penalty
            channel_osnr_db = base_osnr_db - tilt_penalty_db

            # Evaluate performance for this channel
            perf = evaluate_performance(
                osnr_db=channel_osnr_db,
                bitrate_gbps=bitrate,
                modulation=modulation,
                optical_bw_hz=optical_bw,
            )
            perf["channel_index"] = i + 1
            perf["wavelength_nm"] = wavelength
            perf["tilt_penalty_db"] = round(tilt_penalty_db, 3)
            results.append(perf)

        self._results = results
        return results

    def channel_summary(self) -> dict:
        """
        Summary statistics across all WDM channels.

        Returns
        -------
        dict
            Summary with worst/best/average OSNR and pass/fail count.
        """
        if not self._results:
            raise RuntimeError("No analysis results. Call analyze_all_channels() first.")

        osnr_values = [r["osnr_db"] for r in self._results]
        pass_count = sum(1 for r in self._results if r["pass"])

        worst_idx = osnr_values.index(min(osnr_values))
        best_idx = osnr_values.index(max(osnr_values))

        return {
            "num_channels": self.num_channels,
            "worst_osnr_db": round(min(osnr_values), 2),
            "best_osnr_db": round(max(osnr_values), 2),
            "average_osnr_db": round(sum(osnr_values) / len(osnr_values), 2),
            "worst_channel": self._results[worst_idx]["channel_index"],
            "best_channel": self._results[best_idx]["channel_index"],
            "channels_pass": pass_count,
            "channels_fail": self.num_channels - pass_count,
            "all_pass": pass_count == self.num_channels,
        }

    def __repr__(self) -> str:
        return (
            f"WDMSystem(channels={self.num_channels}, "
            f"spacing={self.channel_spacing_ghz} GHz, "
            f"center={self.center_wavelength_nm} nm)"
        )
