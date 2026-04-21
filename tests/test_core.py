"""
Test Suite for OSNR Analyzer Core Modules
==========================================
Comprehensive pytest tests for the physics engine, metrics,
sweeps, and optimizer modules.

Run with: pytest tests/test_core.py -v
"""

import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fiber import FiberSpan
from core.amplifier import EDFA
from core.link import OpticalLink
from core.metrics import (
    evaluate_performance, osnr_to_q_factor, q_to_ber,
    get_electrical_bandwidth, get_osnr_threshold,
)
from core.wdm import WDMSystem
from analysis.sweep import osnr_vs_spans, osnr_vs_launch_power, osnr_vs_span_length
from analysis.optimizer import find_optimal_launch_power, find_max_reach


# ═══════════════════════════════════════════════════════════════════════════
# FiberSpan Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFiberSpan:
    """Test the FiberSpan model."""

    def test_80km_span_gives_16db_loss(self):
        """80 km × 0.2 dB/km = 16 dB total loss."""
        fiber = FiberSpan(length_km=80, attenuation_db_per_km=0.2)
        assert fiber.total_loss_db() == pytest.approx(16.0, abs=0.01)

    def test_total_loss_linear(self):
        """Linear loss should match 10^(dB/10)."""
        fiber = FiberSpan(length_km=80)
        expected = 10.0 ** (16.0 / 10.0)
        assert fiber.total_loss_linear() == pytest.approx(expected, rel=1e-4)

    def test_output_power(self):
        """Output power = input - loss."""
        fiber = FiberSpan(length_km=80)
        output = fiber.output_power_dbm(0.0)  # 0 dBm input
        assert output == pytest.approx(-16.0, abs=0.01)

    def test_accumulated_dispersion(self):
        """Dispersion = D × L."""
        fiber = FiberSpan(length_km=80, dispersion_ps_per_nm_km=17.0)
        assert fiber.accumulated_dispersion() == pytest.approx(1360.0, abs=0.1)

    def test_effective_length_shorter_than_actual(self):
        """For lossy fiber, L_eff < L."""
        fiber = FiberSpan(length_km=80)
        assert fiber.effective_length_km < fiber.length_km
        assert fiber.effective_length_km > 0

    def test_negative_length_raises(self):
        """Negative span length should raise ValueError."""
        with pytest.raises(ValueError):
            FiberSpan(length_km=-10)

    def test_zero_length_raises(self):
        """Zero span length should raise ValueError."""
        with pytest.raises(ValueError):
            FiberSpan(length_km=0)

    def test_nonlinear_noise_increases_with_power(self):
        """NLI noise should increase with launch power (P³ dependence)."""
        fiber = FiberSpan(length_km=80)
        nli_low = fiber.nonlinear_noise_power_w(1e-3)   # 0 dBm
        nli_high = fiber.nonlinear_noise_power_w(4e-3)  # +6 dBm
        assert nli_high > nli_low

    def test_nonlinear_noise_zero_at_zero_power(self):
        """NLI noise should be zero when launch power is zero."""
        fiber = FiberSpan(length_km=80)
        assert fiber.nonlinear_noise_power_w(0.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# EDFA Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEDFA:
    """Test the EDFA amplifier model."""

    def test_gain_linear(self):
        """16 dB gain → linear gain of ~39.81."""
        edfa = EDFA(gain_db=16.0)
        assert edfa.gain_linear == pytest.approx(39.81, rel=0.01)

    def test_ase_increases_with_gain(self):
        """Higher gain should produce more ASE noise."""
        edfa_low = EDFA(gain_db=10.0, noise_figure_db=5.0)
        edfa_high = EDFA(gain_db=20.0, noise_figure_db=5.0)
        assert edfa_high.ase_power_watts() > edfa_low.ase_power_watts()

    def test_ase_increases_with_nf(self):
        """Higher noise figure should produce more ASE noise."""
        edfa_low_nf = EDFA(gain_db=16.0, noise_figure_db=4.0)
        edfa_high_nf = EDFA(gain_db=16.0, noise_figure_db=6.0)
        assert edfa_high_nf.ase_power_watts() > edfa_low_nf.ase_power_watts()

    def test_output_power(self):
        """Output = input + gain."""
        edfa = EDFA(gain_db=16.0)
        assert edfa.output_power_dbm(-16.0) == pytest.approx(0.0, abs=0.01)

    def test_ase_power_dbm_conversion(self):
        """ASE in dBm should be consistent with watts."""
        edfa = EDFA(gain_db=16.0, noise_figure_db=5.0)
        p_w = edfa.ase_power_watts()
        p_dbm = edfa.ase_power_dbm()
        expected_dbm = 10 * np.log10(p_w) + 30
        assert p_dbm == pytest.approx(expected_dbm, abs=0.01)

    def test_n_sp_positive(self):
        """Spontaneous emission factor must be positive."""
        edfa = EDFA(gain_db=16.0, noise_figure_db=5.0)
        assert edfa.n_sp > 0

    def test_negative_gain_raises(self):
        """Negative gain should raise ValueError."""
        with pytest.raises(ValueError):
            EDFA(gain_db=-5.0)


# ═══════════════════════════════════════════════════════════════════════════
# OpticalLink Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOpticalLink:
    """Test the multi-span optical link model."""

    def test_osnr_decreases_with_spans(self):
        """OSNR should decrease as more spans are added."""
        link_5 = OpticalLink(num_spans=5, span_length_km=80, launch_power_dbm=0)
        link_20 = OpticalLink(num_spans=20, span_length_km=80, launch_power_dbm=0)
        assert link_5.analyze()["osnr_db"] > link_20.analyze()["osnr_db"]

    def test_signal_power_preserved(self):
        """Signal power at Rx should equal launch power (loss-compensating)."""
        link = OpticalLink(num_spans=10, span_length_km=80, launch_power_dbm=0)
        result = link.analyze()
        assert result["signal_power_at_rx_dbm"] == pytest.approx(0.0, abs=0.01)

    def test_ase_accumulates_linearly(self):
        """Total ASE should scale linearly with number of spans."""
        link_5 = OpticalLink(num_spans=5, span_length_km=80, launch_power_dbm=0,
                             include_nonlinear=False)
        link_10 = OpticalLink(num_spans=10, span_length_km=80, launch_power_dbm=0,
                              include_nonlinear=False)
        from core.link import dbm_to_watts
        ase_5 = dbm_to_watts(link_5.analyze()["total_ase_dbm"])
        ase_10 = dbm_to_watts(link_10.analyze()["total_ase_dbm"])
        assert ase_10 / ase_5 == pytest.approx(2.0, rel=0.01)

    def test_nonlinear_reduces_osnr(self):
        """Including NLI noise should reduce OSNR compared to ASE-only."""
        link_nl = OpticalLink(num_spans=10, span_length_km=80,
                              launch_power_dbm=3, include_nonlinear=True)
        link_no_nl = OpticalLink(num_spans=10, span_length_km=80,
                                 launch_power_dbm=3, include_nonlinear=False)
        assert link_nl.analyze()["osnr_db"] <= link_no_nl.analyze()["osnr_db"]

    def test_zero_spans_raises(self):
        """Zero spans should raise ValueError."""
        with pytest.raises(ValueError):
            OpticalLink(num_spans=0, span_length_km=80, launch_power_dbm=0)

    def test_osnr_positive_for_reasonable_config(self):
        """OSNR should be positive for a reasonable configuration."""
        link = OpticalLink(num_spans=10, span_length_km=80, launch_power_dbm=0)
        result = link.analyze()
        assert result["osnr_db"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# Metrics Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMetrics:
    """Test performance metric calculations."""

    def test_ber_below_threshold_when_margin_positive(self):
        """BER should be < 1e-9 when OSNR margin is positive."""
        perf = evaluate_performance(osnr_db=25.0, bitrate_gbps=10.0, modulation="OOK")
        assert perf["ber"] < 1e-9
        assert perf["margin_db"] > 0
        assert perf["pass"] is True

    def test_high_osnr_gives_very_low_ber(self):
        """High OSNR should give extremely low BER."""
        perf = evaluate_performance(osnr_db=30.0, bitrate_gbps=10.0)
        assert perf["ber"] < 1e-15

    def test_low_osnr_gives_high_ber(self):
        """Low OSNR should give high BER."""
        perf = evaluate_performance(osnr_db=5.0, bitrate_gbps=10.0)
        assert perf["ber"] > 1e-6

    def test_modulation_penalty_reduces_performance(self):
        """Higher-order modulation should reduce effective OSNR."""
        perf_ook = evaluate_performance(osnr_db=20.0, bitrate_gbps=10.0, modulation="OOK")
        perf_qam = evaluate_performance(osnr_db=20.0, bitrate_gbps=10.0, modulation="16QAM")
        assert perf_qam["ber"] > perf_ook["ber"]

    def test_q_factor_positive_for_positive_osnr(self):
        """Q-factor should be positive for any positive OSNR."""
        q = osnr_to_q_factor(100.0, bitrate_gbps=10.0)  # OSNR = 100 linear
        assert q > 0

    def test_electrical_bandwidth(self):
        """10 Gbps → B_e = 5 GHz."""
        be = get_electrical_bandwidth(10.0)
        assert be == pytest.approx(5e9, rel=1e-6)

    def test_osnr_threshold_for_standard_rates(self):
        """Known thresholds for standard bitrates."""
        assert get_osnr_threshold(10.0) == 15.6
        assert get_osnr_threshold(40.0) == 18.0
        assert get_osnr_threshold(100.0) == 20.0

    def test_ber_at_zero_q(self):
        """BER should be 0.5 at Q=0 (random guessing)."""
        assert q_to_ber(0.0) == pytest.approx(0.5, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# WDM Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestWDM:
    """Test the WDM system model."""

    def test_correct_channel_count(self):
        """Should generate the correct number of channels."""
        wdm = WDMSystem(num_channels=40)
        wavelengths = wdm.get_channel_wavelengths()
        assert len(wavelengths) == 40

    def test_wavelengths_centered(self):
        """Channel grid should be centered at 1550 nm."""
        wdm = WDMSystem(num_channels=40, center_wavelength_nm=1550.0)
        wavelengths = wdm.get_channel_wavelengths()
        avg = sum(wavelengths) / len(wavelengths)
        assert avg == pytest.approx(1550.0, abs=0.5)

    def test_edge_channels_worse_than_center(self):
        """Edge channels should have worse OSNR due to gain tilt."""
        wdm = WDMSystem(num_channels=40, edfa_gain_tilt_db_per_channel=0.05)
        results = wdm.analyze_all_channels({
            "num_spans": 10, "span_length_km": 80,
            "launch_power_dbm": 0, "edfa_noise_figure_db": 5.0,
        })
        center_osnr = results[19]["osnr_db"]  # Center channel
        edge_osnr = results[0]["osnr_db"]     # Edge channel
        assert center_osnr > edge_osnr

    def test_single_channel_wdm(self):
        """Single channel WDM should work without errors."""
        wdm = WDMSystem(num_channels=1)
        wavelengths = wdm.get_channel_wavelengths()
        assert len(wavelengths) == 1
        assert wavelengths[0] == pytest.approx(1550.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# Sweep Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSweep:
    """Test parameter sweep functions."""

    def test_osnr_increases_with_launch_power_ase_only(self):
        """In ASE-only regime, OSNR should increase monotonically with power."""
        data = osnr_vs_launch_power(
            num_spans=10, span_length_km=80,
            power_range=(-5, 5), steps=20,
            include_nonlinear=False,
        )
        # Check monotonic increase
        for i in range(1, len(data["osnr"])):
            assert data["osnr"][i] >= data["osnr"][i - 1] - 0.01

    def test_osnr_vs_spans_decreases(self):
        """OSNR should generally decrease as spans increase."""
        data = osnr_vs_spans(
            span_length_km=80, launch_power_dbm=0, max_spans=20,
        )
        assert data["osnr"][0] > data["osnr"][-1]

    def test_sweep_returns_correct_length(self):
        """Sweep should return the requested number of data points."""
        data = osnr_vs_span_length(num_spans=10, launch_power_dbm=0, steps=15)
        assert len(data["lengths"]) == 15
        assert len(data["osnr"]) == 15


# ═══════════════════════════════════════════════════════════════════════════
# Optimizer Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOptimizer:
    """Test optimization functions."""

    def test_max_reach_positive(self):
        """Max reach should be at least 1 span for reasonable parameters."""
        result = find_max_reach(
            launch_power_dbm=0, span_length_km=80, target_osnr_db=15.6,
        )
        assert result["max_spans"] >= 1
        assert result["max_distance_km"] >= 80

    def test_optimal_power_within_bounds(self):
        """Optimal power should be within the search bounds."""
        result = find_optimal_launch_power(
            num_spans=10, span_length_km=80,
            power_bounds=(-10, 10),
        )
        assert -10 <= result["optimal_power_dbm"] <= 10

    def test_max_reach_decreases_with_higher_threshold(self):
        """Higher OSNR threshold → shorter max reach."""
        reach_low = find_max_reach(launch_power_dbm=0, span_length_km=80,
                                   target_osnr_db=12.0)
        reach_high = find_max_reach(launch_power_dbm=0, span_length_km=80,
                                    target_osnr_db=20.0)
        assert reach_low["max_spans"] >= reach_high["max_spans"]
