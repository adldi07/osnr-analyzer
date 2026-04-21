# WDM Optical Link OSNR/BER Performance Analyzer

> A comprehensive Python tool for simulating and analyzing multi-span WDM optical fiber communication links with ASE noise, fiber nonlinearities (GN model), and real-time web dashboard.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Nokia](https://img.shields.io/badge/Nokia-Associate%20Engineer-124191)

---

## Background

In long-haul optical fiber communication systems, signals degrade over distance due to two primary noise sources:

1. **ASE Noise** — Erbium-Doped Fiber Amplifiers (EDFAs) compensate fiber loss but add broadband Amplified Spontaneous Emission noise that accumulates with each amplifier in the chain.
2. **Nonlinear Interference (NLI)** — At higher launch powers, the Kerr effect in fiber causes signal distortion through self-phase modulation (SPM), cross-phase modulation (XPM), and four-wave mixing (FWM).

The **Optical Signal-to-Noise Ratio (OSNR)** determines link quality. This tool models a complete `TX → [Fiber → EDFA]×N → RX` chain and computes OSNR, Q-factor, and Bit Error Rate (BER) to predict whether a link design will meet performance targets.

---

## Installation

```bash
# Clone the repository
git clone <repo-url> && cd osnr-analyzer

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, NumPy, SciPy, Matplotlib, Flask, pytest

---

## CLI Usage

### Basic Analysis
```bash
# 10 spans × 80 km, 0 dBm launch, 10 Gbps OOK
python main.py --spans 10 --span-length 80 --launch-power 0 --nf 5 --bitrate 10

# 30 spans — should FAIL (OSNR too low)
python main.py --spans 30 --span-length 80 --launch-power 0 --bitrate 10

# 100 Gbps QPSK with 40 WDM channels
python main.py --spans 15 --launch-power 2 --bitrate 100 --modulation QPSK --wdm-channels 40
```

### Configuration Profiles
```bash
# Load submarine cable profile
python main.py --config submarine

# Load metro network profile
python main.py --config metro

# Load long-haul terrestrial profile
python main.py --config long_haul
```

### Plot Generation
```bash
# Generate all plots
python main.py --plot all

# Generate specific plot
python main.py --plot osnr_vs_spans --spans 20 --span-length 100

# Power sweep (shows optimal launch power with NLI)
python main.py --plot power_sweep --spans 15

# Disable nonlinear effects for comparison
python main.py --plot power_sweep --no-nonlinear
```

### Max Reach Computation
```bash
python main.py --max-reach --launch-power 2 --span-length 80
```

### Example Output
```
======================================================
   WDM OPTICAL LINK PERFORMANCE REPORT
======================================================
  Total Length      : 800 km
  Number of Spans   : 10
  Launch Power      : 0.0 dBm
  EDFA NF           : 5.0 dB
  Nonlinear Effects : ON
------------------------------------------------------
  Signal Power (Rx) : 0.00 dBm
  ASE Noise Total   : -20.97 dBm
  NLI Noise Total   : -50.12 dBm
  Total Noise       : -20.97 dBm
  OSNR              : 20.97 dB
  Q-Factor          : 15.81
  BER               : 1.23e-56
  OSNR Margin       : 5.37 dB
------------------------------------------------------
  Max Reach         : 35 spans (2800 km)
  Optimal Power     : 1.5 dBm (OSNR: 22.3 dB)
------------------------------------------------------
  Status            : ✓ PASS
======================================================
```

---

## Web Dashboard

Launch the interactive web dashboard:

```bash
python main.py --web
# Open http://localhost:5000
```

**Dashboard features:**
- Real-time OSNR/BER analysis with auto-updating results
- Interactive sliders for all link parameters
- Configuration profile presets (submarine, metro, long-haul)
- Tab-based plot viewer (OSNR vs spans, power sweep, BER curve, WDM)
- Nokia-themed dark mode UI
- Nonlinear effects toggle (GN model ON/OFF)

---

## Project Structure

```
osnr-analyzer/
├── core/                    # Physics simulation engine
│   ├── fiber.py             # FiberSpan model (loss, dispersion, NLI)
│   ├── amplifier.py         # EDFA model (gain, ASE noise)
│   ├── link.py              # Multi-span link (ASE + NLI accumulation)
│   ├── wdm.py               # WDM multi-channel analysis
│   └── metrics.py           # Q-factor, BER, OSNR margin
├── analysis/                # Parameter sweeps & optimization
│   ├── sweep.py             # OSNR vs spans/power/length sweeps
│   └── optimizer.py         # Optimal launch power, max reach
├── visualization/           # Plot generation
│   └── plotter.py           # 4 matplotlib plot types
├── web/                     # Flask web dashboard
│   ├── app.py               # REST API
│   └── templates/
│       └── index.html       # Single-page dashboard UI
├── tests/
│   └── test_core.py         # pytest test suite (35+ tests)
├── configs/                 # Configuration profiles
│   ├── default_system.json  # Standard 10×80 km link
│   ├── submarine.json       # Ultra-long-haul submarine
│   ├── metro.json           # Short metro network
│   └── long_haul.json       # Terrestrial long-haul
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Physics Model

### Signal Attenuation
```
L_dB = α × d       (fiber loss in dB)
P_out = P_in - L   (dBm arithmetic)
```
- `α = 0.2 dB/km` for standard SMF-28 at 1550 nm

### ASE Noise (per EDFA)
```
P_ASE = 2 × n_sp × (G-1) × h × f × B_o
n_sp  = NF_linear × G / (2 × (G-1))
```
- Factor of 2 for both polarization states
- `h = 6.626×10⁻³⁴ J·s`, `f = 193.4 THz`, `B_o = 12.5 GHz`

### Nonlinear Interference (GN Model)
```
G_NLI = (8/27) × γ² × L_eff × G_sig³ × arcsinh(arg) / (π × |β₂|)
P_NLI = G_NLI × B_ref     (per span, accumulates linearly)
```
- Creates optimal launch power: ASE ↓ vs NLI ↑

### OSNR
```
OSNR = P_signal / (P_ASE_total + P_NLI_total)
```

### Q-Factor & BER
```
Q   = √(2 × OSNR × B_o / B_e)
BER = 0.5 × erfc(Q / √2)
B_e = bitrate / 2   (NRZ electrical bandwidth)
```

---

## Plots Generated

| Plot | Description |
|------|-------------|
| **OSNR vs Spans** | Shows reach limitation with threshold and max-reach annotation |
| **Power Sweep** | OSNR vs launch power with optimal point and nonlinear regime shading |
| **BER Curve** | Waterfall curve showing BER vs OSNR with 10⁻⁹ threshold |
| **WDM Channels** | Bar chart of per-channel OSNR with pass/fail coloring |

---

## Running Tests

```bash
pytest tests/ -v
```

All 35+ tests verify:
- Physical correctness (loss, ASE, NLI calculations)
- Edge cases (zero power, negative values)
- Monotonicity properties (OSNR vs power, spans)
- Metric thresholds (BER, Q-factor)
- WDM channel generation

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Numerics | NumPy, SciPy |
| Plotting | Matplotlib |
| Web Dashboard | Flask + vanilla JS/CSS |
| Testing | pytest |
| Configuration | JSON profiles |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
