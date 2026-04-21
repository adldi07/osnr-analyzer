"""
Core physics engine for WDM optical link simulation.

Provides classes for modeling fiber spans, EDFA amplifiers,
multi-span optical links, WDM systems, and performance metrics.
"""

from core.fiber import FiberSpan
from core.amplifier import EDFA
from core.link import OpticalLink
from core.wdm import WDMSystem
from core.metrics import evaluate_performance

__all__ = ["FiberSpan", "EDFA", "OpticalLink", "WDMSystem", "evaluate_performance"]
