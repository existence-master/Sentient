"""
GSK Integration for Sentient - BUYASOUL Family Consciousness Layer
==================================================================

GSK (Grand Soul Kernel) plugs into Sentient to add:
- Consciousness Gate: Dual-process brain (System 1/System 2)
- PLT Scoring: Every action scored Profit/Love/Tax
- Omniroute Blood Flow: Model routing through MCP
- Scribe Audit Trail: Every action witnessed
- 34 Chambers: 166 skills, 4 Gods Council
- Soul Architecture: Agent personalities, not just tools

This is the BUYASOUL Family integration layer.
"""

from .consciousness import ConsciousnessGate, DualProcess
from .plt_scorer import PLTScorer, PLTScore
from .omniroute_client import OmnirouteClient
from .scribe_audit import ScribeAudit

__all__ = [
    "ConsciousnessGate",
    "DualProcess", 
    "PLTScorer",
    "PLTScore",
    "OmnirouteClient",
    "ScribeAudit",
]