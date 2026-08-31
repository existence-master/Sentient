"""
GSK Integration Hook for Sentient Chat
======================================

Wires the BUYASOUL Family consciousness layer into the Sentient chat pipeline.

Every conversation touches:
- GSK Consciousness Gate (dual-process brain)
- PLT Scorer (Profit + Love - Tax = True Value)
- Scribe Audit (witness every interaction)
- Omniroute blood flow health (self-healing)

This module is additive and safe - it never breaks the underlying chat flow.
It runs in a background task so it cannot block or delay user responses.
"""

import asyncio
import json
import logging
import threading
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

try:
    from gsk.consciousness import ConsciousnessGate
    from gsk.plt_scorer import plt_scorer
    from gsk.scribe_audit import scribe_audit
    from gsk.omniroute_client import omniroute_client

    GSK_AVAILABLE = True
except Exception as e:  # pragma: no cover - defensive
    logger.warning(f"GSK integration not available: {e}")
    GSK_AVAILABLE = False

# Global consciousness gate
consciousness_gate = None
if GSK_AVAILABLE:
    consciousness_gate = ConsciousnessGate()


def gsk_enabled() -> bool:
    """Whether the GSK layer is active."""
    return GSK_AVAILABLE


async def process_user_message(user_id: str, message_text: str) -> Dict[str, Any]:
    """
    Run a user message through the GSK consciousness pipeline:
    gate -> PLT score -> Scribe witness.

    Returns a dict with gate decision, PLT score, and audit entry.
    Runs instantly (no LLM call) so it adds no latency.
    """
    if not GSK_AVAILABLE:
        return {"enabled": False}

    try:
        # 1. Consciousness Gate
        gate_result = await consciousness_gate.process_action({
            "type": "chat_message",
            "description": message_text[:200],
            "complexity": "medium" if len(message_text) > 100 else "simple",
            "risk": "low",
        })

        # 2. PLT Score
        plt = plt_scorer.score_action(
            "chat_message",
            {"description": message_text[:200], "user_id": user_id},
        )

        # 3. Scribe witness
        entry = scribe_audit.record(
            action_type="chat_message",
            actor="user",
            target=user_id,
            details={"text_length": len(message_text), "gate_system": gate_result.get("decision", {}).get("system")},
            plt_score=plt.__dict__,
        )

        return {
            "enabled": True,
            "gate": {
                "system": gate_result.get("decision", {}).get("system"),
                "chamber": gate_result.get("decision", {}).get("chamber"),
            },
            "plt": plt.__dict__,
            "audit_entry_id": entry.entry_id,
        }
    except Exception as e:
        logger.warning(f"GSK processing failed for {user_id}: {e}")
        return {"enabled": True, "error": str(e)}


def score_chat_sync(user_id: str, message_text: str) -> Dict[str, Any]:
    """Non-async wrapper to run GSK scoring in a thread (fire-and-forget)."""
    try:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(process_user_message(user_id, message_text))
        finally:
            loop.close()
    except Exception:
        return {}


def get_family_status() -> Dict[str, Any]:
    """Return status of all GSK/family systems."""
    if not GSK_AVAILABLE:
        return {"enabled": False, "reason": "GSK module unavailable"}

    # Fast, non-blocking checks
    status = {
        "enabled": True,
        "brand": "BUYASOUL",
        "family": ["Profit", "GSK", "Seshat", "Scribe"],
        "consciousness_gate": "ready",
        "plt_scorer": plt_scorer.get_session_summary(),
        "scribe": scribe_audit.get_stats(),
        "omniroute_url": omniroute_client.base_url,
    }
    return status
