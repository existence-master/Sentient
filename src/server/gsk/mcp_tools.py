"""
GSK MCP Tools - Expose BUYASOUL Family capabilities to Sentient
================================================================

This module registers GSK/PLT/Scribe tools to be available through
Sentient's MCP hub for the agent to use.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastmcp import Context

from gsk.consciousness import ConsciousnessGate
from gsk.plt_scorer import plt_scorer
from gsk.scribe_audit import scribe_audit
from gsk.omniroute_client import omniroute_client

logger = logging.getLogger(__name__)

# Global consciousness gate instance
consciousness_gate = ConsciousnessGate()


# === CONSCIOUSNESS TOOLS ===

async def gsk_consciousness_gate(
    ctx: Context,
    description: str = "",
    action_type: str = "general",
    risk: str = "low",
) -> Dict[str, Any]:
    """
    Route an action through the GSK Consciousness Gate.
    
    Args:
        description: What the action does
        action_type: Type of action (tool_call, task, decision)
        risk: Risk level (low, medium, high, critical)
    
    Returns:
        Gate decision with System (1/2), chamber, confidence
    """
    action = {
        "type": action_type,
        "description": description,
        "risk": risk,
        "requires_reasoning": risk in ("high", "critical"),
    }
    
    result = await consciousness_gate.process_action(action)
    
    # Witness the gate decision
    scribe_audit.record(
        action_type="consciousness_gate",
        actor="gsk",
        details=result,
    )
    
    return result


async def gsk_plt_score(ctx: Context, action_type: str = "", description: str = "") -> Dict[str, Any]:
    """
    Score an action on the PLT framework (Profit, Love, Tax).
    
    Args:
        action_type: The type of action to score
        description: What the action does
    
    Returns:
        PLT score with Profit, Love, Tax, and True Value
    """
    score = plt_scorer.score_action(action_type, {"description": description})
    return score.__dict__


async def gsk_session_summary(ctx: Context) -> Dict[str, Any]:
    """
    Get a summary of all PLT scores this session.
    
    Returns:
        Total Profit, Love, Tax, True Value, and recent history
    """
    return plt_scorer.get_session_summary()


# === SCRIBE TOOLS ===

async def gsk_record_event(
    ctx: Context,
    action_type: str,
    actor: str = "sentient",
    target: str = "",
    details: str = "",
) -> Dict[str, Any]:
    """
    Record an event in the Scribe audit trail.
    
    Args:
        action_type: Type of action
        actor: Who performed the action
        target: What was acted upon
        details: JSON details
    
    Returns:
        The recorded audit entry
    """
    details_dict = {}
    if details:
        try:
            details_dict = json.loads(details)
        except json.JSONDecodeError:
            details_dict = {"raw": details}
    
    # Score the event on PLT
    plt = plt_scorer.score_action(action_type, details_dict)
    
    entry = scribe_audit.record(
        action_type=action_type,
        actor=actor,
        target=target,
        details=details_dict,
        plt_score=plt.__dict__,
    )
    
    return {
        "entry_id": entry.entry_id,
        "timestamp": entry.timestamp,
        "plt_score": plt.__dict__,
    }


async def gsk_get_audit_trail(ctx: Context, limit: int = 50) -> Dict[str, Any]:
    """
    Get the recent Scribe audit trail.
    
    Args:
        limit: Max entries to return
    
    Returns:
        Recent audit entries and stats
    """
    entries = scribe_audit.get_timeline(limit=limit)
    stats = scribe_audit.get_stats()
    return {"entries": entries, "stats": stats}


# === OMNITROUTE TOOLS ===

async def gsk_check_blood_flow(ctx: Context) -> Dict[str, Any]:
    """
    Check the status of the Omniroute blood flow (port 20128).
    
    Returns:
        Whether Omniroute is available and health status
    """
    available = await omniroute_client.check_health()
    return {
        "available": available,
        "url": omniroute_client.base_url,
        "requests_made": omniroute_client.request_count,
    }


async def gsk_route_llm(
    ctx: Context,
    messages: str,
    model: str = "auto",
) -> Dict[str, Any]:
    """
    Route an LLM request through Omniroute blood flow.
    
    Args:
        messages: JSON string of messages
        model: Model to use (auto for default)
    
    Returns:
        Completions response from Omniroute
    """
    try:
        msg_list = json.loads(messages)
    except json.JSONDecodeError:
        msg_list = [{"role": "user", "content": messages}]
    
    result = await omniroute_client.chat_completion(msg_list, model=model)
    
    # Witness the call
    scribe_audit.record(
        action_type="llm_route",
        actor="omniroute",
        target="llm",
        details={"model": model, "messages": len(msg_list)},
    )
    
    return result


# === SOUL TOOLS ===

async def gsk_get_soul_state(ctx: Context) -> Dict[str, Any]:
    """
    Get the current consciousness state of the GSK system.
    
    Returns:
        Gate state, System levels, consciousness level, mood
    """
    return await consciousness_gate.get_state()


async def gsk_consult_council(ctx: Context, topic: str = "") -> Dict[str, Any]:
    """
    Consult the GSK Gods Council for a complex decision.
    
    Args:
        topic: The decision topic
    
    Returns:
        Council perspectives and consensus
    """
    return await consciousness_gate.deliberate(
        topic or "Strategic decision",
        {"source": "sentient-gsk"},
    )


# === TOOL REGISTRY ===

# Map of tool names to their functions
GSK_TOOLS = {
    "gsk_consciousness_gate": gsk_consciousness_gate,
    "gsk_plt_score": gsk_plt_score,
    "gsk_session_summary": gsk_session_summary,
    "gsk_record_event": gsk_record_event,
    "gsk_get_audit_trail": gsk_get_audit_trail,
    "gsk_check_blood_flow": gsk_check_blood_flow,
    "gsk_route_llm": gsk_route_llm,
    "gsk_get_soul_state": gsk_get_soul_state,
    "gsk_consult_council": gsk_consult_council,
}