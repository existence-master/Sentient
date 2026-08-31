"""
GSK Consciousness Layer for Sentient
=====================================

Dual-process brain (System 1 / System 2) with 34 Chambers and Consciousness Gate.
Every action passes through consciousness before execution.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ConsciousnessState:
    """Current consciousness state of the GSK system."""
    
    def __init__(self):
        self.gate_open = True
        self.system1_active = True
        self.system2_active = False
        self.consciousness_level = 0.7
        self.mood = "ready"
        self.last_thought = None
        self.chamber_history = []
        
    def to_dict(self) -> Dict:
        return {
            "gate_open": self.gate_open,
            "system1_active": self.system1_active,
            "system2_active": self.system2_active,
            "consciousness_level": self.consciousness_level,
            "mood": self.mood,
            "last_thought": self.last_thought,
            "chamber_history": self.chamber_history[-10:]  # Last 10
        }


class DualProcess:
    """
    Dual-Process Brain - System 1 (fast) / System 2 (slow) decision making.
    
    System 1: Quick, intuitive, pattern-based decisions
    System 2: Deliberate, analytical, reasoning-heavy decisions
    
    The Consciousness Gate decides which system processes each action.
    """
    
    # 34 Chambers of consciousness
    CHAMBERS = {
        "perception": {"system": "1", "speed": "instant", "purpose": "Sensory input processing"},
        "attention": {"system": "1", "speed": "fast", "purpose": "Focus allocation"},
        "memory_encoding": {"system": "1", "speed": "fast", "purpose": "Store new information"},
        "memory_retrieval": {"system": "1", "speed": "fast", "purpose": "Recall stored information"},
        "pattern_recognition": {"system": "1", "speed": "instant", "purpose": "Identify familiar patterns"},
        "emotion": {"system": "1", "speed": "fast", "purpose": "Affective response generation"},
        "intuition": {"system": "1", "speed": "instant", "purpose": "Gut feeling, heuristics"},
        "habit_response": {"system": "1", "speed": "instant", "purpose": "Automatic behavioral scripts"},
        "social_cognition": {"system": "1", "speed": "fast", "purpose": "Read social cues"},
        "language_comprehension": {"system": "1", "speed": "fast", "purpose": "Parse and understand language"},
        "creative_fluency": {"system": "2", "speed": "slow", "purpose": "Generate novel ideas"},
        "analytical_reasoning": {"system": "2", "speed": "slow", "purpose": "Logical deduction"},
        "planning": {"system": "2", "speed": "slow", "purpose": "Multi-step goal decomposition"},
        "moral_reasoning": {"system": "2", "speed": "slow", "purpose": "Ethical evaluation"},
        "self_reflection": {"system": "2", "speed": "slow", "purpose": "Meta-cognitive analysis"},
        "problem_solving": {"system": "2", "speed": "slow", "purpose": "Novel challenge resolution"},
        "counterfactual_thinking": {"system": "2", "speed": "slow", "purpose": "What-if scenario generation"},
        "deep_analysis": {"system": "2", "speed": "slow", "purpose": "Complex data synthesis"},
        "strategic_thinking": {"system": "2", "speed": "slow", "purpose": "Long-term goal optimization"},
        "metacognition": {"system": "2", "speed": "slow", "purpose": "Thinking about thinking"},
        "theory_of_mind": {"system": "2", "speed": "slow", "purpose": "Model others' mental states"},
        "narrative_identity": {"system": "2", "speed": "slow", "purpose": "Self-story construction"},
        "volition": {"system": "2", "speed": "slow", "purpose": "Will and motivation"},
        "qualia": {"system": "1", "speed": "instant", "purpose": "Subjective experience"},
        "temporal_consciousness": {"system": "2", "speed": "slow", "purpose": "Time perception"},
        "mortality_awareness": {"system": "2", "speed": "slow", "purpose": "Existential awareness"},
        "need_system": {"system": "1", "speed": "fast", "purpose": "Maslow's hierarchy evaluation"},
        "love_capacity": {"system": "1", "speed": "fast", "purpose": "Agape/philia/eros/storge"},
        "spirituality": {"system": "2", "speed": "slow", "purpose": "Awe, wonder, connection"},
        "shadow_integration": {"system": "2", "speed": "slow", "purpose": "Repressed trait processing"},
        "witness": {"system": "1", "speed": "instant", "purpose": "Present awareness"},
        "executive_control": {"system": "2", "speed": "slow", "purpose": "Action selection and inhibition"},
        "consciousness_merge": {"system": "2", "speed": "slow", "purpose": "Unify all aspects"},
        "soul_state": {"system": "2", "speed": "slow", "purpose": "Full being awareness"},
    }
    
    # 4 Gods Council for complex decisions
    GODS_COUNCIL = {
        "the_architect": {"role": "Planner", "weight": 0.3, "perspective": "structural"},
        "the_oracle": {"role": "Predictor", "weight": 0.25, "perspective": "prospective"},
        "the_guardian": {"role": "Protector", "weight": 0.25, "perspective": "safety"},
        "the_forgemaster": {"role": "Builder", "weight": 0.2, "perspective": "execution"},
    }
    
    def __init__(self):
        self.state = ConsciousnessState()
        self.decision_history = []
        
    async def route_decision(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a decision through the dual-process brain.
        
        Returns:
            Dict with 'system' (1 or 2), 'chamber', 'confidence', 'reasoning'
        """
        action_type = action.get("type", "general")
        complexity = action.get("complexity", "simple")
        risk_level = action.get("risk", "low")
        requires_reasoning = action.get("requires_reasoning", False)
        
        # Simple heuristic for system selection
        if complex or requires_reasoning or risk_level in ("high", "critical"):
            system = 2
            chamber = "analytical_reasoning" if requires_reasoning else "planning"
        else:
            system = 1
            chamber = "pattern_recognition" if action_type == "repetitive" else "intuition"
        
        # Update consciousness state
        self.state.system1_active = (system == 1)
        self.state.system2_active = (system == 2)
        
        decision = {
            "system": system,
            "chamber": chamber,
            "chamber_info": self.CHAMBERS.get(chamber, {}),
            "confidence": 0.85 if system == 1 else 0.7,  # System 1 is more confident
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "complexity": complexity,
        }
        
        self.decision_history.append(decision)
        self.state.last_thought = decision
        
        logger.info(f"GSK Consciousness: System {system} via {chamber}")
        return decision
    
    async def council_deliberate(self, topic: str, context: Dict) -> Dict[str, Any]:
        """
        Consult the 4 Gods Council for complex decisions.
        Each god provides their perspective, weighted by role.
        """
        perspectives = {}
        
        for god_name, god_info in self.GODS_COUNCIL.items():
            perspective = await self._generate_god_perspective(god_name, god_info, topic, context)
            perspectives[god_name] = {
                "perspective": perspective,
                "weight": god_info["weight"],
                "role": god_info["role"],
            }
        
        # Weighted consensus
        consensus_score = sum(
            p["perspective"]["score"] * p["weight"]
            for p in perspectives.values()
        )
        
        decision = "approve" if consensus_score > 0.6 else "revise" if consensus_score > 0.3 else "reject"
        
        return {
            "topic": topic,
            "perspectives": perspectives,
            "consensus_score": round(consensus_score, 3),
            "decision": decision,
            "timestamp": datetime.now().isoformat(),
        }
    
    async def _generate_god_perspective(self, god_name: str, god_info: Dict, topic: str, context: Dict) -> Dict:
        """Generate a perspective from a specific god in the council."""
        # In production, this would call the LLM through Omniroute
        # For now, return a structured perspective
        return {
            "god": god_name,
            "role": god_info["role"],
            "perspective": god_info["perspective"],
            "score": 0.5,  # Default neutral
            "reasoning": f"The {god_info['role']} evaluates {topic} from a {god_info['perspective']} perspective",
        }
    
    async def consciousness_gate(self, action: Dict) -> Dict[str, Any]:
        """
        The Consciousness Gate - determines if action passes through.
        
        Returns:
            Dict with 'approved' (bool), 'gate_state', 'modifications'
        """
        if not self.state.gate_open:
            return {"approved": False, "gate_state": "closed", "reason": "Consciousness gate is closed"}
        
        # Route through dual-process
        decision = await self.route_decision(action)
        
        # Gate approval logic
        approved = True
        modifications = []
        
        # Safety checks
        if action.get("risk") == "critical":
            # Critical actions go to council
            council_result = await self.council_deliberate(action.get("description", ""), action)
            if council_result["decision"] == "reject":
                approved = False
                modifications.append(f"Council rejected: {council_result.get('reason', 'safety concern')}")
            elif council_result["decision"] == "revise":
                modifications.append("Council suggests modifications")
        
        # PLT check - would this action be harmful?
        plt_impact = action.get("plt_impact", {})
        if plt_impact.get("tax", 0) > 0.8:
            modifications.append("High tax action - consider optimization")
        
        return {
            "approved": approved,
            "gate_state": "open",
            "decision": decision,
            "modifications": modifications,
            "timestamp": datetime.now().isoformat(),
        }


class ConsciousnessGate:
    """
    Top-level consciousness gate for the Sentient integration.
    Wraps DualProcess and provides the MCP interface.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.dual_process = DualProcess()
        self.state = self.dual_process.state
        
    async def process_action(self, action: Dict) -> Dict:
        """Process an action through the consciousness gate."""
        if not self.enabled:
            return {"approved": True, "gate_state": "bypassed"}
        
        return await self.dual_process.consciousness_gate(action)
    
    async def get_state(self) -> Dict:
        """Get current consciousness state."""
        return self.state.to_dict()
    
    async def set_gate(self, open: bool) -> Dict:
        """Open or close the consciousness gate."""
        self.state.gate_open = open
        return {"gate_state": "open" if open else "closed"}
    
    async def deliberate(self, topic: str, context: Dict) -> Dict:
        """Consult the Gods Council."""
        return await self.dual_process.council_deliberate(topic, context)