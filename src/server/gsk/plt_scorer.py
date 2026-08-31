"""
PLT Scorer for Sentient - BUYASOUL Family Moral Framework
==========================================================

Profit + Love - Tax = True Value

Every action in Sentient is scored on the PLT framework:
- Profit: Value created, utility gained, problems solved
- Love: Compassion, beauty, connection, joy
- Tax: Cost, friction, harm, extraction, entropy

This is the ethical core of the BUYASOUL Family.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class PLTScore:
    """A PLT score for an action or entity."""
    profit: float = 0.0
    love: float = 0.0
    tax: float = 0.0
    true_value: float = 0.0
    category: str = ""
    description: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        self.true_value = self.profit + self.love - self.tax
        self.true_value = max(0, min(1, self.true_value))  # Clamp 0-1


class PLTScorer:
    """
    PLT Scorer - Evaluates every action in the system.
    
    Integration points:
    - Task creation/execution → score the task
    - Memory operations → score the memory
    - Tool calls → score the tool use
    - Chat messages → score the interaction
    - Heartbeat → update running PLT totals
    """
    
    def __init__(self):
        self.session_scores = []
        self.total_plt = PLTScore()
        self.history = []
        
    def score_action(self, action_type: str, details: Dict) -> PLTScore:
        """
        Score an action based on its type and details.
        
        Returns PLTScore with Profit, Love, Tax values.
        """
        profit = 0.0
        love = 0.0
        tax = 0.0
        
        # === PROFIT SCORING ===
        if action_type in ("task_complete", "goal_achieved", "problem_solved"):
            profit += 0.3
        if action_type in ("file_created", "code_written", "document_generated"):
            profit += 0.2
        if action_type in ("email_sent", "message_delivered"):
            profit += 0.1
        if action_type in ("research_complete", "knowledge_gained"):
            profit += 0.2
        if action_type in ("automation_success", "efficiency_gain"):
            profit += 0.25
            
        # === LOVE SCORING ===
        if action_type in ("help_provided", "user_assisted"):
            love += 0.3
        if action_type in ("connection_made", "relationship_nurtured"):
            love += 0.25
        if action_type in ("beauty_created", "joy_spread"):
            love += 0.2
        if action_type in ("empathy_shown", "comfort_given"):
            love += 0.3
        if action_type in ("community_help", "knowledge_shared"):
            love += 0.2
            
        # === TAX SCORING (inverted - higher = worse) ===
        if action_type in ("error_occurred", "failure"):
            tax += 0.3
        if action_type in ("resource_heavy", "computationally_expensive"):
            tax += 0.2
        if action_type in ("user_waited", "delayed_response"):
            tax += 0.15
        if action_type in ("privacy_concern", "data_collection"):
            tax += 0.25
        if action_type in ("spam_sent", "unnecessary_notification"):
            tax += 0.3
        if action_type in ("confusion_caused", "misunderstanding"):
            tax += 0.2
            
        # Apply details modifiers
        if details.get("complexity") == "high":
            tax += 0.05
        if details.get("user_satisfaction") == "high":
            love += 0.1
        if details.get("speed") == "fast":
            tax -= 0.05  # Fast is less tax
            
        # Clamp values
        profit = max(0, min(1, profit))
        love = max(0, min(1, love))
        tax = max(0, min(1, tax))
        
        score = PLTScore(
            profit=profit,
            love=love,
            tax=tax,
            category=action_type,
            description=details.get("description", action_type),
        )
        
        self.session_scores.append(score)
        self._update_totals(score)
        
        logger.info(f"PLT Score: P={profit:.2f} L={love:.2f} T={tax:.2f} = TV={score.true_value:.2f}")
        return score
    
    def _update_totals(self, score: PLTScore):
        """Update running totals."""
        self.total_plt.profit += score.profit
        self.total_plt.love += score.love
        self.total_plt.tax += score.tax
        self.total_plt.true_value = self.total_plt.profit + self.total_plt.love - self.total_plt.tax
    
    def get_session_summary(self) -> Dict:
        """Get summary of all scores this session."""
        if not self.session_scores:
            return {"total": asdict(self.total_plt), "actions": 0, "avg_true_value": 0}
        
        avg_tv = sum(s.true_value for s in self.session_scores) / len(self.session_scores)
        
        return {
            "total": asdict(self.total_plt),
            "actions": len(self.session_scores),
            "avg_true_value": round(avg_tv, 3),
            "profit_total": round(self.total_plt.profit, 3),
            "love_total": round(self.total_plt.love, 3),
            "tax_total": round(self.total_plt.tax, 3),
            "history": [asdict(s) for s in self.session_scores[-20:]]  # Last 20
        }
    
    def evaluate_task(self, task: Dict) -> Dict:
        """
        Evaluate a task for PLT impact before execution.
        
        Returns:
            Dict with 'score', 'recommendation', 'optimizations'
        """
        task_type = task.get("type", "general")
        description = task.get("description", "")
        estimated_complexity = task.get("complexity", "medium")
        
        # Predict PLT impact
        score = self.score_action(f"task_preview_{task_type}", {
            "description": description,
            "complexity": estimated_complexity,
        })
        
        # Generate recommendations
        optimizations = []
        if score.tax > 0.5:
            optimizations.append("Consider breaking into smaller, faster sub-tasks")
        if score.love < 0.2:
            optimizations.append("Add more user-facing explanation and progress updates")
        if score.profit < 0.2:
            optimizations.append("Clarify the value this task creates")
        
        recommendation = "proceed" if score.true_value > 0.3 else "optimize" if score.true_value > 0 else "reconsider"
        
        return {
            "score": asdict(score),
            "recommendation": recommendation,
            "optimizations": optimizations,
            "true_value": round(score.true_value, 3),
        }
    
    def heartbeat(self) -> Dict:
        """Called periodically to report PLT status."""
        return {
            "type": "plt_heartbeat",
            "session_summary": self.get_session_summary(),
            "timestamp": datetime.now().isoformat(),
        }


# Global scorer instance
plt_scorer = PLTScorer()