"""
Scribe Audit for Sentient - BUYASOUL Witness Layer
===================================================

Every action in Sentient is witnessed and recorded by Scribe.
This provides full transparency and audit trail.

Scribe is the Witness aspect of the BUYASOUL Family.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """A single audit entry witnessed by Scribe."""
    entry_id: str = ""
    action_type: str = ""
    actor: str = ""
    target: str = ""
    details: Dict = None
    plt_score: Dict = None
    timestamp: str = ""
    session_id: str = ""
    source: str = "sentient-gsk"
    
    def __post_init__(self):
        if not self.entry_id:
            import uuid
            self.entry_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.details is None:
            self.details = {}


class ScribeAudit:
    """
    Scribe Audit System - The Witness.
    
    Records every significant action for:
    - Transparency
    - Debugging
    - Compliance
    - PLT tracking
    - Session replay
    """
    
    def __init__(self, max_entries: int = 10000):
        self.entries: List[AuditEntry] = []
        self.max_entries = max_entries
        self.session_id = None
        self.stats = {
            "total_actions": 0,
            "actions_by_type": {},
            "actions_by_actor": {},
            "session_count": 0,
        }
    
    def start_session(self, session_id: str = None):
        """Start a new audit session."""
        import uuid
        self.session_id = session_id or str(uuid.uuid4())
        self.stats["session_count"] += 1
        logger.info(f"Scribe started session: {self.session_id}")
    
    def record(
        self,
        action_type: str,
        actor: str = "sentient",
        target: str = "",
        details: Dict = None,
        plt_score: Dict = None,
    ) -> AuditEntry:
        """
        Record an action.
        
        Args:
            action_type: Type of action (chat, tool_call, task_complete, etc.)
            actor: Who performed the action
            target: What was acted upon
            details: Additional details
            plt_score: PLT score if available
        """
        entry = AuditEntry(
            action_type=action_type,
            actor=actor,
            target=target,
            details=details or {},
            plt_score=plt_score,
            session_id=self.session_id or "unknown",
        )
        
        self.entries.append(entry)
        self._update_stats(entry)
        
        # Trim if over limit
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        
        logger.debug(f"Scribe: {action_type} by {actor} -> {target}")
        return entry
    
    def _update_stats(self, entry: AuditEntry):
        """Update audit statistics."""
        self.stats["total_actions"] += 1
        self.stats["actions_by_type"][entry.action_type] = (
            self.stats["actions_by_type"].get(entry.action_type, 0) + 1
        )
        self.stats["actions_by_actor"][entry.actor] = (
            self.stats["actions_by_actor"].get(entry.actor, 0) + 1
        )
    
    def get_entries(
        self,
        action_type: str = None,
        actor: str = None,
        limit: int = 100,
        since: str = None,
    ) -> List[Dict]:
        """Get audit entries with optional filters."""
        filtered = self.entries
        
        if action_type:
            filtered = [e for e in filtered if e.action_type == action_type]
        if actor:
            filtered = [e for e in filtered if e.actor == actor]
        if since:
            filtered = [e for e in filtered if e.timestamp >= since]
        
        return [asdict(e) for e in filtered[-limit:]]
    
    def get_stats(self) -> Dict:
        """Get audit statistics."""
        return {
            **self.stats,
            "entries_stored": len(self.entries),
            "session_id": self.session_id,
        }
    
    def get_timeline(self, limit: int = 50) -> List[Dict]:
        """Get a timeline of recent actions."""
        return [asdict(e) for e in self.entries[-limit:]]
    
    def get_plt_history(self) -> List[Dict]:
        """Get history of PLT scores."""
        return [
            {
                "timestamp": e.timestamp,
                "action_type": e.action_type,
                "plt_score": e.plt_score,
            }
            for e in self.entries
            if e.plt_score
        ]
    
    def search(self, query: str) -> List[Dict]:
        """Search audit entries by text."""
        query_lower = query.lower()
        results = []
        
        for entry in self.entries:
            searchable = json.dumps(asdict(entry), default=str).lower()
            if query_lower in searchable:
                results.append(asdict(entry))
        
        return results


# Global Scribe instance
scribe_audit = ScribeAudit()