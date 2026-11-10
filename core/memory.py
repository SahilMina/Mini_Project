"""
Context Memory Engine
Academic Stress & Clinical Referral Ecosystem
Central University of Karnataka, Kadaganchi, Kalaburagi
Author: Sahil Mina (24UCOMP047) | Guide: Prof. Srikantaiah K C

Maintains contextual continuity across conversational turns:
- Rolling multi-turn buffer (retains last 6 turns per student)
- Longitudinal stressor and topic tracking
- Formatted prompt augmentation for the Empathetic SLM Generator
- 100% on-device SQLite persistence
"""

from typing import List, Dict, Any, Optional
from crypto.db import db_manager


class ConversationMemoryManager:
    """Manages short-term multi-turn conversational context and longitudinal episodic memory."""

    def __init__(self, db=db_manager):
        self.db = db

    def record_turn(
        self,
        user_id: int,
        user_text: str,
        assistant_reply: str,
        primary_domain: Optional[str] = None
    ):
        """Records a user-assistant conversation exchange into memory."""
        if not user_id:
            return

        self.db.save_memory_turn(
            user_id=user_id,
            role="user",
            content=user_text,
            summary_topic=primary_domain
        )
        self.db.save_memory_turn(
            user_id=user_id,
            role="assistant",
            content=assistant_reply,
            summary_topic=primary_domain
        )

    def get_context_turns(self, user_id: Optional[int], limit: int = 6) -> List[Dict[str, Any]]:
        """Retrieves recent conversation turns for context retention."""
        if not user_id:
            return []
        return self.db.get_recent_memory(user_id=user_id, limit=limit)

    def format_context_for_prompt(self, user_id: Optional[int], limit: int = 4) -> str:
        """
        Formats prior conversation turns into a succinct prompt context snippet
        that can be injected into the response generator.
        """
        if not user_id:
            return ""

        turns = self.get_context_turns(user_id=user_id, limit=limit)
        if not turns:
            return ""

        context_lines = []
        for turn in turns:
            role_label = "Student" if turn["role"] == "user" else "Counselor"
            # Truncate content to avoid bloating context
            content_snippet = turn["content"][:160] + ("..." if len(turn["content"]) > 160 else "")
            context_lines.append(f"{role_label}: {content_snippet}")

        return "\n".join(context_lines)

    def get_episodic_summary(self, user_id: Optional[int]) -> Dict[str, Any]:
        """
        Extracts recent patterns for longitudinal stress tracking:
        - Common topics/domains
        - Total turns
        """
        if not user_id:
            return {"active": False, "turns_count": 0, "frequent_topic": None}

        turns = self.get_context_turns(user_id=user_id, limit=10)
        if not turns:
            return {"active": True, "turns_count": 0, "frequent_topic": None}

        topics = [t["topic"] for t in turns if t["topic"]]
        frequent_topic = None
        if topics:
            frequent_topic = max(set(topics), key=topics.count)

        return {
            "active": True,
            "turns_count": len(turns),
            "frequent_topic": frequent_topic
        }


# Singleton memory instance
memory_manager = ConversationMemoryManager()

