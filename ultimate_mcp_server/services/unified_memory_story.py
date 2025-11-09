"""Utilities for staging realistic unified memory stories in the SQLite database.

This module provides a small, opinionated fixture that inserts a cohesive
workflow into the unified agent memory database.  The goal is to offer
lightweight integration coverage for every table that powers the unified memory
system so tests can assert that schema migrations and runtime code keep
behaving as expected.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List

from ultimate_mcp_server.working_memory_api import initialize_unified_memory_schema


@dataclass(frozen=True)
class StagedStory:
    """Summary of the staged story with the primary identifiers."""

    workflow_id: str
    goal_ids: List[str]
    action_ids: List[str]
    artifact_ids: List[str]
    memory_ids: List[str]
    timeline_state_ids: List[str]
    memory_link_ids: List[str]
    relationship_ids: List[int]


@dataclass(frozen=True)
class StorySnapshot:
    """Materialized rows that belong to a staged story."""

    workflow: Dict
    goals: List[Dict]
    actions: List[Dict]
    artifacts: List[Dict]
    artifact_relationships: List[Dict]
    memories: List[Dict]
    memory_links: List[Dict]
    cognitive_timeline_states: List[Dict]


class UnifiedMemoryStoryBuilder:
    """Creates cohesive data that touches every unified memory table."""

    def __init__(self, db_path: str = "storage/unified_agent_memory.db") -> None:
        self.db_path = str(db_path)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def stage_support_triage_story(self) -> StagedStory:
        """Populate the database with a realistic customer support workflow.

        The staged workflow walks through a "customer outage triage" task.  It
        includes a high level workflow, nested goals, several actions executed
        by the agent, and the artifacts that are produced along the way.  Every
        unified memory table is exercised so downstream tests can assert that
        data flows and relationships remain intact.
        """

        initialize_unified_memory_schema(self.db_path)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            now = time.time()
            workflow_id = self._uid("workflow")
            workflow_metadata = {
                "customer": "Proxima Robotics",
                "impact": "High",
                "channel": "slack",
            }
            conn.execute(
                """
                INSERT INTO workflows (
                    workflow_id, title, description, goal, status,
                    created_at, updated_at, last_active, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workflow_id,
                    "Customer outage triage",
                    "Diagnose and stabilize an outage reported by Proxima Robotics",
                    "Restore service for warehouse robots",
                    "active",
                    now,
                    now,
                    now,
                    json.dumps(workflow_metadata),
                ),
            )

            # Goals ------------------------------------------------------
            goal_primary = self._uid("goal")
            goal_diagnose = self._uid("goal")
            goal_update = self._uid("goal")
            goals = [goal_primary, goal_diagnose, goal_update]
            conn.executemany(
                """
                INSERT INTO goals (
                    goal_id, workflow_id, parent_goal_id, description, status,
                    priority, created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        goal_primary,
                        workflow_id,
                        None,
                        "Stabilize the reported robotics outage",
                        "in_progress",
                        1,
                        now,
                        now,
                        None,
                    ),
                    (
                        goal_diagnose,
                        workflow_id,
                        goal_primary,
                        "Diagnose sensor desynchronization",
                        "completed",
                        2,
                        now,
                        now,
                        now,
                    ),
                    (
                        goal_update,
                        workflow_id,
                        goal_primary,
                        "Update status page and stakeholders",
                        "blocked",
                        3,
                        now,
                        now,
                        None,
                    ),
                ],
            )

            # Actions ----------------------------------------------------
            action_retrieve_logs = self._uid("action")
            action_compare_versions = self._uid("action")
            action_patch = self._uid("action")
            actions = [action_retrieve_logs, action_compare_versions, action_patch]
            conn.executemany(
                """
                INSERT INTO actions (
                    action_id, workflow_id, parent_action_id, action_type, title,
                    description, reasoning, tool_name, tool_args, tool_result,
                    tool_data, status, created_at, started_at, completed_at,
                    updated_at, sequence_number, priority, tags, result, error,
                    metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        action_retrieve_logs,
                        workflow_id,
                        None,
                        "analysis",
                        "Collect affected robot telemetry",
                        "Pull telemetry logs from malfunctioning robots",
                        "Logs needed to confirm outage scope",
                        "log_fetcher",
                        json.dumps({"robot_ids": ["rx-41", "rx-42"]}),
                        "Telemetry downloaded",
                        json.dumps({"records": 2456}),
                        "completed",
                        now,
                        now,
                        now,
                        now,
                        1,
                        1,
                        "telemetry,triage",
                        "Log bundle stored",
                        None,
                        json.dumps({"goal_id": goal_diagnose}),
                    ),
                    (
                        action_compare_versions,
                        workflow_id,
                        action_retrieve_logs,
                        "analysis",
                        "Compare firmware versions",
                        "Check differences between failing and healthy units",
                        "Identify regression introduced in latest deploy",
                        "firmware_diff",
                        json.dumps({"baseline": "v2.1.0", "candidate": "v2.2.0"}),
                        "Mismatch detected",
                        json.dumps({"changed_modules": ["sensor-fusion"]}),
                        "completed",
                        now,
                        now,
                        now,
                        now,
                        2,
                        2,
                        "firmware,analysis",
                        "Regression isolated",
                        None,
                        json.dumps({"goal_id": goal_diagnose}),
                    ),
                    (
                        action_patch,
                        workflow_id,
                        action_compare_versions,
                        "execution",
                        "Deploy hotfix",
                        "Roll back sensor fusion module for affected fleet",
                        "Rollback mitigates desynchronization",
                        "deployment_manager",
                        json.dumps({"target_fleet": "warehouse-a"}),
                        "Rollback scheduled",
                        json.dumps({"eta_minutes": 15}),
                        "in_progress",
                        now,
                        now,
                        None,
                        now,
                        3,
                        1,
                        "rollback,mitigation",
                        None,
                        None,
                        json.dumps({"goal_id": goal_primary}),
                    ),
                ],
            )

            # Artifacts --------------------------------------------------
            artifact_logs = self._uid("artifact")
            artifact_diff = self._uid("artifact")
            artifact_runbook = self._uid("artifact")
            artifacts = [artifact_logs, artifact_diff, artifact_runbook]
            conn.executemany(
                """
                INSERT INTO artifacts (
                    artifact_id, workflow_id, action_id, artifact_type, name,
                    description, file_path, content, metadata, tags,
                    file_size, checksum, created_at, updated_at, importance,
                    access_count, parent_artifact_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        artifact_logs,
                        workflow_id,
                        action_retrieve_logs,
                        "dataset",
                        "robot-telemetry-2024-05-01.json",
                        "Aggregated telemetry from impacted robots",
                        "/tmp/robot-telemetry-2024-05-01.json",
                        json.dumps({"robot_ids": ["rx-41", "rx-42"]}),
                        json.dumps({"compression": "zip"}),
                        "telemetry,triage",
                        1048576,
                        "abc123",
                        now,
                        now,
                        0.9,
                        3,
                        None,
                    ),
                    (
                        artifact_diff,
                        workflow_id,
                        action_compare_versions,
                        "report",
                        "firmware-diff.html",
                        "HTML diff describing configuration drift",
                        "/tmp/firmware-diff.html",
                        "<html>...</html>",
                        json.dumps({"format": "html"}),
                        "firmware,analysis",
                        32768,
                        "def456",
                        now,
                        now,
                        0.8,
                        5,
                        artifact_logs,
                    ),
                    (
                        artifact_runbook,
                        workflow_id,
                        action_patch,
                        "plan",
                        "rollback-plan.md",
                        "Step-by-step rollback instructions",
                        "/tmp/rollback-plan.md",
                        "# Rollback plan\n1. Notify ops\n2. Deploy patch",
                        json.dumps({"audience": "oncall"}),
                        "rollback,mitigation",
                        8192,
                        "ghi789",
                        now,
                        now,
                        1.0,
                        2,
                        None,
                    ),
                ],
            )

            # Artifact relationships ------------------------------------
            relationships: List[int] = []
            for relation in [
                (artifact_logs, artifact_diff, "informs"),
                (artifact_diff, artifact_runbook, "supports"),
            ]:
                cur = conn.execute(
                    """
                    INSERT INTO artifact_relationships (
                        source_artifact_id, target_artifact_id, relation_type, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (*relation, now),
                )
                relationships.append(cur.lastrowid)

            # Memories ---------------------------------------------------
            memory_summary = self._uid("memory")
            memory_risk = self._uid("memory")
            memory_next = self._uid("memory")
            memories = [memory_summary, memory_risk, memory_next]
            conn.executemany(
                """
                INSERT INTO memories (
                    memory_id, workflow_id, content, memory_type, memory_level,
                    importance, confidence, description, reasoning, created_at,
                    updated_at, last_accessed_at, access_count, archived
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        memory_summary,
                        workflow_id,
                        "Outage isolated to sensor fusion regression",
                        "summary",
                        "working",
                        9,
                        0.92,
                        "Summary of investigation",
                        "Synthesized from telemetry and diff",
                        now,
                        now,
                        now,
                        4,
                        0,
                    ),
                    (
                        memory_risk,
                        workflow_id,
                        "Risk: rollback requires manual oversight",
                        "risk",
                        "supporting",
                        7,
                        0.7,
                        "Operational risk",
                        "Derived from runbook review",
                        now,
                        now,
                        None,
                        2,
                        0,
                    ),
                    (
                        memory_next,
                        workflow_id,
                        "Next: notify customer success once rollback completes",
                        "task",
                        "supporting",
                        6,
                        0.8,
                        "Communication follow-up",
                        "Action item for stakeholders",
                        now,
                        now,
                        None,
                        1,
                        0,
                    ),
                ],
            )

            # Memory links -----------------------------------------------
            memory_links = [self._uid("link"), self._uid("link")]
            conn.executemany(
                """
                INSERT INTO memory_links (
                    link_id, source_memory_id, target_memory_id, link_type,
                    strength, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        memory_links[0],
                        memory_summary,
                        memory_risk,
                        "supports",
                        0.85,
                        now,
                    ),
                    (
                        memory_links[1],
                        memory_summary,
                        memory_next,
                        "yields",
                        0.75,
                        now,
                    ),
                ],
            )

            # Cognitive timeline states ---------------------------------
            timeline_states = [self._uid("state"), self._uid("state"), self._uid("state")]
            conn.executemany(
                """
                INSERT INTO cognitive_timeline_states (
                    state_id, timestamp, state_type, state_data, workflow_id,
                    description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        timeline_states[0],
                        now - 1200,
                        "observation",
                        json.dumps({"event": "customer_alert"}),
                        workflow_id,
                        "Customer reported outages via Slack",
                        now - 1200,
                    ),
                    (
                        timeline_states[1],
                        now - 600,
                        "analysis",
                        json.dumps({"artifact_id": artifact_diff}),
                        workflow_id,
                        "Firmware regression isolated",
                        now - 600,
                    ),
                    (
                        timeline_states[2],
                        now - 60,
                        "action",
                        json.dumps({"action_id": action_patch}),
                        workflow_id,
                        "Rollback initiated",
                        now - 60,
                    ),
                ],
            )

            conn.commit()
        finally:
            conn.close()

        return StagedStory(
            workflow_id=workflow_id,
            goal_ids=goals,
            action_ids=actions,
            artifact_ids=artifacts,
            memory_ids=memories,
            timeline_state_ids=timeline_states,
            memory_link_ids=memory_links,
            relationship_ids=relationships,
        )

    def get_story_snapshot(self, workflow_id: str) -> StorySnapshot:
        """Return every row that belongs to ``workflow_id``."""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            def _rows(query: str, params: Iterable) -> List[Dict]:
                cur = conn.execute(query, tuple(params))
                return [dict(row) for row in cur.fetchall()]

            workflow_rows = _rows(
                "SELECT * FROM workflows WHERE workflow_id = ?",
                (workflow_id,),
            )
            workflow = workflow_rows[0] if workflow_rows else {}

            goals = _rows(
                "SELECT * FROM goals WHERE workflow_id = ? ORDER BY priority",
                (workflow_id,),
            )
            actions = _rows(
                "SELECT * FROM actions WHERE workflow_id = ? ORDER BY sequence_number",
                (workflow_id,),
            )
            artifacts = _rows(
                "SELECT * FROM artifacts WHERE workflow_id = ? ORDER BY created_at",
                (workflow_id,),
            )
            artifact_relationships: List[Dict] = []
            if artifacts:
                artifact_ids = [artifact["artifact_id"] for artifact in artifacts]
                placeholders = ",".join(["?"] * len(artifact_ids))
                artifact_relationships = _rows(
                    f"""
                    SELECT * FROM artifact_relationships
                    WHERE source_artifact_id IN ({placeholders})
                       OR target_artifact_id IN ({placeholders})
                    ORDER BY id
                    """,
                    tuple(artifact_ids) + tuple(artifact_ids),
                )

            memories = _rows(
                "SELECT * FROM memories WHERE workflow_id = ? ORDER BY importance DESC",
                (workflow_id,),
            )
            memory_links: List[Dict] = []
            if memories:
                memory_ids = [memory["memory_id"] for memory in memories]
                placeholders = ",".join(["?"] * len(memory_ids))
                memory_links = _rows(
                    f"""
                    SELECT * FROM memory_links
                    WHERE source_memory_id IN ({placeholders})
                       OR target_memory_id IN ({placeholders})
                    ORDER BY created_at
                    """,
                    tuple(memory_ids) + tuple(memory_ids),
                )

            cognitive_timeline_states = _rows(
                """
                SELECT * FROM cognitive_timeline_states
                WHERE workflow_id = ?
                ORDER BY timestamp
                """,
                (workflow_id,),
            )

            return StorySnapshot(
                workflow=workflow,
                goals=goals,
                actions=actions,
                artifacts=artifacts,
                artifact_relationships=artifact_relationships,
                memories=memories,
                memory_links=memory_links,
                cognitive_timeline_states=cognitive_timeline_states,
            )
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _uid(prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:8]}"


__all__ = [
    "StagedStory",
    "StorySnapshot",
    "UnifiedMemoryStoryBuilder",
]
