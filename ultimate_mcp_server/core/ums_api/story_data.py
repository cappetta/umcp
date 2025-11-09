"""Utilities for staging and inspecting unified memory story data.

This module seeds the unified agent memory database with a realistic story that
touches every major table in the schema.  The staged story models a complex
agent workflow so API features and tests can exercise the full relational
structure without relying on ad-hoc fixtures.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .ums_database import get_database_path


STORY_WORKFLOW_ID = "workflow_mars_habitat_rescue"

try:  # pragma: no cover - the full initializer requires optional dependencies
    from ultimate_mcp_server.working_memory_api import (
        initialize_unified_memory_schema as _initialize_schema,
    )
except ModuleNotFoundError:  # pragma: no cover - fall back to local schema builder
    _initialize_schema = None
except Exception:  # pragma: no cover - guard against partial import failures
    _initialize_schema = None


def _create_minimal_schema(db_path: str) -> None:
    """Create the subset of tables required for story seeding."""

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                goal TEXT,
                status TEXT,
                created_at REAL,
                updated_at REAL,
                last_active REAL,
                metadata TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS goals (
                goal_id TEXT PRIMARY KEY,
                workflow_id TEXT,
                parent_goal_id TEXT,
                description TEXT,
                status TEXT,
                priority INTEGER,
                created_at REAL,
                updated_at REAL,
                completed_at REAL,
                FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS actions (
                action_id TEXT PRIMARY KEY,
                workflow_id TEXT,
                parent_action_id TEXT,
                action_type TEXT,
                title TEXT,
                description TEXT,
                reasoning TEXT,
                tool_name TEXT,
                tool_args TEXT,
                tool_result TEXT,
                status TEXT,
                created_at REAL,
                started_at REAL,
                completed_at REAL,
                updated_at REAL,
                sequence_number INTEGER,
                priority INTEGER,
                tags TEXT,
                result TEXT,
                metadata TEXT,
                FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                workflow_id TEXT,
                action_id TEXT,
                artifact_type TEXT,
                name TEXT,
                description TEXT,
                file_path TEXT,
                content TEXT,
                metadata TEXT,
                tags TEXT,
                file_size INTEGER,
                checksum TEXT,
                created_at REAL,
                updated_at REAL,
                importance REAL,
                access_count INTEGER,
                parent_artifact_id TEXT,
                FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS artifact_relationships (
                id INTEGER PRIMARY KEY,
                source_artifact_id TEXT,
                target_artifact_id TEXT,
                relation_type TEXT,
                created_at REAL,
                FOREIGN KEY (source_artifact_id) REFERENCES artifacts(artifact_id),
                FOREIGN KEY (target_artifact_id) REFERENCES artifacts(artifact_id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                workflow_id TEXT,
                content TEXT,
                memory_type TEXT,
                memory_level TEXT,
                importance REAL,
                confidence REAL,
                description TEXT,
                reasoning TEXT,
                created_at REAL,
                updated_at REAL,
                last_accessed_at REAL,
                access_count INTEGER,
                archived INTEGER,
                FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_links (
                link_id TEXT PRIMARY KEY,
                source_memory_id TEXT,
                target_memory_id TEXT,
                link_type TEXT,
                strength REAL,
                created_at REAL,
                FOREIGN KEY (source_memory_id) REFERENCES memories(memory_id),
                FOREIGN KEY (target_memory_id) REFERENCES memories(memory_id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cognitive_timeline_states (
                state_id TEXT PRIMARY KEY,
                timestamp REAL,
                state_type TEXT,
                state_data TEXT,
                workflow_id TEXT,
                description TEXT,
                created_at REAL,
                FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id)
            )
            """
        )

        conn.commit()


def _ensure_schema(db_path: str) -> None:
    """Ensure that the required database schema exists."""

    if _initialize_schema is not None:
        _initialize_schema(db_path)
    else:
        _create_minimal_schema(db_path)


def _to_json(value: Any) -> str:
    """Serialize helper that keeps database payloads consistent."""

    if value is None:
        return "{}"
    return json.dumps(value, ensure_ascii=False)


def _delete_existing_story_data(cursor: sqlite3.Cursor) -> None:
    """Remove existing rows for the staged story to keep the operation idempotent."""

    story_ids = {
        "artifact_ids": {
            "artifact_mission_brief",
            "artifact_sensor_package",
            "artifact_rescue_plan",
        },
        "memory_ids": {
            "memory_surface_conditions",
            "memory_power_constraints",
            "memory_success_protocol",
        },
    }

    placeholders = {
        key: ",".join("?" for _ in values)
        for key, values in story_ids.items()
    }

    # Remove relationship/link rows first to avoid foreign key issues.
    if story_ids["artifact_ids"]:
        cursor.execute(
            f"DELETE FROM artifact_relationships WHERE source_artifact_id IN ({placeholders['artifact_ids']}) "
            f"OR target_artifact_id IN ({placeholders['artifact_ids']})",
            tuple(story_ids["artifact_ids"]) * 2,
        )

    if story_ids["memory_ids"]:
        cursor.execute(
            f"DELETE FROM memory_links WHERE source_memory_id IN ({placeholders['memory_ids']}) "
            f"OR target_memory_id IN ({placeholders['memory_ids']})",
            tuple(story_ids["memory_ids"]) * 2,
        )

    cursor.execute(
        "DELETE FROM cognitive_timeline_states WHERE workflow_id = ?",
        (STORY_WORKFLOW_ID,),
    )
    cursor.execute(
        "DELETE FROM artifacts WHERE workflow_id = ?",
        (STORY_WORKFLOW_ID,),
    )
    cursor.execute(
        "DELETE FROM actions WHERE workflow_id = ?",
        (STORY_WORKFLOW_ID,),
    )
    cursor.execute(
        "DELETE FROM memories WHERE workflow_id = ?",
        (STORY_WORKFLOW_ID,),
    )
    cursor.execute(
        "DELETE FROM goals WHERE workflow_id = ?",
        (STORY_WORKFLOW_ID,),
    )
    cursor.execute(
        "DELETE FROM workflows WHERE workflow_id = ?",
        (STORY_WORKFLOW_ID,),
    )


def _apply_inserts(cursor: sqlite3.Cursor, payload: Dict[str, Any]) -> Dict[str, int]:
    """Insert the staged story data and return per-table row counts."""

    counts: Dict[str, int] = {}

    cursor.execute(
        """
        INSERT INTO workflows (
            workflow_id, title, description, goal, status,
            created_at, updated_at, last_active, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload["workflow"]["workflow_id"],
            payload["workflow"]["title"],
            payload["workflow"].get("description"),
            payload["workflow"].get("goal"),
            payload["workflow"].get("status"),
            payload["workflow"].get("created_at"),
            payload["workflow"].get("updated_at"),
            payload["workflow"].get("last_active"),
            _to_json(payload["workflow"].get("metadata")),
        ),
    )
    counts["workflows"] = 1

    cursor.executemany(
        """
        INSERT INTO goals (
            goal_id, workflow_id, parent_goal_id, description, status,
            priority, created_at, updated_at, completed_at
        ) VALUES (:goal_id, :workflow_id, :parent_goal_id, :description, :status,
                  :priority, :created_at, :updated_at, :completed_at)
        """,
        payload["goals"],
    )
    counts["goals"] = len(payload["goals"])

    cursor.executemany(
        """
        INSERT INTO actions (
            action_id, workflow_id, parent_action_id, action_type, title,
            description, reasoning, tool_name, tool_args, tool_result,
            status, created_at, started_at, completed_at, updated_at,
            sequence_number, priority, tags, result, metadata
        ) VALUES (:action_id, :workflow_id, :parent_action_id, :action_type, :title,
                  :description, :reasoning, :tool_name, :tool_args, :tool_result,
                  :status, :created_at, :started_at, :completed_at, :updated_at,
                  :sequence_number, :priority, :tags, :result, :metadata)
        """,
        payload["actions"],
    )
    counts["actions"] = len(payload["actions"])

    cursor.executemany(
        """
        INSERT INTO artifacts (
            artifact_id, workflow_id, action_id, artifact_type, name,
            description, file_path, content, metadata, tags,
            file_size, checksum, created_at, updated_at, importance,
            access_count, parent_artifact_id
        ) VALUES (:artifact_id, :workflow_id, :action_id, :artifact_type, :name,
                  :description, :file_path, :content, :metadata, :tags,
                  :file_size, :checksum, :created_at, :updated_at, :importance,
                  :access_count, :parent_artifact_id)
        """,
        payload["artifacts"],
    )
    counts["artifacts"] = len(payload["artifacts"])

    cursor.executemany(
        """
        INSERT INTO artifact_relationships (
            id, source_artifact_id, target_artifact_id, relation_type, created_at
        ) VALUES (:id, :source_artifact_id, :target_artifact_id, :relation_type, :created_at)
        """,
        payload["artifact_relationships"],
    )
    counts["artifact_relationships"] = len(payload["artifact_relationships"])

    cursor.executemany(
        """
        INSERT INTO memories (
            memory_id, workflow_id, content, memory_type, memory_level,
            importance, confidence, description, reasoning,
            created_at, updated_at, last_accessed_at, access_count, archived
        ) VALUES (:memory_id, :workflow_id, :content, :memory_type, :memory_level,
                  :importance, :confidence, :description, :reasoning,
                  :created_at, :updated_at, :last_accessed_at, :access_count, :archived)
        """,
        payload["memories"],
    )
    counts["memories"] = len(payload["memories"])

    cursor.executemany(
        """
        INSERT INTO memory_links (
            link_id, source_memory_id, target_memory_id, link_type, strength, created_at
        ) VALUES (:link_id, :source_memory_id, :target_memory_id, :link_type, :strength, :created_at)
        """,
        payload["memory_links"],
    )
    counts["memory_links"] = len(payload["memory_links"])

    cursor.executemany(
        """
        INSERT INTO cognitive_timeline_states (
            state_id, timestamp, state_type, state_data, workflow_id, description, created_at
        ) VALUES (:state_id, :timestamp, :state_type, :state_data, :workflow_id, :description, :created_at)
        """,
        payload["cognitive_timeline_states"],
    )
    counts["cognitive_timeline_states"] = len(payload["cognitive_timeline_states"])

    return counts


def _build_story_payload() -> Dict[str, Any]:
    """Construct the story payload using the current timestamp."""

    now = time.time()
    hour = 3600
    workflow_metadata = {
        "mission_code": "ARES-7",
        "location": "Valles Marineris Base",
        "crew": ["Ava", "Dr. Rami", "Sera"],
    }

    workflow = {
        "workflow_id": STORY_WORKFLOW_ID,
        "title": "Mars Habitat Rescue Operations",
        "description": "Stabilise the damaged Mars habitat and secure the science payload.",
        "goal": "Restore habitat systems and extract critical research samples.",
        "status": "in_progress",
        "created_at": now - (6 * hour),
        "updated_at": now,
        "last_active": now,
        "metadata": workflow_metadata,
    }

    goals = [
        {
            "goal_id": "goal_restore_habitat",
            "workflow_id": STORY_WORKFLOW_ID,
            "parent_goal_id": None,
            "description": "Stabilise life-support and power infrastructure.",
            "status": "in_progress",
            "priority": 1,
            "created_at": now - (6 * hour),
            "updated_at": now - (2 * hour),
            "completed_at": None,
        },
        {
            "goal_id": "goal_patch_power_grid",
            "workflow_id": STORY_WORKFLOW_ID,
            "parent_goal_id": "goal_restore_habitat",
            "description": "Repair solar array regulators to restore grid stability.",
            "status": "completed",
            "priority": 2,
            "created_at": now - (5 * hour),
            "updated_at": now - (3 * hour),
            "completed_at": now - (3 * hour),
        },
        {
            "goal_id": "goal_secure_samples",
            "workflow_id": STORY_WORKFLOW_ID,
            "parent_goal_id": "goal_restore_habitat",
            "description": "Catalog and secure regolith biofilm samples for evac transport.",
            "status": "in_progress",
            "priority": 3,
            "created_at": now - (4 * hour),
            "updated_at": now - hour,
            "completed_at": None,
        },
    ]

    actions = [
        {
            "action_id": "action_survey_damage",
            "workflow_id": STORY_WORKFLOW_ID,
            "parent_action_id": None,
            "action_type": "analysis",
            "title": "Survey structural damage",
            "description": "Scan habitat sectors and quantify breach severity.",
            "reasoning": "Need baseline before deploying repair drones.",
            "tool_name": "lidar_mapper",
            "tool_args": _to_json({"resolution": "5cm"}),
            "tool_result": "High fidelity sector map generated.",
            "status": "completed",
            "created_at": now - (5 * hour),
            "started_at": now - (5 * hour),
            "completed_at": now - (4.5 * hour),
            "updated_at": now - (4 * hour),
            "sequence_number": 1,
            "priority": 1,
            "tags": json.dumps(["survey", "critical"]),
            "result": "Sector map stored as artifact_mission_brief",
            "metadata": _to_json({"operator": "Ava"}),
        },
        {
            "action_id": "action_drone_patch",
            "workflow_id": STORY_WORKFLOW_ID,
            "parent_action_id": "action_survey_damage",
            "action_type": "execution",
            "title": "Deploy repair drones",
            "description": "Coordinate drones to seal breaches and reinforce panels.",
            "reasoning": "Seal breaches before nightfall temperature drop.",
            "tool_name": "drone_controller",
            "tool_args": _to_json({"drones": 3, "mode": "auto_patch"}),
            "tool_result": "Critical breaches patched, residual micro-fractures logged.",
            "status": "completed",
            "created_at": now - (4 * hour),
            "started_at": now - (4 * hour),
            "completed_at": now - (3.5 * hour),
            "updated_at": now - (3 * hour),
            "sequence_number": 2,
            "priority": 1,
            "tags": json.dumps(["repair", "priority"]),
            "result": "Repair drones successful",
            "metadata": _to_json({"operator": "Dr. Rami"}),
        },
        {
            "action_id": "action_plan_evac",
            "workflow_id": STORY_WORKFLOW_ID,
            "parent_action_id": None,
            "action_type": "planning",
            "title": "Draft evacuation plan",
            "description": "Create contingency for emergency extraction of crew and samples.",
            "reasoning": "Need actionable plan if systems fail again.",
            "tool_name": "tactical_planner",
            "tool_args": _to_json({"scenario": "night_cycle"}),
            "tool_result": "Evacuation decision tree created.",
            "status": "in_progress",
            "created_at": now - (2 * hour),
            "started_at": now - (2 * hour),
            "completed_at": None,
            "updated_at": now - (0.5 * hour),
            "sequence_number": 3,
            "priority": 2,
            "tags": json.dumps(["planning"]),
            "result": "Awaiting review",
            "metadata": _to_json({"operator": "Sera"}),
        },
    ]

    artifacts = [
        {
            "artifact_id": "artifact_mission_brief",
            "workflow_id": STORY_WORKFLOW_ID,
            "action_id": "action_survey_damage",
            "artifact_type": "report",
            "name": "Structural survey briefing",
            "description": "Composite scan of habitat structural integrity.",
            "file_path": "artifacts/mission_brief.md",
            "content": "Sector by sector analysis with heat-map overlays.",
            "metadata": _to_json({"format": "markdown", "sections": 6}),
            "tags": json.dumps(["survey", "analysis"]),
            "file_size": 32_768,
            "checksum": "cb734ff9",
            "created_at": now - (4.5 * hour),
            "updated_at": now - (4 * hour),
            "importance": 0.85,
            "access_count": 7,
            "parent_artifact_id": None,
        },
        {
            "artifact_id": "artifact_sensor_package",
            "workflow_id": STORY_WORKFLOW_ID,
            "action_id": "action_drone_patch",
            "artifact_type": "dataset",
            "name": "Drone sensor telemetry",
            "description": "Telemetry logs of drone patch operations and sensor readings.",
            "file_path": "artifacts/sensor_package.json",
            "content": json.dumps({"thermal_variance": 0.12, "pressure_delta": 0.03}),
            "metadata": _to_json({"format": "json", "samples": 1500}),
            "tags": json.dumps(["telemetry", "repair"]),
            "file_size": 48_576,
            "checksum": "f2a9b00d",
            "created_at": now - (3.5 * hour),
            "updated_at": now - (3 * hour),
            "importance": 0.92,
            "access_count": 11,
            "parent_artifact_id": None,
        },
        {
            "artifact_id": "artifact_rescue_plan",
            "workflow_id": STORY_WORKFLOW_ID,
            "action_id": "action_plan_evac",
            "artifact_type": "document",
            "name": "Evacuation protocol draft",
            "description": "Decision tree covering extraction options and fallback scenarios.",
            "file_path": "artifacts/rescue_plan.pdf",
            "content": "PDF binary placeholder",
            "metadata": _to_json({"format": "pdf", "pages": 12}),
            "tags": json.dumps(["plan", "evacuation"]),
            "file_size": 128_000,
            "checksum": "dd12aa45",
            "created_at": now - hour,
            "updated_at": now - (0.5 * hour),
            "importance": 0.78,
            "access_count": 2,
            "parent_artifact_id": "artifact_mission_brief",
        },
    ]

    artifact_relationships = [
        {
            "id": 9001,
            "source_artifact_id": "artifact_rescue_plan",
            "target_artifact_id": "artifact_mission_brief",
            "relation_type": "derives_from",
            "created_at": now - hour,
        },
        {
            "id": 9002,
            "source_artifact_id": "artifact_rescue_plan",
            "target_artifact_id": "artifact_sensor_package",
            "relation_type": "references",
            "created_at": now - hour,
        },
    ]

    memories = [
        {
            "memory_id": "memory_surface_conditions",
            "workflow_id": STORY_WORKFLOW_ID,
            "content": "Surface storm approaching from east crater rim.",
            "memory_type": "episodic",
            "memory_level": "short_term",
            "importance": 0.8,
            "confidence": 0.7,
            "description": "Weather advisory from orbital sensors.",
            "reasoning": "Impacts scheduling for drone deployments.",
            "created_at": now - (4 * hour),
            "updated_at": now - (3.5 * hour),
            "last_accessed_at": now - hour,
            "access_count": 5,
            "archived": 0,
        },
        {
            "memory_id": "memory_power_constraints",
            "workflow_id": STORY_WORKFLOW_ID,
            "content": "Battery reserves capped at 40% during patch cycle.",
            "memory_type": "episodic",
            "memory_level": "working",
            "importance": 0.9,
            "confidence": 0.8,
            "description": "Power regulator output logs.",
            "reasoning": "Determines which subsystems to triage.",
            "created_at": now - (3.5 * hour),
            "updated_at": now - (3 * hour),
            "last_accessed_at": now - (2 * hour),
            "access_count": 7,
            "archived": 0,
        },
        {
            "memory_id": "memory_success_protocol",
            "workflow_id": STORY_WORKFLOW_ID,
            "content": "Mission success requires retrieval of biofilm samples A12 and C07.",
            "memory_type": "semantic",
            "memory_level": "long_term",
            "importance": 0.95,
            "confidence": 0.9,
            "description": "Success criteria documented in mission charter.",
            "reasoning": "Guides prioritisation of evacuation payload.",
            "created_at": now - (6 * hour),
            "updated_at": now - (2 * hour),
            "last_accessed_at": now - (0.5 * hour),
            "access_count": 9,
            "archived": 0,
        },
    ]

    memory_links = [
        {
            "link_id": "link_weather_to_power",
            "source_memory_id": "memory_surface_conditions",
            "target_memory_id": "memory_power_constraints",
            "link_type": "causal",
            "strength": 0.75,
            "created_at": now - (3 * hour),
        },
        {
            "link_id": "link_power_to_success",
            "source_memory_id": "memory_power_constraints",
            "target_memory_id": "memory_success_protocol",
            "link_type": "supports",
            "strength": 0.68,
            "created_at": now - (2 * hour),
        },
    ]

    timeline_states = [
        {
            "state_id": "state_initial_alert",
            "timestamp": now - (5.5 * hour),
            "state_type": "alert",
            "state_data": _to_json({"severity": "orange", "trigger": "pressure_loss"}),
            "workflow_id": STORY_WORKFLOW_ID,
            "description": "Initial pressure loss alert received.",
            "created_at": now - (5.5 * hour),
        },
        {
            "state_id": "state_repair_ops",
            "timestamp": now - (3.25 * hour),
            "state_type": "operation",
            "state_data": _to_json({"drones_active": 3, "patched_panels": 12}),
            "workflow_id": STORY_WORKFLOW_ID,
            "description": "Repair drones sealing breaches.",
            "created_at": now - (3.25 * hour),
        },
        {
            "state_id": "state_planning_review",
            "timestamp": now - (0.75 * hour),
            "state_type": "planning",
            "state_data": _to_json({"evacuation_ready": False, "pending_tasks": 4}),
            "workflow_id": STORY_WORKFLOW_ID,
            "description": "Evacuation plan under review by command.",
            "created_at": now - (0.75 * hour),
        },
    ]

    return {
        "workflow": workflow,
        "goals": goals,
        "actions": actions,
        "artifacts": artifacts,
        "artifact_relationships": artifact_relationships,
        "memories": memories,
        "memory_links": memory_links,
        "cognitive_timeline_states": timeline_states,
    }


def stage_sample_story(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Stage the unified memory story in the SQLite database.

    Args:
        db_path: Optional override path for the database file.  Defaults to the
            configured unified memory database location.

    Returns:
        A dictionary with the workflow payload and per-table row counts.
    """

    database_path = db_path or get_database_path()
    _ensure_schema(database_path)

    payload = _build_story_payload()

    with closing(sqlite3.connect(database_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        _delete_existing_story_data(cursor)
        counts = _apply_inserts(cursor, payload)
        conn.commit()

    return {"workflow": payload["workflow"], "counts": counts}


def _fetch_rows(cursor: sqlite3.Cursor, query: str, params: Iterable[Any]) -> list[dict[str, Any]]:
    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    return [dict(row) for row in rows]


def fetch_story_snapshot(
    db_path: Optional[str] = None,
    *,
    workflow_id: str = STORY_WORKFLOW_ID,
) -> Optional[Dict[str, Any]]:
    """Return the staged story data or ``None`` if it has not been seeded."""

    database_path = db_path or get_database_path()
    if not Path(database_path).exists():
        return None

    with closing(sqlite3.connect(database_path)) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
        )
        workflow_row = cursor.fetchone()
        if workflow_row is None:
            return None

        workflow = dict(workflow_row)
        workflow["metadata"] = json.loads(workflow.get("metadata") or "{}")

        goals = _fetch_rows(
            cursor,
            "SELECT * FROM goals WHERE workflow_id = ? ORDER BY created_at",
            (workflow_id,),
        )

        actions = _fetch_rows(
            cursor,
            "SELECT * FROM actions WHERE workflow_id = ? ORDER BY sequence_number",
            (workflow_id,),
        )
        for action in actions:
            action["metadata"] = json.loads(action.get("metadata") or "{}")
            action["tags"] = json.loads(action.get("tags") or "[]")
            action["tool_args"] = json.loads(action.get("tool_args") or "{}")

        artifacts = _fetch_rows(
            cursor,
            "SELECT * FROM artifacts WHERE workflow_id = ? ORDER BY created_at",
            (workflow_id,),
        )
        for artifact in artifacts:
            artifact["metadata"] = json.loads(artifact.get("metadata") or "{}")
            artifact["tags"] = json.loads(artifact.get("tags") or "[]")

        artifact_ids = [artifact["artifact_id"] for artifact in artifacts]
        artifact_relationships: list[dict[str, Any]] = []
        if artifact_ids:
            artifact_relationships = _fetch_rows(
                cursor,
                """
                SELECT * FROM artifact_relationships
                WHERE source_artifact_id IN ({})
                   OR target_artifact_id IN ({})
                ORDER BY id
                """.format(
                    ",".join("?" for _ in artifact_ids),
                    ",".join("?" for _ in artifact_ids),
                ),
                tuple(artifact_ids) + tuple(artifact_ids),
            )

        memories = _fetch_rows(
            cursor,
            "SELECT * FROM memories WHERE workflow_id = ? ORDER BY created_at",
            (workflow_id,),
        )

        memory_ids = [memory["memory_id"] for memory in memories]
        memory_links: list[dict[str, Any]] = []
        if memory_ids:
            memory_links = _fetch_rows(
                cursor,
                """
                SELECT * FROM memory_links
                WHERE source_memory_id IN ({})
                   OR target_memory_id IN ({})
                ORDER BY created_at
                """.format(
                    ",".join("?" for _ in memory_ids),
                    ",".join("?" for _ in memory_ids),
                ),
                tuple(memory_ids) + tuple(memory_ids),
            )

        timeline_states = _fetch_rows(
            cursor,
            """
            SELECT * FROM cognitive_timeline_states
            WHERE workflow_id = ?
            ORDER BY timestamp
            """,
            (workflow_id,),
        )
        for state in timeline_states:
            state["state_data"] = json.loads(state.get("state_data") or "{}")

    return {
        "workflow": workflow,
        "goals": goals,
        "actions": actions,
        "artifacts": artifacts,
        "artifact_relationships": artifact_relationships,
        "memories": memories,
        "memory_links": memory_links,
        "cognitive_timeline_states": timeline_states,
    }
