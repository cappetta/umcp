"""Story seeding utilities for the Unified Memory System (UMS) database."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - import path exercised when FastAPI is installed
    from ultimate_mcp_server.working_memory_api import (  # type: ignore
        initialize_unified_memory_schema as _initialize_unified_memory_schema,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs

    def initialize_unified_memory_schema(db_path: str) -> None:  # type: ignore
        """Fallback schema initializer covering core tables required for tests."""

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
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
                    tool_data TEXT,
                    status TEXT,
                    created_at REAL,
                    started_at REAL,
                    completed_at REAL,
                    updated_at REAL,
                    sequence_number INTEGER,
                    priority INTEGER,
                    tags TEXT,
                    result TEXT,
                    error TEXT,
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
                    access_count INTEGER DEFAULT 0,
                    parent_artifact_id TEXT,
                    FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id),
                    FOREIGN KEY (action_id) REFERENCES actions(action_id)
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS artifact_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                    content TEXT NOT NULL,
                    memory_type TEXT,
                    memory_level TEXT,
                    importance REAL DEFAULT 1,
                    confidence REAL DEFAULT 0.5,
                    description TEXT,
                    reasoning TEXT,
                    created_at REAL,
                    updated_at REAL,
                    last_accessed_at REAL,
                    access_count INTEGER DEFAULT 0,
                    archived INTEGER DEFAULT 0,
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
                    strength REAL DEFAULT 1,
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
        finally:
            conn.close()

    _initialize_unified_memory_schema = initialize_unified_memory_schema
else:
    initialize_unified_memory_schema = _initialize_unified_memory_schema

from .ums_database import get_database_path


STORY_WORKFLOW_ID = "workflow_mission_cerberus"


class StoryAlreadyExistsError(RuntimeError):
    """Raised when attempting to seed a story that already exists."""


class StoryNotFoundError(RuntimeError):
    """Raised when a workflow story cannot be located in the database."""


def _parse_json(value: Optional[str]) -> Any:
    """Safely parse a JSON string if provided."""

    if value in (None, "", b""):
        return None
    try:
        return json.loads(value)  # type: ignore[arg-type]
    except (TypeError, json.JSONDecodeError):
        return value


def _open_connection(db_path: Optional[str]) -> Tuple[sqlite3.Connection, bool]:
    """Open a SQLite connection returning the connection and whether we created it."""

    if db_path:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn, True
    conn = sqlite3.connect(get_database_path())
    conn.row_factory = sqlite3.Row
    return conn, True


def _clear_existing_story(cursor: sqlite3.Cursor, workflow_id: str) -> None:
    """Remove all rows tied to a workflow to allow reseeding."""

    cursor.execute(
        "DELETE FROM memory_links WHERE source_memory_id IN ("
        "SELECT memory_id FROM memories WHERE workflow_id = ?) "
        "OR target_memory_id IN (SELECT memory_id FROM memories WHERE workflow_id = ?)",
        (workflow_id, workflow_id),
    )
    cursor.execute(
        "DELETE FROM artifact_relationships WHERE source_artifact_id IN ("
        "SELECT artifact_id FROM artifacts WHERE workflow_id = ?) "
        "OR target_artifact_id IN (SELECT artifact_id FROM artifacts WHERE workflow_id = ?)",
        (workflow_id, workflow_id),
    )
    cursor.execute(
        "DELETE FROM artifacts WHERE workflow_id = ?",
        (workflow_id,),
    )
    cursor.execute(
        "DELETE FROM actions WHERE workflow_id = ?",
        (workflow_id,),
    )
    cursor.execute(
        "DELETE FROM goals WHERE workflow_id = ?",
        (workflow_id,),
    )
    cursor.execute(
        "DELETE FROM memories WHERE workflow_id = ?",
        (workflow_id,),
    )
    cursor.execute(
        "DELETE FROM cognitive_timeline_states WHERE workflow_id = ?",
        (workflow_id,),
    )
    cursor.execute(
        "DELETE FROM workflows WHERE workflow_id = ?",
        (workflow_id,),
    )


@dataclass(frozen=True)
class StorySeedResult:
    """Summary of seeded rows for the demo story."""

    workflow_id: str
    created: bool
    counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "created": self.created,
            "counts": self.counts,
        }


def seed_unified_memory_story(
    db_path: Optional[str] = None,
    *,
    force: bool = False,
) -> StorySeedResult:
    """Populate the UMS database with a cohesive narrative across all core tables."""

    initialize_unified_memory_schema(db_path or get_database_path())
    conn, _ = _open_connection(db_path)

    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")

        cursor.execute(
            "SELECT 1 FROM workflows WHERE workflow_id = ?",
            (STORY_WORKFLOW_ID,),
        )
        exists = cursor.fetchone() is not None

        if exists and not force:
            raise StoryAlreadyExistsError(
                "Mission Cerberus story already exists. Use force=True to reseed."
            )

        if exists and force:
            _clear_existing_story(cursor, STORY_WORKFLOW_ID)

        base_time = 1_700_000_000.0
        now = base_time + 7200

        workflow_metadata = {
            "mission_phase": "habitat-construction",
            "location": "Cerberus Fossae, Mars",
            "crew": ["Navarro", "Ito", "Singh"],
            "mission_clock_hours": 96,
        }

        cursor.execute(
            "INSERT INTO workflows (workflow_id, title, description, goal, status, "
            "created_at, updated_at, last_active, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                STORY_WORKFLOW_ID,
                "Mission Cerberus: Build Mars Research Outpost",
                "Coordinating the establishment of a research habitat on Mars's Cerberus Fossae region.",
                "Deploy a self-sustaining outpost with power, habitat modules, and research ops in 72 hours.",
                "active",
                base_time - 3600,
                now,
                now,
                json.dumps(workflow_metadata),
            ),
        )

        goals = [
            (
                "goal_outpost_ready",
                None,
                "Deliver an operational Mars outpost",
                "in_progress",
                1,
                base_time - 3500,
                now - 600,
                None,
            ),
            (
                "goal_power_grid",
                "goal_outpost_ready",
                "Stabilize geothermal energy supply",
                "in_progress",
                2,
                base_time - 3400,
                now - 400,
                None,
            ),
            (
                "goal_habitat_layout",
                "goal_outpost_ready",
                "Finalize habitat layout and life-support checks",
                "completed",
                2,
                base_time - 3300,
                now - 900,
                now - 800,
            ),
            (
                "goal_research_sync",
                "goal_outpost_ready",
                "Coordinate first wave of research experiments",
                "pending",
                3,
                base_time - 3200,
                base_time - 1800,
                None,
            ),
        ]

        cursor.executemany(
            "INSERT INTO goals (goal_id, workflow_id, parent_goal_id, description, status, "
            "priority, created_at, updated_at, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    goal_id,
                    STORY_WORKFLOW_ID,
                    parent_id,
                    description,
                    status,
                    priority,
                    created_at,
                    updated_at,
                    completed_at,
                )
                for (
                    goal_id,
                    parent_id,
                    description,
                    status,
                    priority,
                    created_at,
                    updated_at,
                    completed_at,
                ) in goals
            ],
        )

        actions = [
            {
                "action_id": "action_site_analysis",
                "action_type": "analysis",
                "title": "Survey Cerberus fissures",
                "description": "Evaluate seismic stability and thermal readings for base placement.",
                "reasoning": "Stable fissures reduce risk of thermal surges during drilling.",
                "tool_name": "orbital_mapper",
                "tool_args": {"resolution": "high", "sweep": "15km radius"},
                "tool_result": {"hazard_zones": ["Sector D3"], "recommended_sites": ["Astra Plain"]},
                "status": "completed",
                "created_at": base_time - 3000,
                "started_at": base_time - 2980,
                "completed_at": base_time - 2910,
                "updated_at": base_time - 2910,
                "sequence_number": 1,
                "priority": 2,
                "tags": ["survey", "geology"],
                "result": "Selected Astra Plain as primary deployment zone.",
                "metadata": {"execution_duration": 70, "confidence": 0.92},
            },
            {
                "action_id": "action_habitat_design",
                "action_type": "planning",
                "title": "Draft modular habitat layout",
                "description": "Design habitat modules with redundancy for dust storms.",
                "reasoning": "Modular configuration eases expansion and maintenance.",
                "tool_name": "habitat_cad",
                "tool_args": {"modules": 4, "redundancy": "dual"},
                "tool_result": {"layout": "ring", "pressure_hulls": 2},
                "status": "completed",
                "created_at": base_time - 2800,
                "started_at": base_time - 2785,
                "completed_at": base_time - 2700,
                "updated_at": base_time - 2700,
                "sequence_number": 2,
                "priority": 2,
                "tags": ["habitat", "design"],
                "result": "Approved ring layout with dual redundancy.",
                "metadata": {"execution_duration": 85, "confidence": 0.88},
            },
            {
                "action_id": "action_power_grid",
                "action_type": "operations",
                "title": "Calibrate geothermal taps",
                "description": "Balance geothermal wells to stabilize microgrid output.",
                "reasoning": "Stable power is required before life-support activation.",
                "tool_name": "geothermal_controller",
                "tool_args": {"wells": 3, "target_output_mw": 12},
                "tool_result": {"output_mw": 10.7, "alerts": ["Well Beta variance"]},
                "status": "in_progress",
                "created_at": base_time - 2600,
                "started_at": base_time - 2575,
                "completed_at": None,
                "updated_at": now - 300,
                "sequence_number": 3,
                "priority": 1,
                "tags": ["power", "critical"],
                "result": "Output stabilizing but Beta well requires monitoring.",
                "metadata": {"execution_duration": 120, "confidence": 0.74},
            },
            {
                "action_id": "action_research_sync",
                "action_type": "coordination",
                "title": "Coordinate experiments",
                "description": "Align first-week research experiments with available resources.",
                "reasoning": "Ensures power and habitat constraints are respected.",
                "tool_name": "ops_scheduler",
                "tool_args": {"teams": 3, "windows": ["sol-5", "sol-6"]},
                "tool_result": None,
                "status": "pending",
                "created_at": base_time - 2400,
                "started_at": None,
                "completed_at": None,
                "updated_at": base_time - 2400,
                "sequence_number": 4,
                "priority": 3,
                "tags": ["coordination"],
                "result": None,
                "metadata": {"execution_duration": None, "confidence": 0.0},
            },
        ]

        cursor.executemany(
            "INSERT INTO actions (action_id, workflow_id, parent_action_id, action_type, title, "
            "description, reasoning, tool_name, tool_args, tool_result, tool_data, status, created_at, "
            "started_at, completed_at, updated_at, sequence_number, priority, tags, result, error, metadata)"
            " VALUES (:action_id, :workflow_id, :parent_action_id, :action_type, :title, :description, :reasoning, :tool_name, :tool_args, :tool_result, :tool_data, :status, :created_at, :started_at, :completed_at, :updated_at, :sequence_number, :priority, :tags, :result, :error, :metadata)",
            [
                {
                    **action,
                    "workflow_id": STORY_WORKFLOW_ID,
                    "parent_action_id": None,
                    "tool_args": json.dumps(action["tool_args"]),
                    "tool_result": json.dumps(action["tool_result"]) if action["tool_result"] is not None else None,
                    "tool_data": None,
                    "tags": json.dumps(action["tags"]),
                    "result": action["result"],
                    "error": None,
                    "metadata": json.dumps(action["metadata"]),
                }
                for action in actions
            ],
        )

        artifacts = [
            {
                "artifact_id": "artifact_topography_map",
                "action_id": "action_site_analysis",
                "artifact_type": "report",
                "name": "Topography and Thermal Map",
                "description": "Thermal overlays identifying safe drilling zones.",
                "file_path": "/mission/cerberus/maps/topography_v3.png",
                "content": None,
                "metadata": {"resolution": "0.5m", "generated_by": "orbital_mapper"},
                "tags": ["map", "thermal"],
                "file_size": 5242880,
                "checksum": "abc123",
                "created_at": base_time - 2950,
                "updated_at": base_time - 2910,
                "importance": 8.5,
                "access_count": 4,
                "parent_artifact_id": None,
            },
            {
                "artifact_id": "artifact_habitat_blueprint",
                "action_id": "action_habitat_design",
                "artifact_type": "design",
                "name": "Habitat Module Blueprint",
                "description": "Ring layout with dual pressure hull redundancy.",
                "file_path": "/mission/cerberus/designs/habitat_blueprint.json",
                "content": json.dumps({"layout": "ring", "modules": ["lab", "med", "quarters"]}),
                "metadata": {"cad_version": "2025.2", "reviewers": ["Ito"]},
                "tags": ["habitat", "layout"],
                "file_size": 786432,
                "checksum": "def456",
                "created_at": base_time - 2720,
                "updated_at": base_time - 2700,
                "importance": 9.1,
                "access_count": 6,
                "parent_artifact_id": None,
            },
            {
                "artifact_id": "artifact_power_report",
                "action_id": "action_power_grid",
                "artifact_type": "log",
                "name": "Geothermal Output Report",
                "description": "Minute-by-minute output readings for geothermal wells.",
                "file_path": "/mission/cerberus/logs/power_output.csv",
                "content": None,
                "metadata": {"well_ids": ["Alpha", "Beta", "Gamma"], "anomaly": "Beta variance"},
                "tags": ["power", "log"],
                "file_size": 262144,
                "checksum": "ghi789",
                "created_at": now - 1800,
                "updated_at": now - 300,
                "importance": 9.5,
                "access_count": 3,
                "parent_artifact_id": None,
            },
            {
                "artifact_id": "artifact_mission_log",
                "action_id": "action_power_grid",
                "artifact_type": "note",
                "name": "Ops Shift Log",
                "description": "Shift log summarizing power calibration decisions.",
                "file_path": None,
                "content": "Beta well showing 6% variance; ramp down 2% for stability.",
                "metadata": {"author": "Navarro", "shift": "sol-3"},
                "tags": ["log", "operations"],
                "file_size": None,
                "checksum": None,
                "created_at": now - 1500,
                "updated_at": now - 1500,
                "importance": 7.2,
                "access_count": 2,
                "parent_artifact_id": "artifact_power_report",
            },
        ]

        cursor.executemany(
            "INSERT INTO artifacts (artifact_id, workflow_id, action_id, artifact_type, name, description, "
            "file_path, content, metadata, tags, file_size, checksum, created_at, updated_at, importance, "
            "access_count, parent_artifact_id) VALUES (:artifact_id, :workflow_id, :action_id, :artifact_type, :name, :description, :file_path, :content, :metadata, :tags, :file_size, :checksum, :created_at, :updated_at, :importance, :access_count, :parent_artifact_id)",
            [
                {
                    **artifact,
                    "workflow_id": STORY_WORKFLOW_ID,
                    "metadata": json.dumps(artifact["metadata"]),
                    "tags": json.dumps(artifact["tags"]),
                }
                for artifact in artifacts
            ],
        )

        artifact_relationships = [
            ("artifact_topography_map", "artifact_habitat_blueprint", "informs", base_time - 2700),
            ("artifact_habitat_blueprint", "artifact_power_report", "depends_on", now - 1700),
            ("artifact_power_report", "artifact_mission_log", "summarized_by", now - 1600),
        ]

        cursor.executemany(
            "INSERT INTO artifact_relationships (source_artifact_id, target_artifact_id, relation_type, created_at)"
            " VALUES (?, ?, ?, ?)",
            artifact_relationships,
        )

        memories = [
            {
                "memory_id": "memory_site_selection",
                "content": "Astra Plain chosen for outpost due to stable seismic readings and thermal yield.",
                "memory_type": "observation",
                "memory_level": "working",
                "importance": 8.7,
                "confidence": 0.93,
                "description": "Site selection rationale",
                "reasoning": "Combines orbital mapper data with historical quake logs.",
                "created_at": base_time - 2900,
                "updated_at": base_time - 2895,
                "last_accessed_at": now - 1200,
                "access_count": 5,
                "archived": 0,
            },
            {
                "memory_id": "memory_habitat_tradeoff",
                "content": "Ring layout accepted; sacrifices rapid expansion for redundancy.",
                "memory_type": "decision",
                "memory_level": "working",
                "importance": 8.2,
                "confidence": 0.88,
                "description": "Habitat trade-off analysis",
                "reasoning": "Crew prioritized survivability over expansion speed.",
                "created_at": base_time - 2750,
                "updated_at": base_time - 2710,
                "last_accessed_at": now - 1400,
                "access_count": 4,
                "archived": 0,
            },
            {
                "memory_id": "memory_power_alert",
                "content": "Beta well variance persists; risk of cascade if output exceeds 11MW.",
                "memory_type": "alert",
                "memory_level": "long_term",
                "importance": 9.6,
                "confidence": 0.71,
                "description": "Power system alert",
                "reasoning": "Variance correlates with subsurface ice pocket; monitor closely.",
                "created_at": now - 1900,
                "updated_at": now - 1800,
                "last_accessed_at": now - 200,
                "access_count": 7,
                "archived": 0,
            },
        ]

        cursor.executemany(
            "INSERT INTO memories (memory_id, workflow_id, content, memory_type, memory_level, importance, confidence, "
            "description, reasoning, created_at, updated_at, last_accessed_at, access_count, archived)"
            " VALUES (:memory_id, :workflow_id, :content, :memory_type, :memory_level, :importance, :confidence, :description, :reasoning, :created_at, :updated_at, :last_accessed_at, :access_count, :archived)",
            [
                {**memory, "workflow_id": STORY_WORKFLOW_ID}
                for memory in memories
            ],
        )

        memory_links = [
            ("link_site_to_habitat", "memory_site_selection", "memory_habitat_tradeoff", "supports", 0.82, base_time - 2700),
            ("link_habitat_to_power", "memory_habitat_tradeoff", "memory_power_alert", "causes", 0.61, now - 1850),
            ("link_site_to_power", "memory_site_selection", "memory_power_alert", "context", 0.74, now - 1700),
        ]

        cursor.executemany(
            "INSERT INTO memory_links (link_id, source_memory_id, target_memory_id, link_type, strength, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            memory_links,
        )

        timeline_states = [
            (
                "state_initial_survey",
                base_time - 2950,
                "analysis",
                json.dumps(
                    {
                        "focus": "geology",
                        "status": "completed",
                        "summary": "Orbital mapper identified stable deployment sites.",
                    }
                ),
                "Survey stabilised base locations",
                base_time - 2950,
            ),
            (
                "state_habitat_planning",
                base_time - 2720,
                "planning",
                json.dumps(
                    {
                        "focus": "habitat",
                        "status": "completed",
                        "summary": "Habitat design approved with redundancy prioritised.",
                    }
                ),
                "Habitat blueprint locked",
                base_time - 2720,
            ),
            (
                "state_power_watch",
                now - 1500,
                "monitoring",
                json.dumps(
                    {
                        "focus": "power",
                        "status": "in_progress",
                        "alert": "Beta well variance",
                    }
                ),
                "Monitoring geothermal output",
                now - 1500,
            ),
        ]

        cursor.executemany(
            "INSERT INTO cognitive_timeline_states (state_id, timestamp, state_type, state_data, description, workflow_id, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    state_id,
                    timestamp,
                    state_type,
                    state_data,
                    description,
                    STORY_WORKFLOW_ID,
                    created_at,
                )
                for (
                    state_id,
                    timestamp,
                    state_type,
                    state_data,
                    description,
                    created_at,
                ) in timeline_states
            ],
        )

        conn.commit()

        counts = {
            "workflows": 1,
            "goals": len(goals),
            "actions": len(actions),
            "artifacts": len(artifacts),
            "artifact_relationships": len(artifact_relationships),
            "memories": len(memories),
            "memory_links": len(memory_links),
            "cognitive_timeline_states": len(timeline_states),
        }

        return StorySeedResult(
            workflow_id=STORY_WORKFLOW_ID,
            created=True,
            counts=counts,
        )
    finally:
        conn.close()


def get_workflow_story(
    workflow_id: str,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a rich story payload by aggregating related entities for a workflow."""

    conn, _ = _open_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM workflows WHERE workflow_id = ?",
            (workflow_id,),
        )
        workflow_row = cursor.fetchone()
        if workflow_row is None:
            raise StoryNotFoundError(f"Workflow {workflow_id} not found")

        workflow = dict(workflow_row)
        workflow["metadata"] = _parse_json(workflow.get("metadata"))

        cursor.execute(
            "SELECT * FROM goals WHERE workflow_id = ? ORDER BY priority",
            (workflow_id,),
        )
        goals = [dict(row) for row in cursor.fetchall()]

        goal_map: Dict[str, Dict[str, Any]] = {}
        for goal in goals:
            goal_copy = goal.copy()
            goal_copy["children"] = []
            goal_map[goal_copy["goal_id"]] = goal_copy

        root_goals: List[Dict[str, Any]] = []
        for goal in goal_map.values():
            parent_id = goal.get("parent_goal_id")
            if parent_id and parent_id in goal_map:
                goal_map[parent_id]["children"].append(goal)
            else:
                root_goals.append(goal)

        cursor.execute(
            "SELECT * FROM actions WHERE workflow_id = ? ORDER BY sequence_number",
            (workflow_id,),
        )
        actions = []
        for row in cursor.fetchall():
            action_dict = dict(row)
            for key in ("tool_args", "tool_result", "tool_data", "tags", "metadata"):
                action_dict[key] = _parse_json(action_dict.get(key))
            actions.append(action_dict)

        cursor.execute(
            "SELECT * FROM artifacts WHERE workflow_id = ?",
            (workflow_id,),
        )
        artifacts = []
        for row in cursor.fetchall():
            artifact_dict = dict(row)
            artifact_dict["metadata"] = _parse_json(artifact_dict.get("metadata"))
            artifact_dict["tags"] = _parse_json(artifact_dict.get("tags")) or []
            artifact_dict["content"] = _parse_json(artifact_dict.get("content")) or artifact_dict.get("content")
            artifacts.append(artifact_dict)

        cursor.execute(
            "SELECT source_artifact_id, target_artifact_id, relation_type, created_at "
            "FROM artifact_relationships WHERE source_artifact_id IN ("
            "SELECT artifact_id FROM artifacts WHERE workflow_id = ?) OR target_artifact_id IN ("
            "SELECT artifact_id FROM artifacts WHERE workflow_id = ?)",
            (workflow_id, workflow_id),
        )
        artifact_relationships = [dict(row) for row in cursor.fetchall()]

        cursor.execute(
            "SELECT * FROM memories WHERE workflow_id = ?",
            (workflow_id,),
        )
        memories = [dict(row) for row in cursor.fetchall()]

        cursor.execute(
            "SELECT * FROM memory_links WHERE source_memory_id IN ("
            "SELECT memory_id FROM memories WHERE workflow_id = ?) OR target_memory_id IN ("
            "SELECT memory_id FROM memories WHERE workflow_id = ?)",
            (workflow_id, workflow_id),
        )
        memory_links = [dict(row) for row in cursor.fetchall()]

        cursor.execute(
            "SELECT * FROM cognitive_timeline_states WHERE workflow_id = ? ORDER BY timestamp",
            (workflow_id,),
        )
        timeline_states = []
        for row in cursor.fetchall():
            state_dict = dict(row)
            state_dict["state_data"] = _parse_json(state_dict.get("state_data")) or {}
            timeline_states.append(state_dict)

        action_status_counts: Dict[str, int] = {}
        for action in actions:
            action_status_counts[action.get("status", "unknown")] = (
                action_status_counts.get(action.get("status", "unknown"), 0) + 1
            )

        goal_status_counts: Dict[str, int] = {}
        for goal in goals:
            goal_status_counts[goal.get("status", "unknown")] = (
                goal_status_counts.get(goal.get("status", "unknown"), 0) + 1
            )

        latest_state = timeline_states[-1] if timeline_states else None

        return {
            "workflow": workflow,
            "goal_tree": root_goals,
            "goals": goals,
            "actions": actions,
            "artifacts": artifacts,
            "artifact_relationships": artifact_relationships,
            "memories": memories,
            "memory_links": memory_links,
            "timeline": timeline_states,
            "insights": {
                "action_status_counts": action_status_counts,
                "goal_status_counts": goal_status_counts,
                "artifact_relationship_count": len(artifact_relationships),
                "memory_count": len(memories),
                "latest_state": latest_state,
            },
        }
    finally:
        conn.close()
