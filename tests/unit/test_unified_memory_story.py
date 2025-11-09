"""Unit tests for unified memory story seeding utilities."""

from __future__ import annotations

import sqlite3

import pytest

from ultimate_mcp_server.core.ums_api.story_seed import (
    STORY_WORKFLOW_ID,
    StoryAlreadyExistsError,
    get_workflow_story,
    seed_unified_memory_story,
)


@pytest.fixture()
def temp_memory_db(tmp_path):
    db_path = tmp_path / "unified_agent_memory.db"
    return str(db_path)


def _count_rows(db_path: str, table: str) -> int:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        return int(cursor.fetchone()[0])


def test_seed_story_populates_all_tables(temp_memory_db: str) -> None:
    result = seed_unified_memory_story(temp_memory_db)

    assert result.workflow_id == STORY_WORKFLOW_ID
    assert result.created is True

    expected_tables = {
        "workflows": 1,
        "goals": 4,
        "actions": 4,
        "artifacts": 4,
        "artifact_relationships": 3,
        "memories": 3,
        "memory_links": 3,
        "cognitive_timeline_states": 3,
    }

    assert result.counts == expected_tables

    for table, expected in expected_tables.items():
        assert _count_rows(temp_memory_db, table) == expected


def test_seed_story_is_idempotent_without_force(temp_memory_db: str) -> None:
    seed_unified_memory_story(temp_memory_db)

    with pytest.raises(StoryAlreadyExistsError):
        seed_unified_memory_story(temp_memory_db)


def test_get_workflow_story_returns_rich_structure(temp_memory_db: str) -> None:
    seed_unified_memory_story(temp_memory_db)

    story = get_workflow_story(STORY_WORKFLOW_ID, db_path=temp_memory_db)

    assert story["workflow"]["title"].startswith("Mission Cerberus")
    assert story["insights"]["action_status_counts"]["completed"] == 2
    assert story["insights"]["action_status_counts"]["in_progress"] == 1
    assert story["insights"]["action_status_counts"]["pending"] == 1

    goal_tree = story["goal_tree"]
    assert any(goal["goal_id"] == "goal_outpost_ready" for goal in goal_tree)

    timeline = story["timeline"]
    assert timeline[0]["state_type"] == "analysis"
    assert timeline[-1]["state_data"]["alert"] == "Beta well variance"

    artifacts = story["artifacts"]
    power_report = next(a for a in artifacts if a["artifact_id"] == "artifact_power_report")
    assert power_report["metadata"]["anomaly"] == "Beta variance"

    mission_metadata = story["workflow"]["metadata"]
    assert "Cerberus Fossae" in mission_metadata["location"]

