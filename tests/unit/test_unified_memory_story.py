"""Tests for the unified memory story seeding utilities."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


if "fastapi" not in sys.modules:  # pragma: no cover - testing shim
    fastapi_stub = types.ModuleType("fastapi")

    class FastAPI:  # noqa: D401 - lightweight stub
        def __init__(self, *args, **kwargs):
            pass

    class HTTPException(Exception):
        def __init__(self, status_code: int = 500, detail: str | None = None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class WebSocket:  # noqa: D401 - stub
        async def accept(self):  # pragma: no cover - unused stub
            pass

    class WebSocketDisconnect(Exception):
        pass

    fastapi_stub.FastAPI = FastAPI
    fastapi_stub.HTTPException = HTTPException
    fastapi_stub.WebSocket = WebSocket
    fastapi_stub.WebSocketDisconnect = WebSocketDisconnect
    sys.modules["fastapi"] = fastapi_stub

if "pydantic" not in sys.modules:  # pragma: no cover - testing shim
    pydantic_stub = types.ModuleType("pydantic")

    class BaseModel:  # noqa: D401 - simplified replacement
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

    def Field(default=None, **_kwargs):  # noqa: D401
        return default

    class ValidationError(Exception):
        pass

    def field_validator(*_args, **_kwargs):  # noqa: D401
        def decorator(func):
            return func

        return decorator

    pydantic_stub.BaseModel = BaseModel
    pydantic_stub.Field = Field
    pydantic_stub.ValidationError = ValidationError
    pydantic_stub.field_validator = field_validator
    sys.modules["pydantic"] = pydantic_stub


from ultimate_mcp_server.services.unified_memory_story import (
    UnifiedMemoryStoryBuilder,
)


class UnifiedMemoryStoryTestCase(unittest.TestCase):
    """Unit tests covering the realistic unified memory story helper."""

    def setUp(self) -> None:  # noqa: D401 - standard unittest hook
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "unified_memory.db"
        self.builder = UnifiedMemoryStoryBuilder(str(self.db_path))

    def tearDown(self) -> None:  # noqa: D401 - standard unittest hook
        self._tmp.cleanup()

    def test_stage_support_triage_story_populates_every_table(self) -> None:
        staged = self.builder.stage_support_triage_story()
        snapshot = self.builder.get_story_snapshot(staged.workflow_id)

        self.assertEqual(snapshot.workflow["title"], "Customer outage triage")
        self.assertSetEqual(
            {goal["goal_id"] for goal in snapshot.goals},
            set(staged.goal_ids),
        )
        self.assertSetEqual(
            {action["action_id"] for action in snapshot.actions},
            set(staged.action_ids),
        )
        self.assertSetEqual(
            {artifact["artifact_id"] for artifact in snapshot.artifacts},
            set(staged.artifact_ids),
        )
        self.assertSetEqual(
            {memory["memory_id"] for memory in snapshot.memories},
            set(staged.memory_ids),
        )
        self.assertEqual(
            len(snapshot.artifact_relationships),
            len(staged.relationship_ids),
        )
        self.assertEqual(
            len(snapshot.memory_links),
            len(staged.memory_link_ids),
        )
        self.assertListEqual(
            [state["state_id"] for state in snapshot.cognitive_timeline_states],
            staged.timeline_state_ids,
        )

    def test_snapshot_preserves_story_context(self) -> None:
        staged = self.builder.stage_support_triage_story()
        snapshot = self.builder.get_story_snapshot(staged.workflow_id)

        metadata = json.loads(snapshot.workflow["metadata"])
        self.assertEqual(metadata["customer"], "Proxima Robotics")
        self.assertEqual(metadata["impact"], "High")

        # Goals capture hierarchy and progression
        root_goal = next(goal for goal in snapshot.goals if goal["parent_goal_id"] is None)
        child_goals = [
            goal for goal in snapshot.goals if goal["parent_goal_id"] == root_goal["goal_id"]
        ]
        self.assertSetEqual({goal["status"] for goal in child_goals}, {"completed", "blocked"})

        # Actions follow the expected sequence numbers
        sequence_numbers = [action["sequence_number"] for action in snapshot.actions]
        self.assertEqual(sequence_numbers, sorted(sequence_numbers))
        self.assertEqual(snapshot.actions[0]["tool_name"], "log_fetcher")
        self.assertIn(
            "in_progress",
            {action["status"] for action in snapshot.actions},
        )

        # Artifacts and relationships illustrate the investigation flow
        relation_types = {rel["relation_type"] for rel in snapshot.artifact_relationships}
        self.assertSetEqual(relation_types, {"informs", "supports"})

        # Memories cross-link the investigation summary to follow-up work
        link_types = {link["link_type"] for link in snapshot.memory_links}
        self.assertSetEqual(link_types, {"supports", "yields"})

        # Timeline states narrate the triage story chronologically
        timestamps = [state["timestamp"] for state in snapshot.cognitive_timeline_states]
        self.assertEqual(timestamps, sorted(timestamps))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
