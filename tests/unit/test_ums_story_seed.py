"""Unit tests for unified memory story seeding utilities."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "ultimate_mcp_server/core/ums_api/story_data.py"


def _load_story_module():
    package_root = Path(__file__).resolve().parents[2]

    # Prepare minimal package structure to satisfy relative imports without pulling heavy deps
    package = types.ModuleType("ultimate_mcp_server")
    package.__path__ = [str(package_root / "ultimate_mcp_server")]
    sys.modules.setdefault("ultimate_mcp_server", package)

    core_pkg = types.ModuleType("ultimate_mcp_server.core")
    core_pkg.__path__ = [str(package_root / "ultimate_mcp_server/core")]
    sys.modules.setdefault("ultimate_mcp_server.core", core_pkg)

    ums_api_pkg = types.ModuleType("ultimate_mcp_server.core.ums_api")
    ums_api_pkg.__path__ = [str(package_root / "ultimate_mcp_server/core/ums_api")]
    sys.modules.setdefault("ultimate_mcp_server.core.ums_api", ums_api_pkg)

    spec = importlib.util.spec_from_file_location(
        "ultimate_mcp_server.core.ums_api.story_data",
        MODULE_PATH,
        submodule_search_locations=[str(MODULE_PATH.parent)],
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError("Unable to load story_data module for testing")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "ultimate_mcp_server.core.ums_api"
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


story_data = _load_story_module()
STORY_WORKFLOW_ID = story_data.STORY_WORKFLOW_ID
stage_sample_story = story_data.stage_sample_story
fetch_story_snapshot = story_data.fetch_story_snapshot


class StorySeedTests(unittest.TestCase):
    """Verify that the story seeding helpers populate the database correctly."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "story.db"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_stage_sample_story_creates_complete_story(self) -> None:
        """Staging the sample story should populate every major table."""

        result = stage_sample_story(str(self.db_path))

        self.assertEqual(result["workflow"]["workflow_id"], STORY_WORKFLOW_ID)
        self.assertEqual(
            result["counts"],
            {
                "workflows": 1,
                "goals": 3,
                "actions": 3,
                "artifacts": 3,
                "artifact_relationships": 2,
                "memories": 3,
                "memory_links": 2,
                "cognitive_timeline_states": 3,
            },
        )

        snapshot = fetch_story_snapshot(str(self.db_path))
        self.assertIsNotNone(snapshot)
        assert snapshot is not None  # mypy friendly
        self.assertEqual(snapshot["workflow"]["workflow_id"], STORY_WORKFLOW_ID)
        self.assertEqual(len(snapshot["goals"]), 3)
        self.assertEqual(len(snapshot["actions"]), 3)
        self.assertEqual(len(snapshot["artifacts"]), 3)
        self.assertEqual(len(snapshot["artifact_relationships"]), 2)
        self.assertEqual(len(snapshot["memories"]), 3)
        self.assertEqual(len(snapshot["memory_links"]), 2)
        self.assertEqual(len(snapshot["cognitive_timeline_states"]), 3)

        # Ensure JSON metadata is parsed into Python objects
        self.assertIsInstance(snapshot["actions"][0]["tags"], list)
        self.assertIsInstance(snapshot["artifacts"][0]["metadata"], dict)
        self.assertIsInstance(snapshot["cognitive_timeline_states"][0]["state_data"], dict)

        artifact_ids = {artifact["artifact_id"] for artifact in snapshot["artifacts"]}
        for relation in snapshot["artifact_relationships"]:
            self.assertIn(relation["source_artifact_id"], artifact_ids)
            self.assertIn(relation["target_artifact_id"], artifact_ids)

        memory_ids = {memory["memory_id"] for memory in snapshot["memories"]}
        for link in snapshot["memory_links"]:
            self.assertIn(link["source_memory_id"], memory_ids)
            self.assertIn(link["target_memory_id"], memory_ids)

    def test_stage_sample_story_is_idempotent(self) -> None:
        """Running the staging helper multiple times should not duplicate data."""

        first = stage_sample_story(str(self.db_path))
        second = stage_sample_story(str(self.db_path))

        self.assertEqual(second["counts"], first["counts"])

        snapshot = fetch_story_snapshot(str(self.db_path))
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(len(snapshot["artifacts"]), 3)
        self.assertEqual(len(snapshot["artifact_relationships"]), 2)
        self.assertEqual(len(snapshot["memory_links"]), 2)


if __name__ == "__main__":
    unittest.main()
