import pytest

pytest_bdd = pytest.importorskip("pytest_bdd")
from pytest_bdd import given, scenarios, then, when  # type: ignore

from ultimate_mcp_server.services.unified_memory_story import UnifiedMemoryStoryBuilder


scenarios("unified_memory_story.feature")


@pytest.fixture()
def story_builder(tmp_path):
    return UnifiedMemoryStoryBuilder(str(tmp_path / "bdd_unified_memory.db"))


@given("a fresh unified memory database")
def given_fresh_db(story_builder):
    return story_builder


@when("the support triage story is staged")
def when_story_staged(given_fresh_db):
    staged = given_fresh_db.stage_support_triage_story()
    snapshot = given_fresh_db.get_story_snapshot(staged.workflow_id)
    return staged, snapshot


@then("the workflow captures the outage response narrative")
def then_workflow_captures_story(when_story_staged):
    _, snapshot = when_story_staged
    assert snapshot.workflow["title"] == "Customer outage triage"
    assert snapshot.workflow["goal"] == "Restore service for warehouse robots"
    assert snapshot.workflow["status"] == "active"


@then("the goals track investigative and communication work")
def then_goals_track_work(when_story_staged):
    staged, snapshot = when_story_staged
    assert len(snapshot.goals) == 3
    goal_ids = {goal["goal_id"] for goal in snapshot.goals}
    assert goal_ids == set(staged.goal_ids)
    statuses = {goal["status"] for goal in snapshot.goals}
    assert statuses == {"in_progress", "completed", "blocked"}


@then("the actions record how the agent executed tools")
def then_actions_record_execution(when_story_staged):
    staged, snapshot = when_story_staged
    assert {action["action_id"] for action in snapshot.actions} == set(staged.action_ids)
    assert snapshot.actions[0]["tool_name"] == "log_fetcher"
    assert any(action["status"] == "in_progress" for action in snapshot.actions)


@then("the artifacts document the produced evidence")
def then_artifacts_document_evidence(when_story_staged):
    staged, snapshot = when_story_staged
    assert {artifact["artifact_id"] for artifact in snapshot.artifacts} == set(
        staged.artifact_ids
    )
    types = {artifact["artifact_type"] for artifact in snapshot.artifacts}
    assert types == {"dataset", "report", "plan"}


@then("the artifact relationships show how deliverables connect")
def then_artifact_relationships_connect(when_story_staged):
    staged, snapshot = when_story_staged
    assert len(snapshot.artifact_relationships) == len(staged.relationship_ids)
    assert {rel["relation_type"] for rel in snapshot.artifact_relationships} == {
        "informs",
        "supports",
    }


@then("the memories surface triage insights and follow-ups")
def then_memories_surface_insights(when_story_staged):
    staged, snapshot = when_story_staged
    assert {memory["memory_id"] for memory in snapshot.memories} == set(staged.memory_ids)
    assert any(memory["memory_type"] == "summary" for memory in snapshot.memories)
    assert any("Next:" in memory["content"] for memory in snapshot.memories)


@then("the memory links map the supporting knowledge graph")
def then_memory_links_map_graph(when_story_staged):
    staged, snapshot = when_story_staged
    assert len(snapshot.memory_links) == len(staged.memory_link_ids)
    assert {link["link_type"] for link in snapshot.memory_links} == {"supports", "yields"}


@then("the cognitive timeline explains the investigation flow")
def then_cognitive_timeline_explains_flow(when_story_staged):
    _, snapshot = when_story_staged
    assert len(snapshot.cognitive_timeline_states) == 3
    assert [state["state_type"] for state in snapshot.cognitive_timeline_states] == [
        "observation",
        "analysis",
        "action",
    ]
