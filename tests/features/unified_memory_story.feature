Feature: Unified memory story demonstrates full database coverage
  The unified agent memory database should capture the complete narrative of a
  critical support triage engagement so that downstream tools can reason about
  goals, actions, artifacts, and cognitive state without losing context.

  Background:
    Given a fresh unified memory database

  Scenario: Support triage workflow persists the full cognitive trace
    When the support triage story is staged
    Then the workflow captures the outage response narrative
    And the goals track investigative and communication work
    And the actions record how the agent executed tools
    And the artifacts document the produced evidence
    And the artifact relationships show how deliverables connect
    And the memories surface triage insights and follow-ups
    And the memory links map the supporting knowledge graph
    And the cognitive timeline explains the investigation flow
