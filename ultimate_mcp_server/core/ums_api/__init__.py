"""Ultimate MCP Server UMS API Module.

This module provides the UMS (Unified Memory System) API endpoints and services
for monitoring and managing cognitive states, actions, performance, and artifacts.
"""

from .story_seed import (
    STORY_WORKFLOW_ID,
    StoryAlreadyExistsError,
    StoryNotFoundError,
    get_workflow_story,
    seed_unified_memory_story,
)

try:
    from .ums_endpoints import setup_ums_api
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs
    def setup_ums_api(*_, **__):  # type: ignore[no-redef]
        raise RuntimeError("FastAPI is required to set up UMS API endpoints")

# Import database utilities
from .ums_database import (
    get_database_path,
    get_db_connection,
    execute_query,
    execute_update,
    ensure_database_exists,
    _dict_depth,
    _count_values,
    calculate_state_complexity,
    compute_state_diff,
    generate_timeline_segments,
    calculate_timeline_stats,
    get_action_status_indicator,
    categorize_action_performance,
    get_action_resource_usage,
    estimate_wait_time,
    get_priority_label,
    calculate_action_performance_score,
    calculate_efficiency_rating,
    format_file_size,
    calculate_performance_summary,
    generate_performance_insights,
    find_cognitive_patterns,
    calculate_sequence_similarity,
    calculate_single_state_similarity,
    analyze_state_transitions,
    detect_cognitive_anomalies,
)

# Import all models for easy access when dependencies are installed
try:  # pragma: no cover - exercised in minimal envs
    from .ums_models import *  # type: ignore[misc]
except ModuleNotFoundError:
    pass

# Import all services
try:  # pragma: no cover - exercised in minimal envs
    from .ums_services import *  # type: ignore[misc]
except ModuleNotFoundError:
    pass

__all__ = [
    "setup_ums_api",
    "STORY_WORKFLOW_ID",
    "StoryAlreadyExistsError",
    "StoryNotFoundError",
    "get_workflow_story",
    "seed_unified_memory_story",
    # Database utilities
    "get_database_path",
    "get_db_connection", 
    "execute_query",
    "execute_update",
    "ensure_database_exists",
    "_dict_depth",
    "_count_values",
    "calculate_state_complexity",
    "compute_state_diff",
    "generate_timeline_segments",
    "calculate_timeline_stats",
    "get_action_status_indicator",
    "categorize_action_performance",
    "get_action_resource_usage",
    "estimate_wait_time",
    "get_priority_label",
    "calculate_action_performance_score",
    "calculate_efficiency_rating",
    "format_file_size",
    "calculate_performance_summary",
    "generate_performance_insights",
    "find_cognitive_patterns",
    "calculate_sequence_similarity",
    "calculate_single_state_similarity",
    "analyze_state_transitions",
    "detect_cognitive_anomalies",
] 