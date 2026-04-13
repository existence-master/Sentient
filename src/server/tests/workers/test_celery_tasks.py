import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from workers.tasks import async_refine_and_plan_ai_task, cud_memory_task
from mcp_hub.memory.utils import initialize_embedding_model, initialize_agents

# --- Test refine_and_plan_ai_task ---

@pytest.mark.asyncio
async def test_refine_and_plan_ai_task_success(mocker):
    # Mock DB manager
    mock_db = AsyncMock()
    mock_task = {
        "_id": "some_id",
        "task_id": "task-123",
        "user_id": "test-user-123",
        "description": "raw user prompt",
        "assignee": "ai"
    }
    mock_db.get_task.return_value = mock_task
    mock_db.user_profiles_collection.find_one.return_value = {
        "userData": {"personalInfo": {"name": "Tester", "timezone": "UTC"}}
    }
    # The conftest mock_mongo_manager fixture now handles patching MongoManager

    # Mock LLM call
    refined_details = {
        "name": "Refined Task Name",
        "description": "Refined description.",
        "priority": 1,
        "schedule": {"type": "once", "run_at": "2024-01-01T12:00:00"}
    }
    async def mock_run_agent(*args, **kwargs):
        yield [{"role": "assistant", "content": f"```json\n{json.dumps(refined_details)}\n```"}]

    mocker.patch('workers.tasks.run_main_agent', new=mock_run_agent)

    # Mock Celery task call
    mock_planner_delay = mocker.patch('workers.tasks.generate_plan_from_context.delay')

    # Run the async function
    await async_refine_and_plan_ai_task("task-123", "test-user-123")

    # Assertions
    mock_db.get_task.assert_called_once_with("task-123")
    mock_db.update_task.assert_called_once()
    update_args = mock_db.update_task.call_args[0]
    assert update_args[0] == "task-123"
    assert update_args[2]["name"] == "Refined Task Name"
    assert update_args[2]["schedule"]["timezone"] == "UTC" # Check if timezone was injected

    mock_planner_delay.assert_called_once_with("task-123", "test-user-123")
    mock_db.close.assert_called_once()


# --- Test cud_memory_task ---

@pytest.mark.asyncio
async def test_cud_memory_add_action(mocker):
    # Mock DB and its methods
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []  # No similar facts exist
    mock_conn.fetchval.return_value = 1 # Return a new fact_id
    mock_pool = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mocker.patch('mcp_hub.memory.db.get_db_pool', return_value=mock_pool)
    mocker.patch('mcp_hub.memory.db.register_vector', new_callable=AsyncMock)

    # Mock embedding and LLM calls
    mocker.patch('main.gemini_embed.get_normalized_embedding', return_value=[0.1] * 768)
    
    cud_decision_response = {
        "action": "ADD",
        "fact_id": None,
        "content": "The user's favorite color is blue.",
        "analysis": {
            "topics": ["Personal Identity"],
            "memory_type": "long-term",
            "duration": None
        }
    }
    mock_run_agent = mocker.patch('mcp_hub.memory.llm.run_agent_with_prompt', return_value=json.dumps(cud_decision_response))
    
    # Initialize models (mocks will be used)
    initialize_embedding_model()
    initialize_agents()

    # Run the function
    await cud_memory_task.apply_async(args=["test-user-123", "my favorite color is blue", "test_source"])

    # Assertions
    # We can't easily assert the result of the async task, but we can check mocks
    mock_run_agent.assert_called_once()
    # Check that the insert was called
    assert mock_conn.fetchval.call_count > 0
    assert "INSERT INTO facts" in str(mock_conn.fetchval.call_args_list)
