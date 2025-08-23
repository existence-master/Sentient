import logging
import datetime
from workers.celery_app import celery_app
from . import state_manager

logger = logging.getLogger(__name__)

async def set_waiting_state(task_id: str, user_id: str, waiting_for: str, timeout_minutes: int, max_retries: int):
    """
    Sets the task state to WAITING and schedules a timeout handler.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    timeout_at = now + datetime.timedelta(minutes=timeout_minutes)

    waiting_config = {
        "waiting_for": waiting_for,
        "timeout_at": timeout_at,
        "started_at": now,
        "max_retries": max_retries,
        "current_retries": 0
    }

    await state_manager.update_orchestrator_state(task_id, user_id, {
        "current_state": "WAITING",
        "waiting_config": waiting_config
    })

    # Schedule the Celery task to handle the timeout
    celery_app.send_task('handle_waiting_timeout', args=[task_id, waiting_for], eta=timeout_at)
    logger.info(f"Task {task_id} set to WAITING state. Timeout handler scheduled for {timeout_at}.")
