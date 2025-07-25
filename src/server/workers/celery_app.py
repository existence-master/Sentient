# workers/celery_app.py (or wherever this file is)

import os
from celery import Celery
from celery.schedules import crontab
from dotenv import load_dotenv
import logging

# --- Environment Loading Logic ---
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
logging.info(f"[CeleryApp] Initializing configuration for ENVIRONMENT='{ENVIRONMENT}'")

server_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if ENVIRONMENT == 'dev-local':
    dotenv_path = os.path.join(server_root, '.env')
    logging.info(f"[CeleryApp] Loading .env file for 'dev-local' mode from: {dotenv_path}")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
elif ENVIRONMENT == 'selfhost':
    dotenv_path = os.path.join(server_root, '.env.selfhost')
    logging.info(f"[CeleryApp] Loading .env file for 'selfhost' mode from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    logging.info(f"[CeleryApp] Skipping dotenv loading for '{ENVIRONMENT}' mode.")

CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND')

if not CELERY_BROKER_URL or not CELERY_RESULT_BACKEND:
    raise ValueError("CELERY_BROKER_URL and CELERY_RESULT_BACKEND must be set in the environment or .env file")

celery_app = Celery(
    'tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        'workers.executor.tasks',
        'workers.tasks'
    ]
)

celery_app.conf.update(
    task_track_started=True,
    beat_schedule = {
        'run-due-tasks-every-minute': {
            'task': 'run_due_tasks',
            'schedule': 300.0,
        },
        'schedule-polling-tasks-every-minute': {
            'task': 'schedule_all_polling',
            'schedule': 3600.0,
        },
    }
)

if __name__ == '__main__':
    celery_app.start()