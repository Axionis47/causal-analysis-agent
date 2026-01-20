"""Celery worker configuration."""

import sentry_sdk
from celery import Celery
from sentry_sdk.integrations.celery import CeleryIntegration

from app import __version__
from app.core.config import settings
from app.core.logging import configure_logging, get_release, sentry_before_send

configure_logging()

if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.SENTRY_ENVIRONMENT or settings.ENVIRONMENT,
        traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        integrations=[CeleryIntegration()],
        before_send=sentry_before_send,
        release=get_release(),
    )
    sentry_sdk.set_tag("app_name", settings.APP_NAME)
    sentry_sdk.set_tag("environment", settings.ENVIRONMENT)
    sentry_sdk.set_tag("version", __version__)

celery_app = Celery(
    "causal_analysis",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes max
    task_soft_time_limit=25 * 60,  # 25 minutes soft limit
    worker_prefetch_multiplier=1,
    worker_concurrency=4,
    worker_hijack_root_logger=False,
)

# Auto-discover tasks from all app modules
celery_app.autodiscover_tasks(["app.tasks"])


@celery_app.task(bind=True)
def debug_task(self) -> str:
    """Debug task for testing Celery setup."""
    return f"Request: {self.request!r}"
