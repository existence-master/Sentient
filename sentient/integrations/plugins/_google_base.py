"""Shared base for Google integrations (OAuth via the one Desktop client)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sentient.integrations import google
from sentient.integrations.base import IntegrationError, IntegrationPlugin

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager


class GooglePlugin(IntegrationPlugin):
    auth_type = "oauth"
    api_name: str = "Google API"
    api_slug: str = ""

    def __init__(self) -> None:
        super().__init__()
        self.setup_fields = google.google_setup_fields()
        self.instructions_md = google.google_instructions(self.api_name, self.api_slug)
        self.docs_url = google.GOOGLE_DOCS_URL

    async def begin_oauth(self, fields: dict[str, str], mgr: IntegrationManager) -> dict | None:
        return await mgr.start_google_flow(self.id, fields)

    async def validate(self, fields: dict[str, str], mgr: IntegrationManager) -> tuple[dict, str | None]:
        raise IntegrationError("Use Connect to sign in with Google.")

    async def test(self, credentials: dict | None, mgr: IntegrationManager) -> str:
        token = await google.access_token(mgr, self.id, force_refresh=True)
        email = await google.fetch_email(token)
        return f"Signed in to Google{f' as {email}' if email else ''}."
