"""Google Slides: build a presentation from an outline; read one back."""

from __future__ import annotations

import secrets as pysecrets

from pydantic import BaseModel, Field

from sentient.integrations.base import IntegrationError, itool
from sentient.integrations.google import gapi
from sentient.integrations.plugins._google_base import GooglePlugin
from sentient.tools.base import Risk, ToolContext

API = "https://slides.googleapis.com/v1/presentations"
PID = "gslides"


class SlideSpec(BaseModel):
    title: str = Field(description="Slide heading")
    bullets: list[str] = Field(default_factory=list, description="Bullet points on the slide")


def _oid(prefix: str) -> str:
    return f"{prefix}_{pysecrets.token_hex(8)}"


def outline_requests(first_slide: dict | None, title: str, subtitle: str | None, slides: list[dict]) -> list[dict]:
    reqs: list[dict] = []
    if first_slide is not None:
        for el in first_slide.get("pageElements") or []:
            ph = ((el.get("shape") or {}).get("placeholder") or {}).get("type")
            if ph in {"CENTERED_TITLE", "TITLE"} and title:
                reqs.append({"insertText": {"objectId": el["objectId"], "text": title}})
            elif ph == "SUBTITLE" and subtitle:
                reqs.append({"insertText": {"objectId": el["objectId"], "text": subtitle}})
    for i, s in enumerate(slides):
        slide_id, title_id, body_id = _oid("slide"), _oid("title"), _oid("body")
        reqs.append({"createSlide": {
            "objectId": slide_id, "insertionIndex": i + 1,
            "slideLayoutReference": {"predefinedLayout": "TITLE_AND_BODY"},
            "placeholderIdMappings": [
                {"layoutPlaceholder": {"type": "TITLE", "index": 0}, "objectId": title_id},
                {"layoutPlaceholder": {"type": "BODY", "index": 0}, "objectId": body_id},
            ]}})
        reqs.append({"insertText": {"objectId": title_id, "text": s.get("title") or " "}})
        bullets = [b for b in s.get("bullets") or [] if str(b).strip()]
        if bullets:
            reqs.append({"insertText": {"objectId": body_id, "text": "\n".join(bullets)}})
            reqs.append({"createParagraphBullets": {"objectId": body_id, "textRange": {"type": "ALL"},
                                                    "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE"}})
    return reqs


def _url(pid: str) -> str:
    return f"https://docs.google.com/presentation/d/{pid}/edit"


@itool(PID, "gslides_create_presentation", risk=Risk.write)
async def gslides_create_presentation(ctx: ToolContext, title: str, slides: list[SlideSpec],
                                      subtitle: str | None = None) -> dict:
    """Create a Google Slides deck from an outline: a title slide, then one slide per item with a heading and
    bullet points. Example slides: [{"title": "Why now", "bullets": ["Market is growing", "Costs fell 40%"]}]."""
    if not slides:
        raise IntegrationError("Give at least one slide in the outline.")
    pres = await gapi(ctx, PID, "POST", API, json={"title": title})
    pres_id = pres["presentationId"]
    first = (pres.get("slides") or [None])[0]
    reqs = outline_requests(first, title, subtitle, [s if isinstance(s, dict) else s.model_dump() for s in slides])
    await gapi(ctx, PID, "POST", f"{API}/{pres_id}:batchUpdate", json={"requests": reqs})
    return {"presentation_id": pres_id, "title": title, "slides": len(slides) + 1, "url": _url(pres_id)}


@itool(PID, "gslides_read_presentation")
async def gslides_read_presentation(ctx: ToolContext, presentation_id: str) -> dict:
    """Read the text on every slide of a Google Slides presentation."""
    pres = await gapi(ctx, PID, "GET", f"{API}/{presentation_id}")
    out = []
    for i, slide in enumerate(pres.get("slides") or []):
        texts = []
        for el in slide.get("pageElements") or []:
            for te in ((el.get("shape") or {}).get("text") or {}).get("textElements") or []:
                run = (te.get("textRun") or {}).get("content")
                if run:
                    texts.append(run)
        out.append({"slide": i + 1, "text": "".join(texts).strip()})
    return {"presentation_id": presentation_id, "title": pres.get("title"), "slides": out, "url": _url(presentation_id)}


class GSlidesPlugin(GooglePlugin):
    id = PID
    display_name = "Google Slides"
    description = (
        "Turn an outline into a real Google Slides deck with a title slide and bullet-point slides, and read "
        "existing presentations."
    )
    category = "productivity"
    icon = "google-slides"
    api_name = "Google Slides API"
    api_slug = "slides.googleapis.com"
    selection_hint = "Use to create a slide deck / presentation in Google Slides or read one."
    tools = [gslides_create_presentation, gslides_read_presentation]


PLUGIN = GSlidesPlugin()
