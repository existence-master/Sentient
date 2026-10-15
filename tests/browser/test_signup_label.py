"""A free sign-up reads "Signs up"; anything that may cost money stays "Purchase" and always asks (#378)."""

from __future__ import annotations

import pytest

from sentient.browser import safety
from sentient.browser import tools as bt
from sentient.browser.service import BrowserService
from sentient.tools.base import Risk
from sentient.tools.rules import PURCHASE_LABEL, is_purchase
from tests.browser.conftest import make_app, make_ctx

PAID_BUTTON = f"Subscribe {chr(0x2013)} {chr(0x20B9)}499/month"  # with an en dash, as sites write it


@pytest.mark.parametrize("name", [
    "Subscribe", "Subscribe to newsletter", "Follow", "Notify me", "Sign up", "Get updates", "Join the waitlist",
])
def test_free_signup_reads_signs_up(name):
    assert safety.risk_label_for(name, "click") == "Signs up"
    assert safety.risk_label_for(name, "click", near="Our weekly newsletter. Free, no spam. " + name) == "Signs up"


@pytest.mark.parametrize("name, near", [
    (PAID_BUTTON, ""),
    ("Subscribe for $5", ""),
    ("Subscribe", "Pro plan ₹499/month Subscribe"),
    ("Subscribe", "Start your 7-day free trial"),
    ("Subscribe", "Billed yearly at 4,999 INR"),
    ("Follow", "Upgrade to follow more creators"),
    ("Notify me", "Price drops to 20 EUR per month"),
    ("Sign up", "Premium membership"),
    ("Subscribe and pay", ""),
    ("Buy now", ""),
])
def test_paid_subscribe_stays_purchase(name, near):
    assert safety.risk_label_for(name, "click", near=near) == PURCHASE_LABEL


def test_other_labels_unchanged():
    assert safety.risk_label_for("Place order", "click") == "Purchase"
    assert safety.risk_label_for("Delete my account", "click") == "Deletes"
    assert safety.risk_label_for("Unsubscribe", "click") == "Clicks"
    assert safety.risk_label_for("Next page", "click") == "Clicks"


def _service_with(elements: list[dict]):
    app = make_app()
    app.config.tools.approvals.mode = "ask"
    svc = BrowserService(app)
    app.browser = svc
    svc._snap = {"page": None, "url": "http://shop.test/", "refs": {e["ref"]: e for e in elements}}
    return svc, make_ctx(app)


async def test_free_subscribe_is_not_a_purchase_but_paid_one_is():
    svc, ctx = _service_with([
        {"ref": "e1", "role": "button", "tag": "button", "name": "Subscribe",
         "near": "Get our weekly newsletter. Free, no spam. Subscribe"},
        {"ref": "e2", "role": "button", "tag": "button", "name": PAID_BUTTON,
         "near": "Pro plan " + PAID_BUTTON},
        {"ref": "e3", "role": "button", "tag": "button", "name": "Subscribe", "near": "Pro plan ₹499/month Subscribe"},
    ])
    click = bt.browser_click
    # all are still outward clicks, so they ask under the normal rules
    assert svc.risk_for("click", {"ref": "e1"}) == Risk.send
    assert svc.risk_for("click", {"ref": "e2"}) == Risk.send
    assert svc.describe_for("click", {"ref": "e1"}) == {"risk_label": "Signs up", "target": "Subscribe"}
    assert svc.describe_for("click", {"ref": "e2"})["risk_label"] == PURCHASE_LABEL
    assert svc.describe_for("click", {"ref": "e3"})["risk_label"] == PURCHASE_LABEL
    assert not await is_purchase(click, {"ref": "e1"}, ctx, Risk.send)
    assert await is_purchase(click, {"ref": "e2"}, ctx, Risk.send)
    assert await is_purchase(click, {"ref": "e3"}, ctx, Risk.send)


async def test_live_snapshot_sees_the_price_beside_a_button(browser, site):
    """Identical "Subscribe" buttons: those in priced cards are purchases, the free one is a sign-up."""
    ctx = make_ctx(browser.app)
    await bt.browser_open.call(ctx, {"url": f"{site}/signup.html"})
    refs = [r for r, e in browser._snap["refs"].items() if e.get("name") == "Subscribe"]
    assert len(refs) == 3
    labels = [browser.describe_for("click", {"ref": r})["risk_label"] for r in refs]
    assert labels == ["Signs up", PURCHASE_LABEL, PURCHASE_LABEL]  # the third card's price sits two levels up
    assert "₹499/month" in browser._snap["refs"][refs[1]]["near"]
    assert "₹499" not in browser._snap["refs"][refs[0]]["near"]
