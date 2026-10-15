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
    "Subscribe to newsletter", "Follow", "Notify me", "Get updates", "Join the waitlist", "Subscribe to alerts",
])
def test_free_signup_reads_signs_up(name):
    assert safety.risk_label_for(name, "click") == "Signs up"
    assert safety.risk_label_for(name, "click", near="Our weekly newsletter. Free, no spam. " + name) == "Signs up"


def test_a_free_word_in_a_cut_short_context_is_not_enough():
    # the context stopped at another offer and holds only the button: a price may be just out of reach
    assert safety.risk_label_for("Subscribe to the newsletter", "click", "Subscribe to the newsletter", True) == "Purchase"
    # a whole free card was read before the stop: fine
    near = "The weekly newsletter. Short tips every Sunday. Free, no spam. Subscribe"
    assert safety.risk_label_for("Subscribe", "click", near, True) == "Signs up"


def test_a_bare_subscribe_needs_a_free_signal_nearby():
    assert safety.risk_label_for("Subscribe", "click") == PURCHASE_LABEL
    assert safety.risk_label_for("Subscribe", "click", near="Team plan Subscribe") == PURCHASE_LABEL
    near = "The weekly newsletter. Free, no spam. Subscribe"
    assert safety.risk_label_for("Subscribe", "click", near=near) == "Signs up"


def test_a_bare_sign_up_keeps_its_plain_label_unless_money_shows():
    # "Sign up" was never a risky click; with no free and no money signal it stays an ordinary click, as before
    assert safety.risk_label_for("Sign up", "click") == "Clicks"
    assert safety.click_risk({"role": "button", "name": "Sign up"}) == Risk.write
    assert safety.risk_label_for("Sign up", "click", near="Free forever. Sign up") == "Signs up"
    assert safety.risk_label_for("Sign up", "click", near="Pro plan, $9/month. Sign up") == PURCHASE_LABEL


@pytest.mark.parametrize("name, near", [
    ("Sign up", "Pro plan, $9/month. Sign up"),
    ("Follow", "Upgrade to follow more creators"),
    ("Notify me", "Price drops to 20 EUR per month"),
    ("Subscribe to the newsletter", ""),  # with near_cut below: the price may be out of reach
])
def test_a_purchase_label_on_a_sign_up_is_always_send(name, near):
    """CodeRabbit #382: a 'Purchase' label must come with send risk, or is_purchase would skip its hard stop."""
    el = {"role": "button", "name": name, "near": near or name, "near_cut": not near}
    assert safety.risk_label_for(name, "click", el["near"], el["near_cut"]) == PURCHASE_LABEL
    assert safety.click_risk(el) == Risk.send
    field = {"name": "Email", "form_submit_text": name, "near": el["near"], "near_cut": el["near_cut"]}
    assert safety.type_risk(field, submit=True) == Risk.send


def test_free_sign_ups_keep_their_risk():
    assert safety.click_risk({"role": "button", "name": "Follow"}) == Risk.write
    assert safety.click_risk({"role": "button", "name": "Notify me"}) == Risk.write
    assert safety.click_risk({"role": "button", "name": "Subscribe", "near": "Free newsletter. Subscribe"}) == Risk.send


@pytest.mark.parametrize("name, near", [
    (PAID_BUTTON, ""),
    ("Subscribe to newsletter", "Free for 14 days, then 199 INR"),
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
    assert len(refs) == 4
    labels = [browser.describe_for("click", {"ref": r})["risk_label"] for r in refs]
    # the third card's price sits two levels up; the fourth card has more text than one small card holds
    assert labels == ["Signs up", PURCHASE_LABEL, PURCHASE_LABEL, PURCHASE_LABEL]
    assert "₹499/month" in browser._snap["refs"][refs[1]]["near"]
    assert "₹499" not in browser._snap["refs"][refs[0]]["near"]


async def test_live_priced_card_whose_context_stops_early_is_a_purchase(browser, site):
    """The context stops at the "Choose" button before the price. With no free word, or with a free word but no whole
    card read, it must stay a purchase so an allow rule can never let it through."""
    ctx = make_ctx(browser.app)
    await bt.browser_open.call(ctx, {"url": f"{site}/signup-choose.html"})
    refs = browser._snap["refs"]
    for name in ("Subscribe", "Subscribe to the newsletter"):
        ref = next(r for r, e in refs.items() if e.get("name") == name)
        assert refs[ref]["near_cut"] is True
        assert "2,999" not in refs[ref]["near"] and "1,499" not in refs[ref]["near"]  # the price really is out of reach
        assert browser.describe_for("click", {"ref": ref})["risk_label"] == PURCHASE_LABEL, name
