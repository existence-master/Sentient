"""Pure-logic tests: risk heuristics, sensitive fields, domains, snapshot formatting. No browser needed."""

from __future__ import annotations

from sentient.browser import safety
from sentient.browser.service import _clean_ref, normalize_key
from sentient.browser.snapshot import focus_on_question, format_element, format_snapshot
from sentient.tools.base import Risk


# ----------------------------------------------------------------------------- click risk
def test_click_risk_send_for_outward_actions():
    for name in ["Place order", "Buy now", "Pay ₹499", "Proceed to checkout", "Send", "Post", "Publish",
                 "Delete account", "Remove from cart", "Confirm", "Subscribe", "Transfer money", "Checkout"]:
        assert safety.click_risk({"role": "button", "name": name}) == Risk.send, name


def test_click_risk_write_for_ordinary_clicks():
    for name in ["Search", "Next", "Sign in", "Posts", "Sender details", "Show more", "Add filter", ""]:
        assert safety.click_risk({"role": "button", "name": name}) == Risk.write, name
    assert safety.click_risk(None) == Risk.write


def test_click_risk_uses_aria_value_and_payment_forms():
    assert safety.click_risk({"name": "", "aria_label": "deleteItem"}) == Risk.send
    assert safety.click_risk({"tag": "input", "type": "submit", "value": "Pay now", "name": ""}) == Risk.send
    assert safety.click_risk({"name": "Continue", "is_submit": True, "form_payment": True}) == Risk.send
    assert safety.click_risk({"name": "Continue", "is_submit": True, "form_payment": False}) == Risk.write


def test_type_and_press_risk():
    search = {"name": "Search", "form_submit_text": "Search"}
    assert safety.type_risk(search, submit=True) == Risk.write
    assert safety.type_risk(search, submit=False) == Risk.write
    assert safety.type_risk({"name": "Write a message", "placeholder": "Message"}, submit=True) == Risk.send
    assert safety.type_risk({"name": "Coupon", "form_payment": True}, submit=True) == Risk.send
    assert safety.type_risk({"name": "Note", "form_submit_text": "Place order"}, submit=True) == Risk.send
    assert safety.press_risk("Enter", {"name": "Comment"}) == Risk.send
    assert safety.press_risk("ArrowDown", {"name": "Comment"}) == Risk.write
    assert safety.press_risk("Enter", None) == Risk.write


# ----------------------------------------------------------------------------- sensitive fields
def test_sensitive_fields_detected():
    cases = [
        ({"type": "password"}, "password"),
        ({"autocomplete": "current-password"}, "password"),
        ({"autocomplete": "section-pay billing cc-number"}, "card"),
        ({"autocomplete": "one-time-code"}, "one-time code"),
        ({"name_attr": "cardNumber"}, "card"),
        ({"id": "cvv"}, "card"),
        ({"label": "Security code"}, "card"),
        ({"placeholder": "Enter OTP"}, "one-time code"),
        ({"label": "Verification code"}, "one-time code"),
        ({"name_attr": "user_pwd"}, "password"),
        ({"label": "UPI PIN"}, "PIN"),
        ({"label": "Aadhaar number"}, "bank or ID number"),
        ({"label": "API key"}, "secret key"),
    ]
    for el, kind in cases:
        assert safety.sensitive_field(el) == kind, el


def test_ordinary_fields_not_sensitive():
    for el in [{"type": "text", "label": "PIN code"}, {"name_attr": "pincode"}, {"type": "email", "label": "Email"},
               {"type": "search", "placeholder": "Search"}, {"autocomplete": "cc-name", "label": "Full name"},
               {"label": "Passenger name"}, {}, None]:
        assert safety.sensitive_field(el) is None, el


def test_card_number_in_text():
    assert safety.looks_like_card_number("4111 1111 1111 1111")
    assert safety.looks_like_card_number("my card is 5555-5555-5555-4444 thanks")
    assert not safety.looks_like_card_number("1234567890123")
    assert not safety.looks_like_card_number("call 9876543210")
    assert "Open browser" in safety.sensitive_refusal("password")


# ----------------------------------------------------------------------------- domains
def test_url_rules():
    assert safety.normalize_url("example.com/a") == "https://example.com/a"
    assert safety.url_problem("https://example.com", [], []) is None
    assert safety.url_problem("file:///C:/secret.txt", [], [])
    assert safety.url_problem("javascript:alert(1)", [], [])
    assert "blocked" in safety.url_problem("https://shop.evil.com/x", [], ["evil.com"])
    assert safety.url_problem("https://notevil.com", [], ["evil.com"]) is None
    assert "allowed" in safety.url_problem("https://other.org", ["amazon.in"], [])
    assert safety.url_problem("https://www.amazon.in/dp/1", ["*.amazon.in"], []) is None
    assert safety.url_problem("https://amazon.in", ["https://amazon.in/"], []) is None
    assert safety.url_problem("about:blank", ["amazon.in"], []) is None


# ----------------------------------------------------------------------------- formatting
def test_format_element_lines():
    assert format_element({"ref": "e12", "role": "button", "name": "Sign in"}) == '[e12] button "Sign in"'
    assert format_element({"ref": "e4", "role": "textbox", "name": "Search", "value": ""}) == '[e4] textbox "Search" value=""'
    line = format_element({"ref": "e5", "role": "textbox", "name": "Password", "type": "password", "value": ""})
    assert "(password)" in line
    assert format_element({"ref": "e6", "role": "checkbox", "name": "Remember", "checked": False}).endswith("unchecked")
    sel = format_element({"ref": "e7", "role": "combobox", "name": "Country", "value": "India",
                          "options": ["India", "Japan"], "more_options": 3})
    assert 'value="India" options: India | Japan (+3 more)' in sel
    assert format_element({"ref": "e8", "role": "link", "name": "Docs", "href": "/docs"}) == '[e8] link "Docs" -> /docs'
    assert format_element({"ref": "e9", "role": "button", "name": 'Say "hi"', "disabled": True}) == "[e9] button \"Say 'hi'\" disabled"


def test_format_snapshot_caps_and_keeps_elements():
    elements = [{"ref": f"e{i}", "role": "button", "name": f"Button number {i}"} for i in range(1, 400)]
    text, cut = format_snapshot("Shop", "https://x.test", elements, "word " * 5000, 2_000, {"y": 0, "height": 3000, "view": 800})
    assert cut and len(text) <= 2_000
    assert text.startswith("Page: Shop\nURL: https://x.test\nScroll: 0px from top, 2200px more below")
    assert '[e1] button "Button number 1"' in text and "more elements not shown" in text
    small, cut2 = format_snapshot("T", "u", elements[:2], "Hello\n\n\n\nworld", 2_000)
    assert not cut2 and "Page text:\nHello\n\nworld" in small


def test_focus_on_question():
    text = "\n\n".join([f"Paragraph {i} about gardening and soil." for i in range(200)] + ["The refund policy lasts 30 days."])
    out, cut = focus_on_question(text, "what is the refund policy", 300)
    assert cut and "refund policy lasts 30 days" in out and len(out) < 400
    short, cut2 = focus_on_question("tiny page", "", 300)
    assert short == "tiny page" and not cut2


def test_refs_and_keys():
    assert _clean_ref("e12") == "e12" and _clean_ref("[E7]") == "e7" and _clean_ref("12") is None
    assert normalize_key("enter") == "Enter" and normalize_key("esc") == "Escape"
    assert normalize_key("ctrl+a") == "Control+a" and normalize_key("page down") == "PageDown"
    assert normalize_key("+") == "+" and normalize_key("f5") == "F5"
