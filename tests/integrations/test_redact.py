"""One-time codes, magic sign-in links and password reset links never reach a model (#128).

Unit tests of the deterministic redactor in both directions: secrets are hidden, ordinary numbers and links are not.
The Gmail, IMAP, proactivity and follow-up paths are covered next to their own tests.
"""

from __future__ import annotations

import pytest

from sentient.config.schema import SentientConfig
from sentient.integrations import redact
from sentient.integrations.redact import (
    CODE_PLACEHOLDER,
    LINK_PLACEHOLDER,
    RESET_PLACEHOLDER,
    hide_email_secrets,
    hide_secrets,
)

RESET_URL = "https://example.com/reset?token=abcDEF1234567890xyz"
MAGIC_URL = "https://app.example.com/l/9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c"


@pytest.mark.parametrize("text, secret", [
    ("Your verification code is 482913.", "482913"),
    ("Your verification code is: 482913", "482913"),
    ("482913 is your Instagram code. Don't share it.", "482913"),
    ("G-482913 is your Google verification code.", "G-482913"),
    ("Your OTP for login to SomeBank is 482913. Valid for 10 minutes.", "482913"),
    ("Use code 482913 to sign in.", "482913"),
    ("Enter the code below to sign in:\n\n482913\n\nIt expires in 10 minutes.", "482913"),
    ("Your security code: X7K9P2", "X7K9P2"),
    ("Enter this code: 482913", "482913"),
    ("Use this code to log in: 5521", "5521"),
    ("Your one-time password is 123 456", "123 456"),
    ("Two-factor code: 908172", "908172"),
])
def test_codes_are_hidden(text, secret):
    out = hide_secrets(text)
    assert secret not in out and CODE_PLACEHOLDER in out


def test_a_code_alone_on_a_line_is_hidden_in_a_sign_in_email():
    body = "Hi Maya,\n\n  4 8 2 9 1 3\n\nThanks, the Acme team"
    assert "4 8 2 9 1 3" in hide_secrets(body)  # nothing says it is a code
    out = hide_secrets(body, subject="Your sign-in code")
    assert "4 8 2 9 1 3" not in out and CODE_PLACEHOLDER in out
    out = hide_secrets("Hello\n\n730215\n", sender="no-reply@accounts.example.com")
    assert "730215" not in out


@pytest.mark.parametrize("text", [
    "Your order #482913 has shipped. Total $1234.56",
    "Order number 482913 will arrive on 2026-10-12.",
    "Call me at 415-555-0134 or +1 415 555 0134.",
    "Use promo code SAVE20 for 20% off. Discount code 4821 too.",
    "The code is 1500 lines long.",
    "error code 1603 during install",
    "PIN code: 411001, Pune",
    "Meeting on October 12, 2026 at 14:30 in room 4021.",
    "Invoice 2026-0042 for 12,500 INR, due 15/10/2026.",
    "Your flight confirmation code is ABC12D",
    "Your booking confirmation code is 482913",
    "Your reservation code is 482913.",
    "Booking ref ABC123",
    "Your PNR is X7K9P2. Booking reference: 4ZQ9TR",
    "Ticket number 482913, order code 77812.",
    "482913 is your booking code.",
    "Your code is 4821 at the front desk.",
    "See you in 2026\n2026\n",
    "The budget is 250000 and the headcount is 12.",
    "Read the post https://blog.example.com/2026/10/how-we-build",
    "Doc: https://docs.google.com/document/d/1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789/edit",
    "Newsletter link https://click.mail.example.com/ls/click?upn=aB3dE5fG7hI9jK1lM3nO5pQ7",
    "Login at https://example.com/login",
    "Pull request https://github.com/acme/app/pull/4821 is ready.",
])
def test_ordinary_numbers_and_links_are_left_alone(text):
    assert hide_secrets(text) == text


def test_booking_emails_keep_their_codes():
    """Booking, order and ticket codes are what people ask for and give no account access: always kept."""
    body = "\n".join(["Your code is 482913.", "Enter this code at check-in: 482913", "", "ABC12D", ""])
    for subject in ("Your booking confirmation", "Flight booking confirmed: Mumbai to Delhi", "Your e-ticket",
                    "Order #77812 confirmed", "Reservation at Hotel Lumen"):
        assert hide_secrets(body, subject=subject, sender="no-reply@accounts.airline.example") == body
    pnr = "\n".join(["Confirmation code: X7K9P2", "Record locator", "X7K9P2"])
    assert hide_secrets(pnr, subject="Your trip to Delhi") == pnr
    # a sign-in cue in the subject still wins over a stray booking word
    out = hide_secrets("Your code is 482913", subject="Your verification code for your booking")
    assert "482913" not in out


def test_verification_codes_are_still_hidden_next_to_bookings():
    assert "482913" not in hide_secrets("Your verification code is 482913")
    assert hide_secrets("Your booking confirmation code is 482913") == "Your booking confirmation code is 482913"


def test_reset_and_magic_links_are_hidden():
    out = hide_secrets(f"Reset your password: {RESET_URL}")
    assert out == f"Reset your password: {RESET_PLACEHOLDER}"
    out = hide_secrets(f"Click to sign in:\n{MAGIC_URL}.")
    assert out == f"Click to sign in:\n{LINK_PLACEHOLDER}."
    out = hide_secrets("https://accounts.example.com/magic-link/eyJhbGciOiJIUzI1NiJ9abc123")
    assert out == LINK_PLACEHOLDER
    # a tracking link with no hint is kept, unless the email itself is a sign-in email
    tracked = "Confirm: https://click.sendgrid.net/ls/click?upn=aB3dE5fG7hI9jK1lM3nO5pQ7"
    assert hide_secrets(tracked) == tracked
    assert hide_secrets(tracked, subject="Confirm your email") == f"Confirm: {LINK_PLACEHOLDER}"


def test_hiding_is_idempotent():
    once = hide_secrets(f"Your verification code is 482913. Or reset: {RESET_URL}")
    assert hide_secrets(once) == once


def test_email_item_is_masked_in_subject_snippet_and_body():
    item = {"id": "m1", "subject": "482913 is your Acme code", "snippet": "Your verification code is 482913",
            "body": f"Your verification code is 482913.\nForgot your password? {RESET_URL}",
            "sender_email": "no-reply@acme.example", "url": "https://mail.google.com/mail/u/0/#all/m1"}
    out = hide_email_secrets(item)
    assert "482913" not in f"{out['subject']} {out['snippet']} {out['body']}"
    assert RESET_URL not in out["body"] and RESET_PLACEHOLDER in out["body"]
    assert out["url"] == "https://mail.google.com/mail/u/0/#all/m1"  # the link to open it in the mail app stays


def test_switch_defaults_on_and_is_described():
    cfg = SentientConfig()
    assert cfg.integrations.hide_one_time_codes is True
    field = type(cfg.integrations).model_fields["hide_one_time_codes"]
    assert field.description and chr(0x2014) not in field.description
    assert redact.enabled(type("App", (), {"config": cfg})()) is True
    cfg.integrations.hide_one_time_codes = False
    assert redact.enabled(type("Mgr", (), {"app": type("App", (), {"config": cfg})()})()) is False
    assert redact.enabled(object()) is True  # nothing to read: stay safe
