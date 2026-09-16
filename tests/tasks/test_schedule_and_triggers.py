from datetime import UTC, datetime, timedelta

from sentient.tasks.schedule import (
    calculate_next_run,
    get_tz,
    normalize_schedule,
    parse_run_at,
)
from sentient.tasks.triggers import event_matches_filter


# ---------------------------------------------------------------------------- recurring
def test_daily_next_run_respects_timezone():
    s = {"type": "recurring", "frequency": "daily", "time": "09:00", "timezone": "Asia/Kolkata"}
    # 10:30 IST -> tomorrow 09:00 IST
    assert calculate_next_run(s, datetime(2026, 9, 15, 5, 0, tzinfo=UTC)) == datetime(2026, 9, 16, 3, 30, tzinfo=UTC)
    # 07:30 IST -> today 09:00 IST
    assert calculate_next_run(s, datetime(2026, 9, 15, 2, 0, tzinfo=UTC)) == datetime(2026, 9, 15, 3, 30, tzinfo=UTC)
    # exactly at the run time -> strictly after
    assert calculate_next_run(s, datetime(2026, 9, 15, 3, 30, tzinfo=UTC)) == datetime(2026, 9, 16, 3, 30, tzinfo=UTC)


def test_same_wall_time_differs_by_zone():
    base = {"type": "recurring", "frequency": "daily", "time": "09:00"}
    now = datetime(2026, 9, 15, 0, 0, tzinfo=UTC)
    la = calculate_next_run({**base, "timezone": "America/Los_Angeles"}, now)
    ist = calculate_next_run({**base, "timezone": "Asia/Kolkata"}, now)
    assert la == datetime(2026, 9, 15, 16, 0, tzinfo=UTC)
    assert ist == datetime(2026, 9, 15, 3, 30, tzinfo=UTC)


def test_weekly_named_days():
    s = {"type": "recurring", "frequency": "weekly", "days": ["Monday", "Friday"], "time": "09:00",
         "timezone": "America/New_York"}
    # Tuesday 2026-09-15 08:00 EDT -> Friday 2026-09-18 09:00 EDT
    assert calculate_next_run(s, datetime(2026, 9, 15, 12, 0, tzinfo=UTC)) == datetime(2026, 9, 18, 13, 0, tzinfo=UTC)
    # Friday after 09:00 -> next Monday
    assert calculate_next_run(s, datetime(2026, 9, 18, 14, 0, tzinfo=UTC)) == datetime(2026, 9, 21, 13, 0, tzinfo=UTC)


def test_weekly_across_dst_end_keeps_local_time():
    s = {"type": "recurring", "frequency": "weekly", "days": ["Sunday"], "time": "09:00", "timezone": "America/New_York"}
    # Monday 2026-10-26 (EDT) -> Sunday 2026-11-01 09:00 EST (UTC-5)
    assert calculate_next_run(s, datetime(2026, 10, 26, 12, 0, tzinfo=UTC)) == datetime(2026, 11, 1, 14, 0, tzinfo=UTC)


def test_weekdays_and_bad_input():
    s = normalize_schedule({"type": "recurring", "frequency": "weekdays", "time": "9am"}, "Asia/Kolkata")
    assert s["frequency"] == "weekly" and s["days"] == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    assert s["time"] == "09:00" and s["timezone"] == "Asia/Kolkata"
    # Friday 2026-09-18 10:00 IST -> Monday 09:00 IST
    assert calculate_next_run(s, datetime(2026, 9, 18, 4, 30, tzinfo=UTC)) == datetime(2026, 9, 21, 3, 30, tzinfo=UTC)
    assert calculate_next_run({"frequency": "monthly"}, datetime(2026, 9, 18, tzinfo=UTC)) is None
    assert normalize_schedule("garbage", "UTC") == {"type": "once", "run_at": None, "timezone": "UTC"}
    assert normalize_schedule({"type": "once", "run_at": "null"}, "UTC")["run_at"] is None


def test_run_at_and_timezones():
    assert parse_run_at({"type": "once", "run_at": "2026-09-16T09:00", "timezone": "Asia/Kolkata"}) == datetime(
        2026, 9, 16, 3, 30, tzinfo=UTC
    )
    assert parse_run_at({"run_at": "2026-09-16T09:00:00+00:00", "timezone": "Asia/Kolkata"}) == datetime(
        2026, 9, 16, 9, 0, tzinfo=UTC
    )
    assert parse_run_at({"run_at": None}) is None
    assert parse_run_at({"run_at": "tomorrow"}) is None
    assert datetime(2026, 1, 1, tzinfo=get_tz("UTC+05:30")).utcoffset() == timedelta(hours=5, minutes=30)
    assert get_tz("Not/AZone") is UTC


# ---------------------------------------------------------------------------- trigger DSL
EMAIL = {
    "id": "m1", "thread_id": "t1", "from": "Jane Doe <Jane@Example.com>", "sender_email": "jane@example.com",
    "to": "me@example.com", "subject": "Invoice #42 due Friday", "snippet": "Please pay", "body": "Please pay",
    "labels": ["INBOX", "IMPORTANT"], "url": "https://mail.google.com/x",
}
EVENT = {
    "id": "e1", "summary": "Team standup", "attendees": ["a@x.com", "b@x.com"], "organizer_email": "boss@x.com",
    "location": "Room 4",
}


def test_gmail_from_matches_email_case_insensitively():
    assert event_matches_filter(EMAIL, {"from": "jane@example.com"}, "gmail")
    assert event_matches_filter(EMAIL, {"from": "JANE@example.com"}, "gmail")
    assert event_matches_filter(EMAIL, {"from": "Jane <jane@example.com>"}, "gmail")
    assert not event_matches_filter(EMAIL, {"from": "bob@example.com"}, "gmail")
    assert event_matches_filter(EMAIL, {"from": {"$contains": "jane doe"}}, "gmail")
    assert event_matches_filter(EMAIL, {"from": {"$in": ["x@y.com", "jane@example.com"]}}, "gmail")
    assert not event_matches_filter({**EMAIL, "sender_email": None}, {"from": "bob@x.com"}, "gmail")
    assert event_matches_filter({**EMAIL, "sender_email": None}, {"from": {"$eq": "jane@example.com"}}, "gmail")


def test_field_operators():
    assert event_matches_filter(EMAIL, {"subject": {"$contains": "INVOICE"}}, "gmail")
    assert event_matches_filter(EMAIL, {"subject": {"$regex": r"#\d+"}}, "gmail")
    assert not event_matches_filter(EMAIL, {"subject": {"$regex": "[unclosed"}}, "gmail")
    assert event_matches_filter(EMAIL, {"labels": {"$in": ["IMPORTANT"]}}, "gmail")
    assert event_matches_filter(EMAIL, {"labels": "INBOX"}, "gmail")
    assert not event_matches_filter(EMAIL, {"labels": {"$nin": ["INBOX"]}}, "gmail")
    assert event_matches_filter(EMAIL, {"to": {"$ne": "other@example.com"}}, "gmail")
    assert event_matches_filter(EMAIL, {"subject": {"$eq": "Invoice #42 due Friday"}}, "gmail")
    assert not event_matches_filter(EMAIL, {"subject": "invoice #42 due friday"}, "gmail")  # v2: plain equality is exact
    assert not event_matches_filter(EMAIL, {"subject": {"$fuzzy": "x"}}, "gmail")


def test_logical_operators_and_empty_filter():
    assert event_matches_filter(EMAIL, {}, "gmail")
    assert event_matches_filter(EMAIL, None, "gmail")
    f_or = {"$or": [{"from": "bob@example.com"}, {"subject": {"$contains": "invoice"}}]}
    assert event_matches_filter(EMAIL, f_or, "gmail")
    f_and = {"$and": [{"from": "jane@example.com"}, {"subject": {"$contains": "receipt"}}]}
    assert not event_matches_filter(EMAIL, f_and, "gmail")
    assert event_matches_filter(EMAIL, {"$not": {"from": "bob@example.com"}}, "gmail")
    assert not event_matches_filter(EMAIL, {"$or": "not-a-list"}, "gmail")
    combined = {"$or": [{"subject": {"$contains": "invoice"}}], "from": "jane@example.com"}
    assert event_matches_filter(EMAIL, combined, "gmail")


def test_calendar_filters():
    assert event_matches_filter(EVENT, {"summary": {"$contains": "standup"}}, "gcalendar")
    assert event_matches_filter(EVENT, {"attendees": "b@x.com"}, "gcalendar")
    assert event_matches_filter(EVENT, {"organizer_email": "boss@x.com"}, "gcalendar")
    assert not event_matches_filter(EVENT, {"location": "Room 5"}, "gcalendar")
