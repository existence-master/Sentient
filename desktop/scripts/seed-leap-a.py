"""Seed a throwaway SENTIENT_HOME with demo chats for tool cards, devices and messaging channels.

No LLM traffic: a tiny fake provider stands in while seeding, and every model role in the saved
config points at a black-holed address, so the real engine never reaches a model either.

    .venv/Scripts/python.exe desktop/scripts/seed-leap-a.py --home <dir> [--reset] [--theme dark|light|system]

Sessions and messages are written with plain SQL so ids and timestamps are fixed. Re-running on the
same home first deletes the rows of the fixed session ids. Engine tables for subagents, devices and
channels are filled only when they exist (columns matched by name); otherwise they are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sqlite3
import struct
import sys
import zlib
from datetime import UTC, datetime, timedelta
from math import cos, radians, sin
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MARKER = ".seeded-demo"
OFFLINE_API_BASE = "http://10.255.255.1:11434"
NOW = datetime.now(UTC).replace(microsecond=0)
GAPS = [38, 64, 27, 81, 45, 22, 57, 90, 33, 70, 49, 25, 86, 41]  # seconds between messages (20..90)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


class FakeProvider:
    """Deterministic, offline stand-in for app.llm. Seeding never calls a model."""

    def model_for(self, role: str) -> str:
        return f"fake/{role}"

    async def stream(self, role, messages, tools=None, *, model=None):
        from sentient.llm.provider import StreamChunk

        yield StreamChunk(text="ok", model="fake")
        yield StreamChunk(done=True, model="fake")

    async def complete_text(self, role, messages, *, model=None) -> str:
        return "ok"

    async def complete_json(self, role, messages, *, model=None):
        return {}

    async def embed(self, texts, *, model=None):
        return [[0.0] * 64 for _ in texts]


# ============================================================================ conversation content
BILLS = [("June", 282, 2540), ("July", 300, 2710), ("August", 354, 3220)]
AVG_KWH = round(sum(k for _, k, _ in BILLS) / len(BILLS))
CHANGE = f"{(BILLS[-1][1] - BILLS[-2][1]) / BILLS[-2][1] * 100:+.0f}%"
HIGHEST = max(BILLS, key=lambda b: b[1])[0]

BILL_CODE = '''import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("bills/electricity-2026.csv", newline="") as f:
    rows = [{"month": r["month"], "kwh": int(r["units_kwh"]), "amount": int(r["amount_inr"])}
            for r in csv.DictReader(f)]

last_three = rows[-3:]
average = statistics.mean(r["kwh"] for r in last_three)
change = (last_three[-1]["kwh"] - last_three[-2]["kwh"]) / last_three[-2]["kwh"] * 100
highest = max(last_three, key=lambda r: r["kwh"])

print(f"{'Month':<10}{'Units (kWh)':>12}{'Amount (Rs)':>14}")
for r in last_three:
    print(f"{r['month']:<10}{r['kwh']:>12}{r['amount']:>14,}")
print(f"Average: {average:.0f} kWh, change vs last month: {change:+.0f}%")
print(f"Highest: {highest['month']}")

with open("bill-comparison.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["month", "kwh", "amount"])
    writer.writeheader()
    writer.writerows(last_three)

plt.figure(figsize=(4.8, 2.8))
plt.bar([r["month"] for r in last_three], [r["kwh"] for r in last_three], color="#f5a524")
plt.title("Electricity use (kWh)")
plt.tight_layout()
plt.savefig("bill-comparison.png")
print("Saved bill-comparison.csv and bill-comparison.png")

result({"average_kwh": round(average), "change_vs_last_month": f"{change:+.0f}%", "highest": highest["month"]})
'''


def bill_stdout() -> str:
    lines = [f"{'Month':<10}{'Units (kWh)':>12}{'Amount (Rs)':>14}"]
    lines += [f"{m:<10}{k:>12}{a:>14,}" for m, k, a in BILLS]
    lines.append(f"Average: {AVG_KWH} kWh, change vs last month: {CHANGE}")
    lines.append(f"Highest: {HIGHEST}")
    lines.append("Saved bill-comparison.csv and bill-comparison.png")
    return "\n".join(lines) + "\n"


RESTAURANT_GOAL = "Shortlist 3 quiet restaurants near Koregaon Park with good vegetarian options, open Saturday 8 pm"
RESTAURANT_SUMMARY = (
    "Three quiet places near Koregaon Park with strong vegetarian menus, all taking tables on Saturday at 8 pm:\n\n"
    "1. **The Flour Works**, Lane 5: European bistro with calm garden seating and plenty of vegetarian mains.\n"
    "2. **Olive Courtyard**, Lane 6: Mediterranean, soft music, a good mezze platter and a separate vegetarian menu.\n"
    "3. **Lotus Leaf Kitchen**, Lane 5: Southeast Asian with a big vegetarian section; quieter before 9 pm.\n\n"
    "All three are rated 4.3 or higher and take bookings on TableFinder."
)
LISTING_URL = "https://tablefinder.example/pune/koregaon-park"
LISTING_TITLE = "Restaurants in Koregaon Park, Pune - TableFinder"
BOOK_URL = "https://tablefinder.example/pune/the-flour-works/book"
SNAPSHOT_TEXT = "\n".join([
    '[e3] link "TableFinder"',
    '[e7] textbox "Search for restaurant, cuisine or a dish"',
    '[e9] button "Filters"',
    '[e12] link "The Flour Works"',
    "    European, Cafe. Lane 5, Koregaon Park. Rated 4.4. 650 m from Riverside Park",
    '[e14] button "Book a table"',
    '[e18] link "Olive Courtyard"',
    "    Mediterranean. Lane 6, Koregaon Park. Rated 4.3. 1.2 km from Riverside Park",
    '[e20] button "Book a table"',
    '[e24] link "Lotus Leaf Kitchen"',
    "    Asian, Thai. Lane 5, Koregaon Park. Rated 4.4. 900 m from Riverside Park",
    '[e26] button "Book a table"',
])
MAIN_FINAL = (
    "**Electricity bills**\n\n"
    f"Your last three bills averaged **{AVG_KWH} kWh** a month. {HIGHEST} was the highest at 354 kWh "
    f"(about Rs 3,220), **{CHANGE[1:]} more** than July. June and July came in at 282 and 300 kWh. "
    "I saved the comparison as `bill-comparison.csv` and a small bar chart as `bill-comparison.png` in your outputs folder.\n\n"
    "**Saturday dinner**\n\n"
    "I suggest **The Flour Works** on Lane 5, Koregaon Park. It is about an 8 minute walk from Riverside Park, "
    "has calm garden seating and lots of vegetarian mains, and has tables at 8 pm on Saturday.\n\n"
    "If it is full, **Lotus Leaf Kitchen** (Lane 5) is also walkable, and **Olive Courtyard** (Lane 6) is a short auto ride.\n\n"
    "The booking page for The Flour Works is open in the browser. Pick 8 pm and the number of guests, "
    "then confirm the reservation yourself when you are ready."
)

DESK_GOAL = "Compare 4 standing desks under 40,000 rupees available in Pune"
DESK_SUMMARY = (
    "Your standing desk comparison is ready. All four deliver to Pune and cost under Rs 40,000.\n\n"
    "| Desk | Price | Motors | Height range | Warranty | Delivery |\n"
    "|---|---|---|---|---|---|\n"
    "| Featherlite Optima Dual Motor | Rs 36,500 | Dual | 62 to 127 cm | 5 years | 5 days |\n"
    "| Green Soul Ergo Pro | Rs 32,999 | Dual | 60 to 125 cm | 3 years | 4 days |\n"
    "| Sunon Electric Desk | Rs 27,900 | Single | 72 to 121 cm | 3 years | 7 days |\n"
    "| Wakefit Stand Up Desk | Rs 24,999 | Single | 71 to 117 cm | 1 year | 3 days |\n\n"
    "**My pick:** the Featherlite Optima. Dual motors stay steady with two monitors, and the 5 year warranty "
    "is the longest here. If you want to save, the Green Soul Ergo Pro is nearly as sturdy for Rs 3,500 less."
)

RENAME_CODE = '''import os
from datetime import datetime
from pathlib import Path

folder = Path("Pictures") / "Goa 2025"
names = sorted(os.listdir(folder))
renamed = 0
for name in names:
    if not name.lower().endswith((".jpg", ".jpeg", ".heic")):
        continue
    photo = folder / name
    taken = datetime.fromtimestamp(photo.stat().st_mtime)
    photo.rename(folder / f"{taken:%Y-%m-%d_%H%M%S}_{name}")
    renamed += 1

print(f"Renamed {renamed} photos")
result({"renamed": renamed})
'''
RENAME_STDERR = (
    "Traceback (most recent call last):\n"
    '  File "script.py", line 6, in <module>\n'
    "    names = sorted(os.listdir(folder))\n"
    "FileNotFoundError: [WinError 3] The system cannot find the path specified: 'Pictures\\\\Goa 2025'\n"
)


class Convo:
    """Collects message rows for one session; timestamps are assigned afterwards."""

    def __init__(self, sid: str, channel: str, title: str, ends_ago: timedelta):
        self.sid, self.channel, self.title, self.ends_ago = sid, channel, title, ends_ago
        self.rows: list[dict[str, Any]] = []
        self.calls = 0

    def user(self, text: str) -> None:
        self.rows.append({"role": "user", "content": text})

    def assistant(self, text: str, *, thinking: str | None = None) -> None:
        self.rows.append({"role": "assistant", "content": text, "thinking": thinking})

    def tool(self, name: str, args: dict, result: dict, *, text: str = "", thinking: str | None = None) -> str:
        self.calls += 1
        cid = f"call_{self.sid.replace('-', '_')}_{self.calls:02d}"
        call = {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        self.rows.append({"role": "assistant", "content": text, "thinking": thinking, "tool_calls": [call]})
        self.rows.append({"role": "tool", "tool_call_id": cid, "name": name,
                          "content": json.dumps(result, ensure_ascii=False)})
        return cid

    def stamp(self) -> None:
        end = NOW - self.ends_ago
        gaps = [GAPS[i % len(GAPS)] for i in range(len(self.rows) - 1)]
        t = end - timedelta(seconds=sum(gaps))
        for i, row in enumerate(self.rows):
            row["id"] = f"{self.sid}-{i + 1:02d}"
            row["created_at"] = iso(t)
            if i < len(gaps):
                t += timedelta(seconds=gaps[i])


def build_conversations() -> tuple[list[Convo], dict[str, Any]]:
    refs: dict[str, Any] = {}

    # 1. main: code, foreground subagent, interjection, browser ------------------
    main = Convo("leap-a-main", "desktop", "Electricity bills and Saturday dinner", timedelta(minutes=4))
    main.user(
        "Can you compare my last three electricity bills from my files? And then find a quiet place for dinner "
        "near Koregaon Park on Saturday at 8, somewhere with good vegetarian food."
    )
    main.tool(
        "execute_code", {"code": BILL_CODE, "purpose": "Compare the last three electricity bills"},
        {
            "ok": True, "backend": "process", "stdout": bill_stdout(), "stderr": "",
            "result": {"average_kwh": AVG_KWH, "change_vs_last_month": CHANGE, "highest": HIGHEST},
            "files_created": ["bill-comparison.csv", "bill-comparison.png"], "tool_calls": 0,
            "duration_ms": 2140, "error": None,
        },
        text="Let me crunch the numbers first.",
        thinking="Two parts. The bills are in files/bills as a CSV, so a short script can compare them and save a "
        "chart. Then I can hand the restaurant search to a helper.",
    )
    refs["main_delegate_idx"] = len(main.rows)
    refs["main_delegate_call"] = main.tool(
        "delegate_task",
        {
            "goal": RESTAURANT_GOAL,
            "context": "Sarthak lives in Pune and prefers calm places where you can talk. Table for two or three, "
            "Saturday at 8 pm. Vegetarian options matter.",
            "background": False,
        },
        {"subagent_id": "sub-leap-a-1", "status": "completed", "summary": RESTAURANT_SUMMARY, "files_created": []},
        text="Now I will ask a helper to shortlist restaurants.",
    )
    main.user("Oh and it should be walking distance from the Riverside Park if possible")
    main.tool("browser_open", {"url": LISTING_URL}, {"ok": True, "url": LISTING_URL, "title": LISTING_TITLE},
              thinking="He wants it walkable from Riverside Park. TableFinder lists distance, so I will check there.")
    main.tool("browser_snapshot", {}, {"url": LISTING_URL, "title": LISTING_TITLE, "text": SNAPSHOT_TEXT})
    main.tool("browser_click", {"ref": "e14"},
              {"ok": True, "url": BOOK_URL, "title": "Book a table - The Flour Works"})
    main.assistant(MAIN_FINAL)

    # 2. background subagent ---------------------------------------------------------
    bg = Convo("leap-a-background", "desktop", "Research standing desks", timedelta(days=1, hours=3, minutes=12))
    bg.user("Can you research standing desks for me? Sturdy enough for two monitors, under 40k, and available in Pune.")
    refs["bg_delegate_idx"] = len(bg.rows)
    refs["bg_delegate_call"] = bg.tool(
        "delegate_task",
        {
            "goal": DESK_GOAL,
            "context": "Two monitors on the desk, used for long work days. Prefer dual motor frames and delivery to Pune.",
            "background": True,
        },
        {"subagent_id": "sub-leap-a-2", "status": "running"},
    )
    bg.assistant("I have a helper researching that in the background. I will post the comparison here when it is ready.")
    bg.assistant(DESK_SUMMARY)

    # 3. telegram ----------------------------------------------------------------------
    tg = Convo("leap-a-telegram", "telegram", "Groceries for the week", timedelta(days=1, hours=9, minutes=40))
    tg.user("hey can u make a grocery list for the week")
    tg.assistant("Sure. Your usual, or anything different this week?")
    tg.user("usual + oats, bananas and paneer. also out of coffee")
    tg.assistant(
        "Here you go:\n\n- Oats, bananas, paneer\n- Filter coffee powder\n- Tomatoes, onions, spinach\n"
        "- Rice, toor dal\n- Curd, eggs, bread"
    )
    tg.user("perfect thanks")
    tg.assistant("Anytime. Happy shopping!")

    # 4. glasses -----------------------------------------------------------------------
    gl = Convo("leap-a-glasses", "glasses", "What plant is this?", timedelta(days=2, hours=1, minutes=5))
    gl.user("what plant is this on the balcony")
    gl.tool(
        "device_take_photo", {"device": "glasses", "question": "What plant is this?"},
        {"file": "outputs/glasses-photo-leap-a.png",
         "description": "A money plant (golden pothos) in a terracotta pot, leaves slightly yellow at the edges."},
    )
    gl.assistant(
        "That is a money plant, also called golden pothos. The yellow edges usually mean a little too much water "
        "or strong sun. Let the soil dry out between waterings and it should perk up."
    )

    # 5. voice -------------------------------------------------------------------------
    vo = Convo("leap-a-voice", "voice", "Morning check-in", timedelta(days=2, hours=6, minutes=30))
    vo.user("Good morning. What does my day look like?")
    vo.assistant(
        "Morning, Sarthak. Your morning is clear for deep work. At 3 you have a call with Rohan about the glasses "
        "prototype. It should be sunny in Pune, around 29 degrees."
    )
    vo.user("Okay. Remind me to call the electrician before lunch.")
    vo.assistant("Sure. I will remind you at 12:30 to call the electrician.")

    # 6. code error --------------------------------------------------------------------
    ce = Convo("leap-a-code-error", "desktop", "Rename my holiday photos", timedelta(days=2, hours=20, minutes=15))
    ce.user("Rename my holiday photos in Pictures/Goa 2025 so each name starts with the date it was taken")
    ce.tool(
        "execute_code", {"code": RENAME_CODE, "purpose": "Rename the Goa 2025 photos by the date they were taken"},
        {"ok": False, "backend": "process", "stdout": "", "stderr": RENAME_STDERR, "result": None,
         "files_created": [], "tool_calls": 0, "duration_ms": 380,
         "error": "The folder Pictures/Goa 2025 was not found"},
        text="I will write a short script that renames them by the date each photo was taken.",
    )
    ce.assistant(
        "I could not rename anything because there is no folder called **Goa 2025** inside Pictures. "
        "Nothing was changed.\n\n"
        "It may have a slightly different name, like *Goa-2025* or *Goa trip*, or the photos might still be on "
        "your phone. Tell me where they are, or drop one photo here, and I will try again."
    )

    # 7. declined approvals: new engine shape (declined: true) and an older transcript (error text only)
    dc = Convo("leap-a-declined", "desktop", "Order a blue mug", timedelta(minutes=1))
    shop = "https://mugshop.example/shop"
    dc.user("Add the blue mug to the cart and then click the Place order button.")
    dc.tool("browser_open", {"url": shop}, {"ok": True, "url": shop, "title": "Mug Shop"})
    dc.tool("browser_click", {"ref": '[e1] button "Add to cart"'},
            {"ok": True, "url": "https://mugshop.example/cart", "title": "Your cart - Mug Shop"})
    dc.tool(
        "browser_click", {"ref": '[e4] button "Place order"'},
        {"error": "NOT DONE. The user declined this action, so it did not happen. Tell the user plainly that it was "
                  "not done and ask what they would like instead.", "declined": True},
    )
    dc.tool(
        "execute_code", {"code": "total = 499 + 60\nprint(f'Total with delivery: Rs {total}')", "purpose": "Work out the order total"},
        {"error": "The user declined this action."},
        text="Let me at least work out the total for you.",
    )
    dc.assistant(
        "I did **not** place the order, because you said no. The blue mug is still in your cart on Mug Shop, "
        "so you can check out yourself whenever you like. I also did not run the total calculation."
    )

    convos = [main, bg, tg, gl, vo, ce, dc]
    for c in convos:
        c.stamp()
    return convos, refs


# ============================================================================ images
def write_png(path: Path, width: int, height: int, buf: bytearray, annotate=None) -> str:
    try:
        from PIL import Image, ImageDraw

        img = Image.frombytes("RGB", (width, height), bytes(buf))
        if annotate:
            annotate(ImageDraw.Draw(img))
        img.save(path, optimize=True)
        return "pillow"
    except ImportError:
        pass

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    stride = width * 3
    raw = b"".join(b"\x00" + bytes(buf[y * stride:(y + 1) * stride]) for y in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return "zlib"


def fill_rect(buf: bytearray, width: int, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    run = bytes(color) * (x1 - x0)
    for y in range(y0, y1):
        buf[(y * width + x0) * 3:(y * width + x1) * 3] = run


def bill_chart(path: Path) -> str:
    w, h = 480, 280
    left, right, top, base = 64, 456, 44, 236
    buf = bytearray(bytes((250, 248, 243)) * (w * h))
    scale = (base - top) / 400
    for kwh in (100, 200, 300, 400):
        y = base - round(kwh * scale)
        fill_rect(buf, w, left, y, right, y + 1, (228, 224, 215))
    fill_rect(buf, w, left, base, right, base + 2, (170, 165, 155))
    bars = []
    for i, (month, kwh, _) in enumerate(BILLS):
        cx = left + (right - left) * (2 * i + 1) // (2 * len(BILLS))
        y = base - round(kwh * scale)
        color = (232, 140, 20) if month == HIGHEST else (243, 190, 120)
        fill_rect(buf, w, cx - 42, y, cx + 42, base, color)
        bars.append((cx, y, month, kwh))

    def annotate(draw) -> None:
        from PIL import ImageFont

        try:
            font = ImageFont.load_default(size=13)
            title = ImageFont.load_default(size=16)
        except TypeError:
            font = title = ImageFont.load_default()
        draw.text((left, 14), "Electricity use, last three bills (kWh)", fill=(60, 56, 50), font=title)
        for kwh in (100, 200, 300, 400):
            draw.text((20, base - round(kwh * scale) - 8), str(kwh), fill=(120, 115, 105), font=font)
        for cx, y, month, kwh in bars:
            draw.text((cx - 12, y - 18), str(kwh), fill=(60, 56, 50), font=font)
            draw.text((cx - 22, base + 8), month, fill=(90, 85, 78), font=font)

    return write_png(path, w, h, buf, annotate)


LEAVES = [  # (cx, cy, half length, half width, angle degrees)
    (150, 150, 70, 38, -35), (260, 110, 64, 34, 10), (380, 140, 72, 38, 40), (490, 120, 66, 35, -20),
    (110, 290, 62, 33, 25), (220, 240, 74, 40, -15), (330, 260, 70, 37, 60), (450, 250, 76, 40, -45),
    (560, 260, 62, 33, 15), (180, 360, 58, 30, -60), (300, 350, 60, 32, 20), (420, 345, 58, 31, -25),
    (530, 380, 56, 30, 50), (70, 150, 52, 28, 70), (600, 150, 54, 29, -65),
]


def plant_photo(path: Path) -> str:
    w, h = 640, 480
    buf = bytearray(w * h * 3)
    for y in range(h):
        ty = y / h
        row = bytearray()
        for x in range(w):
            t = 0.6 * ty + 0.4 * x / w
            row += bytes((int(214 - 90 * t), int(232 - 70 * t), int(200 - 95 * t)))
        buf[y * w * 3:(y + 1) * w * 3] = row
    for cx, cy, a, b, deg in LEAVES:
        ca, sa = cos(radians(deg)), sin(radians(deg))
        for y in range(max(0, cy - a), min(h, cy + a)):
            for x in range(max(0, cx - a), min(w, cx + a)):
                dx, dy = x - cx, y - cy
                u, v = dx * ca + dy * sa, -dx * sa + dy * ca
                d = (u / a) ** 2 + (v / b) ** 2
                if d > 1:
                    continue
                shade = 1 - abs(v) / b
                color = [int(60 + 50 * shade), int(130 + 60 * shade), int(50 + 30 * shade)]
                if d > 0.8:  # slightly yellow edges
                    color = [int(c * 0.4 + y_ * 0.6) for c, y_ in zip(color, (205, 196, 92), strict=True)]
                if abs(v) < 1.6 and abs(u) < a * 0.85:  # centre vein
                    color = [min(255, c + 55) for c in color]
                i = (y * w + x) * 3
                buf[i:i + 3] = bytes(color)
    for y in range(392, h):  # terracotta pot
        half = 118 - (y - 392) * 30 // (h - 392)
        color = (168, 82, 52) if y < 412 else (192, 98, 62)
        fill_rect(buf, w, 320 - half, y, 320 + half, y + 1, color)
    return write_png(path, w, h, buf)


def write_files(home: Path) -> tuple[list[str], str]:
    out = home / "files" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    csv_lines = ["month,kwh,amount"] + [f"{m},{k},{a}" for m, k, a in BILLS]
    (out / "bill-comparison.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    encoder = bill_chart(out / "bill-comparison.png")
    plant_photo(out / "glasses-photo-leap-a.png")
    names = ["bill-comparison.csv", "bill-comparison.png", "glasses-photo-leap-a.png"]
    return [f"files/outputs/{n}" for n in names], encoder


# ============================================================================ engine tables
def encode(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


async def columns(store, table: str) -> list[str]:
    return [r["name"] for r in await store.fetchall(f"PRAGMA table_info({table})")]


async def insert_matching(store, table: str, records: list[dict]) -> int:
    cols = await columns(store, table)
    n = 0
    for rec in records:
        keys = [c for c in cols if c in rec]
        if not keys:
            continue
        await store.execute(
            f"INSERT OR REPLACE INTO {table}({', '.join(keys)}) VALUES({', '.join('?' for _ in keys)})",
            [encode(rec[k]) for k in keys],
        )
        n += 1
    return n


async def seed_engine_tables(store, convos: list[Convo], refs: dict) -> tuple[dict[str, int], list[str]]:
    by_id = {c.sid: c for c in convos}
    main, bg = by_id["leap-a-main"], by_id["leap-a-background"]
    sub1_start = main.rows[refs["main_delegate_idx"]]["created_at"]
    sub1_end = main.rows[refs["main_delegate_idx"] + 1]["created_at"]
    sub2_start = bg.rows[refs["bg_delegate_idx"]]["created_at"]
    subagents = [
        {"id": "sub-leap-a-1", "subagent_id": "sub-leap-a-1", "session_id": "leap-a-main",
         "parent_call_id": refs["main_delegate_call"], "goal": RESTAURANT_GOAL, "status": "completed",
         "background": 0, "summary": RESTAURANT_SUMMARY, "error": None, "tool_calls": 6,
         "started_at": sub1_start, "finished_at": sub1_end, "created_at": sub1_start, "updated_at": sub1_end},
        {"id": "sub-leap-a-2", "subagent_id": "sub-leap-a-2", "session_id": "leap-a-background",
         "parent_call_id": refs["bg_delegate_call"], "goal": DESK_GOAL, "status": "running",
         "background": 1, "summary": None, "error": None, "tool_calls": 3,
         "started_at": sub2_start, "finished_at": None, "created_at": sub2_start, "updated_at": sub2_start},
    ]
    nodes = [
        {"id": "desktop", "node_id": "desktop", "name": "This computer", "kind": "desktop", "platform": "win32",
         "capabilities": ["screen.capture", "camera.photo", "clipboard.read", "clipboard.write", "notify.show"],
         "online": 1, "last_seen_at": iso(NOW - timedelta(minutes=1)), "battery": None,
         "created_at": iso(NOW - timedelta(days=6, hours=2)), "revoked": 0},
        {"id": "node-phone", "node_id": "node-phone", "name": "Sarthak's phone", "kind": "phone",
         "platform": "android",
         "capabilities": ["camera.photo", "location.get", "notify.show", "speak", "mic.stream", "battery"],
         "online": 1, "battery": 0.72, "last_seen_at": iso(NOW - timedelta(minutes=2)),
         "created_at": iso(NOW - timedelta(days=5, hours=4)), "revoked": 0},
        {"id": "node-glasses", "node_id": "node-glasses", "name": "Smart glasses", "kind": "glasses",
         "platform": "glasses",
         "capabilities": ["camera.photo", "display.text", "display.card", "mic.stream", "button.events", "battery",
                          "speak"],
         "online": 0, "battery": 0.34, "last_seen_at": iso(NOW - timedelta(hours=3)),
         "created_at": iso(NOW - timedelta(days=4, hours=7)), "revoked": 0},
    ]
    channel_state = [{"id": "telegram", "channel": "telegram", "display_name": "Telegram", "enabled": 1,
                      "status": "connected", "account_label": "@sentient_sarthak_bot", "error": None,
                      "updated_at": iso(NOW - timedelta(days=2, hours=1))}]
    paired = [{"channel": "telegram", "channel_id": "telegram", "chat_id": "581230944", "label": "Sarthak Karandikar",
               "paired_at": iso(NOW - timedelta(days=2, hours=1)), "deliver": 1, "session_id": "leap-a-telegram"}]

    names = [r["name"] for r in await store.fetchall("SELECT name FROM sqlite_master WHERE type = 'table'")]
    targets: dict[str, list[tuple[str, list[dict]]]] = {"subagents": [], "nodes": [], "channels": [], "paired chats": []}
    for name in names:
        low = name.lower()
        if "fts" in low:
            continue
        cols = set(await columns(store, name))
        if low.startswith("subagent") and cols & {"goal", "status"}:
            targets["subagents"].append((name, subagents))
        elif low.startswith("node") and cols & {"name", "kind", "capabilities"}:
            targets["nodes"].append((name, nodes))
        elif low.startswith("channel"):
            if "chat_id" in cols and cols & {"label", "deliver", "paired_at"}:
                targets["paired chats"].append((name, paired))
            elif cols & {"status", "account_label"}:
                targets["channels"].append((name, channel_state))

    inserted: dict[str, int] = {}
    skipped: list[str] = []
    for thing, tables in targets.items():
        if not tables:
            msg = f"skipped {thing}: engine table not found yet"
            print(msg)
            skipped.append(msg)
            continue
        for table, records in tables:
            try:
                inserted[table] = await insert_matching(store, table, records)
            except sqlite3.Error as exc:
                msg = f"skipped {thing} ({table}): {exc}"
                print(msg)
                skipped.append(msg)
    return inserted, skipped


# ============================================================================ seeding
def configure(theme: str):
    from sentient.config import SentientConfig, save_config
    from sentient.config.schema import ModelRoles, ProviderConfig

    cfg = SentientConfig()
    cfg.assistant.name = "Sentient"
    cfg.assistant.user_name = "Sarthak"
    cfg.assistant.timezone = "Asia/Kolkata"
    cfg.assistant.location = "Pune, India"
    cfg.assistant.onboarding_complete = True
    cfg.ui.theme = theme
    # keep the real engine from reaching any model while the demo is open
    for role in ModelRoles.model_fields:
        setattr(cfg.models.roles, role, "ollama_chat/qwen3:8b")
    cfg.models.roles.embedding = "ollama/seeded-demo-no-embeddings"
    cfg.models.fallbacks = {}
    if hasattr(cfg.models, "request_timeout_s"):
        cfg.models.request_timeout_s = 3600
    for prefix in ("ollama", "ollama_chat"):
        cfg.models.providers[prefix] = ProviderConfig(api_base=OFFLINE_API_BASE)
    cfg.chat.auto_title = False
    cfg.memory.extract_after_turn = False
    cfg.memory.summaries_enabled = False
    cfg.memory.flush_enabled = False
    cfg.evolution.review_enabled = False
    cfg.evolution.curator_enabled = False
    cfg.evolution.user_profile_updates = False
    cfg.evolution.skill_repair = False
    cfg.proactivity.enabled = False
    cfg.user_model.enabled = False
    cfg.dreaming.enabled = False
    save_config(cfg)
    return cfg


async def seed(home: Path, theme: str) -> None:
    from sentient.app import SentientApp

    cfg = configure(theme)
    files, encoder = write_files(home)
    convos, refs = build_conversations()

    app = SentientApp(cfg, llm=FakeProvider(), db_path=home / "sentient.db", enable_background=False)
    await app.start()
    try:
        store = app.store
        ids = [c.sid for c in convos]
        marks = ", ".join("?" for _ in ids)
        await store.execute(f"DELETE FROM messages WHERE session_id IN ({marks})", ids)
        await store.execute(f"DELETE FROM sessions WHERE id IN ({marks})", ids)
        total = 0
        for c in convos:
            await store.execute(
                "INSERT INTO sessions(id, title, channel, created_at, updated_at) VALUES(?,?,?,?,?)",
                (c.sid, c.title, c.channel, c.rows[0]["created_at"], c.rows[-1]["created_at"]),
            )
            for row in c.rows:
                await store.execute(
                    "INSERT INTO messages(id, session_id, role, content, tool_calls, tool_call_id, name, thinking,"
                    " created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        row["id"], c.sid, row["role"], row.get("content"),
                        json.dumps(row["tool_calls"]) if row.get("tool_calls") else None,
                        row.get("tool_call_id"), row.get("name"), row.get("thinking"), row["created_at"],
                    ),
                )
                total += 1
        inserted, skipped = await seed_engine_tables(store, convos, refs)
        for key in ("memory.purge_last_run", "memory.summaries_last_run", "evolution.curator_last_run",
                    "evolution.profile_last_run", "proactivity.poll_last_run"):
            await store.set_meta(key, iso(NOW))
    finally:
        await app.stop()

    print(json.dumps({
        "home": str(home),
        "sessions": [{"id": c.sid, "channel": c.channel, "messages": len(c.rows), "updated_at": c.rows[-1]["created_at"]}
                     for c in convos],
        "messages": total,
        "files": files,
        "png_encoder": encoder,
        "engine_rows": inserted,
        "skipped": skipped,
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", default=os.environ.get("SENTIENT_HOME"), help="target SENTIENT_HOME (required)")
    ap.add_argument("--reset", action="store_true", help=f"wipe the home first (only if it contains {MARKER})")
    ap.add_argument("--theme", choices=["dark", "light", "system"], default="dark")
    args = ap.parse_args()
    if not args.home:
        ap.error("--home or SENTIENT_HOME is required")
    home = Path(args.home).expanduser().resolve()
    if home.exists() and any(home.iterdir()):
        if not args.reset:
            ap.error(f"{home} is not empty; pass --reset to re-seed it")
        if not (home / MARKER).exists():
            ap.error(f"refusing to wipe {home}: it was not created by a seed script")
        shutil.rmtree(home)
    home.mkdir(parents=True, exist_ok=True)
    (home / MARKER).write_text("created by desktop/scripts/seed-leap-a.py\n", encoding="utf-8")
    os.environ["SENTIENT_HOME"] = str(home)
    asyncio.run(seed(home, args.theme))


if __name__ == "__main__":
    main()
