"""The cost meter and the monthly budget (#148).

Every model call the provider makes is counted here (``record``), with its price from LiteLLM's bundled price list
when the model is paid per token. Models on the user's Claude or ChatGPT plan have no price per call, so the meter
shows them as on the plan, and local models are free.

``admit`` runs before every call to a paid cloud model: once this month's spending reaches
``models.monthly_budget_usd`` the call is refused with a plain message (``BudgetReached``), unless the budget is set to
ask and the user chose Keep going for this month. A notification goes out once at 80% and once at 100%.

Days, weeks (from Monday) and months follow this computer's clock; the usage table stores UTC times.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sentient.llm.locality import billing, short_name
from sentient.llm.provider import BudgetReached

if TYPE_CHECKING:  # pragma: no cover
    from sentient.app import SentientApp

log = logging.getLogger(__name__)

WARN_PERCENT = 80
KEEP_GOING_META = "spending.keep_going"  # the month ("2026-10") the user chose Keep going for
NOTIFIED_META = "spending.notified"  # the last threshold notified: "2026-10:80" or "2026-10:100"
PERIODS = (("today", "Today"), ("week", "This week"), ("month", "This month"))


class BudgetError(Exception):
    """A plain sentence for the user; ``status`` is the HTTP status for routes."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


def money(usd: float) -> str:
    if 0 < usd < 0.01:
        return "less than $0.01"
    return f"${usd:,.2f}"


def period_starts(now: datetime | None = None) -> dict[str, datetime]:
    """The start of today, this week (Monday) and this month on this computer's clock, as UTC times."""
    local = (now or datetime.now(UTC)).astimezone()
    day = local.replace(hour=0, minute=0, second=0, microsecond=0)
    week = day - timedelta(days=day.weekday())
    month = day.replace(day=1)
    return {"today": day.astimezone(UTC), "week": week.astimezone(UTC), "month": month.astimezone(UTC)}


def month_key(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).astimezone().strftime("%Y-%m")


class Spending:
    def __init__(self, app: SentientApp):
        self.app = app

    @property
    def models(self) -> Any:
        return self.app.config.models

    # ------------------------------------------------------------------ the gate and the count (from the provider)
    async def admit(self, model: str, role: str) -> None:
        """Refuse a paid cloud model once this month's budget is used up. Plan and local models always pass."""
        budget = float(self.models.monthly_budget_usd or 0)
        if not budget or billing(model) != "paid":
            return
        spent = await self.month_spent()
        if spent < budget or await self.kept_going():
            return
        raise BudgetReached(self.refusal(model, budget))

    async def record(self, model: str, role: str | None, usage: dict, cost: float | None, source: str) -> None:
        paid = billing(model) == "paid"
        await self.app.store.record_usage(
            model, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0),
            role=role, source=source, cost_usd=float(cost) if paid and cost is not None else None,
        )
        if paid and cost:
            await self._maybe_notify()

    # ------------------------------------------------------------------ budget state
    async def month_spent(self) -> float:
        since = period_starts()["month"].isoformat()
        row = await self.app.store.fetchone("SELECT SUM(cost_usd) AS c FROM usage WHERE created_at >= ?", (since,))
        return float((row["c"] if row else None) or 0.0)

    async def kept_going(self) -> bool:
        """The user chose Keep going for this month (only when the budget is set to ask)."""
        if self.models.budget_at_limit != "ask":
            return False
        return (await self.app.store.get_meta(KEEP_GOING_META)) == month_key()

    async def keep_going(self) -> dict[str, Any]:
        """Allow paid cloud models again until the month ends (budget set to ask)."""
        if not self.models.monthly_budget_usd:
            raise BudgetError("There's no monthly budget set.")
        if self.models.budget_at_limit != "ask":
            raise BudgetError("Your budget is set to stop at the limit. Raise it in Settings > Usage to allow more.")
        await self.app.store.set_meta(KEEP_GOING_META, month_key())
        state = await self.budget()
        self.app.bus.publish("budget.updated", state)
        return state

    async def budget(self) -> dict[str, Any]:
        budget = float(self.models.monthly_budget_usd or 0)
        spent = await self.month_spent()
        at_limit = self.models.budget_at_limit
        out: dict[str, Any] = {"monthly_usd": budget, "at_limit": at_limit, "spent_usd": round(spent, 6),
                               "month": month_key(), "percent": None, "state": "off", "kept_going": False,
                               "message": None}
        if not budget:
            return out
        percent = int(100 * spent / budget)
        kept = await self.kept_going()
        out.update(percent=percent, kept_going=kept)
        if spent >= budget:
            out["state"] = "reached"
            out["message"] = (
                f"This month's {money(budget)} budget is used up ({money(spent)} so far). "
                + ("You chose to keep going this month." if kept else
                   "Paid cloud models are paused until next month. " + self._how_to_allow())
            )
        elif percent >= WARN_PERCENT:
            out["state"] = "warning"
            out["message"] = f"You've used {percent}% of this month's {money(budget)} budget ({money(spent)} so far)."
        else:
            out["state"] = "ok"
        return out

    def _how_to_allow(self) -> str:
        if self.models.budget_at_limit == "ask":
            return "Choose Keep going to allow more."
        return "Raise the budget in Settings > Usage to allow more."

    def refusal(self, model: str, budget: float) -> str:
        return (f"This month's {money(budget)} model budget is used up, so nothing was sent to {short_name(model)}. "
                + ("Choose Keep going this month to allow more, or use a local model."
                   if self.models.budget_at_limit == "ask"
                   else "Raise the budget in Settings > Usage, or use a local model."))

    async def _maybe_notify(self) -> None:
        """One notification at 80% and one at 100% each month."""
        state = await self.budget()
        level = {"warning": WARN_PERCENT, "reached": 100}.get(state["state"])
        if level is None:
            return
        month = state["month"]
        last = await self.app.store.get_meta(NOTIFIED_META) or ""
        done = int(last.split(":", 1)[1]) if last.startswith(f"{month}:") and last.split(":", 1)[1].isdigit() else 0
        if done >= level:
            return
        await self.app.store.set_meta(NOTIFIED_META, f"{month}:{level}")
        try:
            await self.app.notify("budget", state["message"], title="Model budget",
                                  payload={"percent": state["percent"], "state": state["state"]})
        except Exception:
            log.exception("could not send the budget notification")
        self.app.bus.publish("budget.updated", state)

    # ------------------------------------------------------------------ the meter
    async def report(self) -> dict[str, Any]:
        """Today, this week and this month per model and per role, with dollars for paid models and plan usage
        kept apart. Also the budget and whether Local only is on."""
        starts = period_starts()
        rows = await self.app.store.fetchall(
            "SELECT model, COALESCE(role, '') AS role, created_at, prompt_tokens AS p, completion_tokens AS c,"
            " cost_usd FROM usage WHERE created_at >= ?",
            (min(starts.values()).isoformat(),),
        )
        periods = []
        for key, label in PERIODS:
            since = starts[key].isoformat()
            periods.append({"key": key, "label": label, "since": since,
                            **_summarise([r for r in rows if r["created_at"] >= since])})
        return {"local_only": bool(self.models.local_only), "periods": periods, "budget": await self.budget()}


def _summarise(rows: list[Any]) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    roles: dict[str, dict[str, Any]] = {}
    totals = {"cost_usd": 0.0, "paid_calls": 0, "unpriced_calls": 0, "plan_calls": 0, "plan_tokens": 0,
              "local_calls": 0, "local_tokens": 0, "calls": 0, "tokens": 0}
    for r in rows:
        kind = billing(r["model"])
        tokens = int(r["p"] or 0) + int(r["c"] or 0)
        cost = r["cost_usd"]
        totals["calls"] += 1
        totals["tokens"] += tokens
        if kind == "paid":
            totals["paid_calls"] += 1
            if cost is None:
                totals["unpriced_calls"] += 1
            else:
                totals["cost_usd"] += float(cost)
        else:
            totals[f"{kind}_calls"] += 1
            totals[f"{kind}_tokens"] += tokens
        for table, key in ((models, r["model"]), (roles, r["role"] or "other")):
            item = table.get(key)
            if item is None:
                item = table[key] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": None,
                                     "unpriced_calls": 0, "billing": kind if table is models else None,
                                     "plan_calls": 0, "local_calls": 0}
            item["calls"] += 1
            item["prompt_tokens"] += int(r["p"] or 0)
            item["completion_tokens"] += int(r["c"] or 0)
            if kind == "paid":
                if cost is None:
                    item["unpriced_calls"] += 1
                else:
                    item["cost_usd"] = (item["cost_usd"] or 0.0) + float(cost)
            else:
                item[f"{kind}_calls"] += 1

    def ordered(table: dict[str, dict[str, Any]], name: str) -> list[dict[str, Any]]:
        out = [{name: k, **v} for k, v in table.items()]
        for item in out:
            if item["cost_usd"] is not None:
                item["cost_usd"] = round(item["cost_usd"], 6)
        if name == "role":
            for item in out:
                del item["billing"]
        else:
            for item in out:
                del item["plan_calls"], item["local_calls"]
        return sorted(out, key=lambda x: (-(x["cost_usd"] or 0.0), -(x["prompt_tokens"] + x["completion_tokens"])))

    totals["cost_usd"] = round(totals["cost_usd"], 6)
    return {**totals, "by_model": ordered(models, "model"), "by_role": ordered(roles, "role")}
