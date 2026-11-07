"""Smart Business Insights - transparent, rule-based recommendation engine.

Every rule inspects one business signal, compares it with an explicit threshold and,
when triggered, emits an `Insight` that states the evidence, *why* it fired
(rule + threshold + observed value) and a recommended action.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics import occupancy
from utils.formatting import GYM_CAPACITY, WEEKDAYS, hour_label, money, pct

SEV_ORDER = {"critical": 0, "warning": 1, "opportunity": 2, "positive": 3}


@dataclass
class Insight:
    id: str
    title: str
    severity: str          # critical | warning | opportunity | positive
    category: str          # Retention | Revenue | Occupancy | Equipment | Growth | Operations
    metric: str
    evidence: str
    why: str
    action: str
    impact: float = 0.0    # estimated monthly € impact (for ranking), 0 if n/a


def _churn_risk(ctx) -> list[Insight]:
    sc = ctx.get("churn_scores")
    if sc is None or sc.empty:
        return []
    high = sc[sc["risk_band"] == "High"]
    share = len(high) / len(sc)
    at_risk = float(high["monthly_fee"].sum())
    if share < 0.03:
        return []
    sev = "critical" if share >= 0.08 else "warning"
    top = high.head(3)
    names = ", ".join(f"{r.first_name} {r.last_name[0]}. ({r.churn_probability:.0%})" for r in top.itertuples())
    drivers = high["risk_drivers"].str.split(", ").explode().value_counts()
    drivers = drivers[drivers.index != "—"].head(3)
    dtxt = ", ".join(f"{d} ({n})" for d, n in drivers.items())
    return [Insight(
        "churn_risk", "High membership cancellation risk", sev, "Retention",
        f"{len(high)} members · {money(at_risk)}/month at risk",
        f"<b>{len(high)}</b> of {len(sc)} active members ({pct(share)}) are in the <b>High</b> churn-risk band. "
        f"Most likely to leave: {names}. Most common risk drivers: {dtxt or 'n/a'}.",
        f"The churn model (Random Forest, 90-day horizon) scores every active member. This rule fires when "
        f"the high-risk share is ≥ 3% (critical ≥ 8%); observed {pct(share)}.",
        "Launch a personal retention call/offer for the high-risk list (Predictive Analytics → export).",
        at_risk)]


def _dormant(ctx) -> list[Insight]:
    sc = ctx.get("churn_scores")
    if sc is None or sc.empty:
        return []
    dormant = sc[sc["recency_days"] >= 21]
    share = len(dormant) / len(sc)
    if share < 0.05:
        return []
    return [Insight(
        "dormant", "Members who stopped coming", "warning", "Retention",
        f"{len(dormant)} members inactive ≥ 21 days",
        f"<b>{len(dormant)}</b> paying members ({pct(share)}) have not checked in for 3+ weeks. "
        f"Inactivity is one of the strongest predictors of cancellation in this dataset.",
        f"Rule: share of active members with no visit in the last 21 days ≥ 5%; observed {pct(share)}.",
        "Trigger an automated 'we miss you' sequence with a free PT session or class pass.",
        float(dormant["monthly_fee"].sum()) * 0.3)]


def _revenue_trend(ctx) -> list[Insight]:
    mo = ctx["monthly"]
    if len(mo) < 2:
        return []
    cur, prev = mo.iloc[-1], mo.iloc[-2]
    g = cur["revenue_growth"]
    month = cur["month"].strftime("%B %Y")
    pm = prev["month"].strftime("%B")
    out = []
    if g <= -0.05:
        d_mem = cur["membership_revenue"] - prev["membership_revenue"]
        d_anc = (cur["revenue"] - cur["membership_revenue"]) - (prev["revenue"] - prev["membership_revenue"])
        out.append(Insight(
            "revenue_drop", f"Revenue decreased vs {pm}", "critical" if g <= -0.10 else "warning", "Revenue",
            f"{pct(g, signed=True)} month-over-month",
            f"Revenue fell from <b>{money(prev['revenue'])}</b> to <b>{money(cur['revenue'])}</b> in {month}. "
            f"Membership fees changed by <b>{money(d_mem)}</b>, ancillary revenue (PT, shop) by <b>{money(d_anc)}</b>. "
            f"New sign-ups: {int(cur['new_members'])} vs {int(prev['new_members'])}; cancellations: "
            f"{int(cur['churned'])} vs {int(prev['churned'])}.",
            f"Rule: month-over-month revenue change ≤ −5% (critical ≤ −10%); observed {pct(g, signed=True)}. "
            f"Note: annual/quarterly plans make monthly revenue naturally lumpy - check the 3-month trend.",
            "Review the acquisition funnel and run a renewal campaign for plans expiring next month.",
            abs(cur["revenue"] - prev["revenue"])))
    elif g >= 0.05:
        out.append(Insight(
            "revenue_up", f"Revenue growing vs {pm}", "positive", "Revenue", f"{pct(g, signed=True)} month-over-month",
            f"Revenue reached <b>{money(cur['revenue'])}</b> in {month} (previous: {money(prev['revenue'])}).",
            f"Rule: month-over-month revenue change ≥ +5%; observed {pct(g, signed=True)}.",
            "Identify what drove the growth (campaign, season, plan mix) and double down on it."))
    if len(mo) >= 6:
        last3, prev3 = mo["revenue"].iloc[-3:].sum(), mo["revenue"].iloc[-6:-3].sum()
        g3 = (last3 - prev3) / prev3 if prev3 else 0
        if g3 <= -0.03:
            out.append(Insight(
                "revenue_3m", "Quarterly revenue momentum is negative", "warning", "Revenue",
                f"{pct(g3, signed=True)} last 3 months vs prior 3",
                f"The last three months generated <b>{money(last3)}</b> vs <b>{money(prev3)}</b> in the three before.",
                f"Rule: rolling 3-month revenue change ≤ −3%; observed {pct(g3, signed=True)}.",
                "Plan a seasonal promotion and upsell Premium / PT packs to existing members.", (prev3 - last3) / 3))
    return out


def _signups(ctx) -> list[Insight]:
    mo = ctx["monthly"]
    if len(mo) < 4:
        return []
    cur = mo.iloc[-1]
    ref = mo["new_members"].iloc[-4:-1].mean()
    if ref <= 0:
        return []
    d = (cur["new_members"] - ref) / ref
    if d > -0.25:
        return []
    return [Insight(
        "signups", "New registrations slowing down", "warning", "Growth",
        f"{int(cur['new_members'])} sign-ups ({pct(d, signed=True)} vs 3-month avg)",
        f"Only <b>{int(cur['new_members'])}</b> new members joined in {cur['month'].strftime('%B')}, versus an average "
        f"of <b>{ref:.0f}</b> over the previous three months.",
        f"Rule: new registrations ≥ 25% below the trailing 3-month average; observed {pct(d, signed=True)}.",
        "Boost acquisition: referral rewards, a free trial week and local social-media ads.")]


def _renewals(ctx) -> list[Insight]:
    subs = ctx["T"]["subscriptions"]
    ref = ctx["ref"]
    recent = subs[(subs["end_date"] > ref - pd.Timedelta(days=90)) & subs["status"].isin(["Renewed", "Cancelled"])]
    if len(recent) < 30:
        return []
    overall = (recent["status"] == "Renewed").mean()
    g = recent.groupby("billing_cycle")["status"].apply(lambda s: (s == "Renewed").mean())
    n = recent.groupby("billing_cycle").size()
    worst = g.idxmin()
    out = []
    if g[worst] < overall - 0.04 and n[worst] >= 15:
        out.append(Insight(
            "renewal_cycle", f"{worst} plans renew less often", "opportunity", "Retention",
            f"{pct(g[worst])} renewal rate vs {pct(overall)} overall",
            "Renewal rate over the last 90 days by billing cycle: " +
            ", ".join(f"<b>{k}</b> {pct(v)} (n={n[k]})" for k, v in g.items()) + ".",
            f"Rule: a billing cycle renews ≥ 4 pp below the overall rate; {worst} is "
            f"{(overall - g[worst]) * 100:.1f} pp below.",
            f"Offer {worst.lower()} members a discounted upgrade to a longer commitment before their renewal date."))
    m = ctx["T"]["members"]
    if "auto_renew" in m:
        ch = m.groupby("auto_renew")["churn_date"].apply(lambda s: s.notna().mean())
        if True in ch.index and False in ch.index and ch[True] > 0 and ch[False] / ch[True] >= 1.5:
            active_off = int(((m["status"] == "Active") & ~m["auto_renew"]).sum())
            out.append(Insight(
                "auto_renew", "Auto-renew is a strong retention lever", "opportunity", "Retention",
                f"{ch[False] / ch[True]:.1f}× higher churn without auto-renew",
                f"Historical churn is <b>{pct(ch[False])}</b> for members without auto-renew vs <b>{pct(ch[True])}</b> "
                f"with it. <b>{active_off}</b> active members still have auto-renew switched off.",
                "Rule: churn ratio (auto-renew off / on) ≥ 1.5×.",
                "Incentivise auto-renew activation (e.g. a free month or merch voucher) at the next touchpoint."))
    if overall < 0.9:
        out.append(Insight(
            "renewal_low", "Overall renewal rate below target", "warning", "Retention", f"{pct(overall)} renewal rate",
            f"Only {pct(overall)} of subscriptions that ended in the last 90 days were renewed.",
            "Rule: 90-day renewal rate < 90% target.",
            "Contact members 14 days before renewal with a personalised progress summary."))
    return out


def _occupancy(ctx) -> list[Insight]:
    att = ctx["T"]["attendance"]
    att = att[att["check_in"] > ctx["ref"] - pd.Timedelta(days=84)]
    if att.empty:
        return []
    conc = occupancy.concurrency_matrix(att)
    out = []
    peak = conc.stack().idxmax()
    pv = conc.loc[peak]
    if pv / GYM_CAPACITY >= 0.70:
        busy = (conc / GYM_CAPACITY >= 0.70).sum().sum()
        out.append(Insight(
            "crowding", "Peak-hour crowding", "warning" if pv / GYM_CAPACITY >= 0.85 else "opportunity", "Occupancy",
            f"{pct(pv / GYM_CAPACITY, 0)} of capacity on {peak[0]} {hour_label(peak[1])}",
            f"On average <b>{pv:.0f}</b> people are on the floor on {peak[0]} at {hour_label(peak[1])} "
            f"(capacity {GYM_CAPACITY}). <b>{busy}</b> weekly time slots exceed 70% capacity.",
            f"Rule: average concurrent occupancy ≥ 70% of capacity (warning ≥ 85%); observed {pct(pv / GYM_CAPACITY, 0)}.",
            "Shift demand with off-peak pricing, extra staff on the floor and class scheduling outside peaks."))
    weekday = conc.loc[WEEKDAYS[:5], [h for h in conc.columns if 9 <= h <= 21]]
    low = weekday.mean()
    low = low[low / GYM_CAPACITY < 0.15]
    if len(low):
        hrs = ", ".join(hour_label(h) for h in low.index[:5])
        out.append(Insight(
            "quiet", "Low attendance during weekday off-peak hours", "opportunity", "Occupancy",
            f"{len(low)} weekday hours below 15% occupancy",
            f"Weekday slots {hrs} average only <b>{low.mean():.0f}</b> people on the floor.",
            "Rule: weekday hours between 09:00 and 21:00 with average occupancy < 15% of capacity.",
            "Launch an off-peak membership, senior/student daytime offers or small-group classes in these slots."))
    return out


def _equipment(ctx) -> list[Insight]:
    s = ctx.get("equipment")
    if s is None or s.empty:
        return []
    out = []
    low = s[s["peer_util_ratio"] < 0.45]
    for r in low.itertuples():
        out.append(Insight(
            f"eq_low_{r.equipment_id}", f"{r.name} is unusually under-used", "opportunity", "Equipment",
            f"{pct(r.utilization)} utilisation ({pct(r.peer_util_ratio, 0)} of peers)",
            f"<b>{r.name}</b> ({r.zone}) runs at {pct(r.utilization)} utilisation versus a median of "
            f"{pct(r.utilization / r.peer_util_ratio)} for other {r.equipment_type} units.",
            "Rule: utilisation below 45% of the median of the same equipment type.",
            "Check for a hidden fault or poor placement; relocate it, promote it in programmes or redeploy the budget."))
    for r in s[s["service_overdue"]].itertuples():
        out.append(Insight(
            f"eq_overdue_{r.equipment_id}", f"{r.name}: preventive service overdue", "warning", "Equipment",
            f"{int(r.days_since_service)} days since last service",
            f"Recommended interval is <b>{int(r.service_interval_days)}</b> days; last preventive service on "
            f"{r.last_service:%d %b %Y}.",
            "Rule: days since last preventive service > 115% of the recommended interval.",
            "Schedule a preventive inspection this week to avoid breakdowns and safety issues."))
    for r in s[s["repairs_12m"] >= 4].itertuples():
        out.append(Insight(
            f"eq_fail_{r.equipment_id}", f"{r.name} breaks down repeatedly", "critical", "Equipment",
            f"{int(r.repairs_12m)} repairs in 12 months · {money(r.maintenance_cost)} total",
            f"<b>{r.name}</b> has needed {int(r.repairs_12m)} repairs in the last year "
            f"({r.downtime_hours:.0f} h of downtime overall, health score {int(r.health_score)}/100).",
            "Rule: ≥ 4 corrective repairs within 12 months.",
            "Compare cumulative repair cost with a replacement quote; consider replacing the unit.",
            float(r.maintenance_cost) / 24))
    return out


def _costs(ctx) -> list[Insight]:
    mo = ctx["monthly"]
    out = []
    if len(mo) >= 6:
        l3, p3 = mo["expenses"].iloc[-3:].sum(), mo["expenses"].iloc[-6:-3].sum()
        g = (l3 - p3) / p3 if p3 else 0
        if g >= 0.05:
            ex = ctx["T"]["expenses"]
            ex = ex[ex["month"] > mo["month"].iloc[-7]]
            by = ex.groupby([ex["month"] > mo["month"].iloc[-4], "category"])["amount"].sum().unstack(0).fillna(0)
            by["delta"] = by.get(True, 0) - by.get(False, 0)
            top = by["delta"].sort_values(ascending=False).head(2)
            out.append(Insight(
                "costs", "Operating expenses rising", "warning", "Operations", f"{pct(g, signed=True)} last 3 months",
                f"Expenses reached <b>{money(l3)}</b> over the last 3 months vs {money(p3)} before. Biggest increases: " +
                ", ".join(f"{k} (+{money(v)})" for k, v in top.items() if v > 0) + ".",
                f"Rule: rolling 3-month expenses up ≥ 5%; observed {pct(g, signed=True)}.",
                "Review supplier contracts and energy usage; check whether maintenance spikes are one-offs.",
                (l3 - p3) / 3))
    cur = mo.iloc[-1]
    if cur["profit_margin"] >= 0.25:
        out.append(Insight(
            "margin", "Healthy profit margin", "positive", "Revenue", f"{pct(cur['profit_margin'])} margin",
            f"Estimated profit for {cur['month']:%B} is <b>{money(cur['profit'])}</b> on {money(cur['revenue'])} revenue.",
            "Rule: monthly profit margin ≥ 25%.",
            "Consider reinvesting part of the margin in retention programmes and equipment renewal."))
    pay = ctx["T"]["payments"]
    last = pay[pay["payment_date"] > ctx["ref"] - pd.Timedelta(days=30)]
    if len(last) > 50:
        fr = (last["status"] == "Failed").mean()
        if fr >= 0.02:
            lost = last.loc[last["status"] == "Failed", "amount"].sum()
            out.append(Insight(
                "payments", "Failed payments to recover", "warning", "Revenue", f"{pct(fr)} failure rate · {money(lost)}",
                f"<b>{int((last['status'] == 'Failed').sum())}</b> payments failed in the last 30 days.",
                f"Rule: payment failure rate ≥ 2% over 30 days; observed {pct(fr)}.",
                "Enable automatic retry and send card-update reminders by SMS/e-mail.", float(lost)))
    return out


RULES = [_churn_risk, _dormant, _revenue_trend, _signups, _renewals, _occupancy, _equipment, _costs]


def generate_insights(T, monthly, ref, churn_scores=None, equipment=None) -> list[Insight]:
    ctx = {"T": T, "monthly": monthly, "ref": ref, "churn_scores": churn_scores, "equipment": equipment}
    out: list[Insight] = []
    for rule in RULES:
        try:
            out.extend(rule(ctx))
        except Exception as exc:  # a broken rule must never break the page
            out.append(Insight(f"err_{rule.__name__}", f"Rule {rule.__name__} could not run", "warning",
                               "Operations", "—", str(exc), "Rule raised an exception on the current data.",
                               "Check the imported data columns."))
    return sorted(out, key=lambda i: (SEV_ORDER.get(i.severity, 9), -i.impact))


def summary_counts(ins: list[Insight]) -> dict[str, int]:
    s = {k: 0 for k in SEV_ORDER}
    for i in ins:
        s[i.severity] = s.get(i.severity, 0) + 1
    return s


__all__ = ["Insight", "generate_insights", "summary_counts", "np"]
