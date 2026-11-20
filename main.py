#!/usr/bin/env python3
"""Forecast GTA VI (PS5) trophy earn rates at a case's target time.

Model, per trophy:
  * A rate is earners / players_tracked, so the two parts are forecast
    separately and recombined (the decomposition view): earners only
    grow, while the tracked-player pool grows and dilutes the rate.
  * Progression view: if earning times are roughly log-normal, the
    probit of the earn rate is roughly linear in log time since
    release, so a weighted probit-on-log-time fit extrapolates the
    campaign curve.  Story trophies follow that curve closely;
    collection / stunt grinds are damped toward their last rate.
  * The median blends the two views with the last observed rate in
    logit space.  The spread grows with the square root of the
    horizon (a random walk on the logit scale), widens with the
    fit's residuals, and widens a lot when the recorded page never
    refreshed inside the history: a refresh the program has not seen
    is the biggest single risk, and the quantiles say so honestly.

Standard library only, deterministic, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from statistics import NormalDist

NORM = NormalDist()
EPS = 0.01
CEIL = 99.99


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(rate):
    p = clamp(rate, EPS, CEIL) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(z):
    if z >= 0:
        p = 1.0 / (1.0 + math.exp(-z))
    else:
        e = math.exp(z)
        p = e / (1.0 + e)
    return clamp(100.0 * p, EPS, CEIL)


def probit_pct(rate):
    p = clamp(rate, 0.02, 99.98) / 100.0
    return NORM.inv_cdf(p)


def parse_time(text):
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; returns (a, b, resid_sd)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return (ys[-1] if ys else 0.0), 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
    return a, b, math.sqrt(max(var, 0.0))


def trophy_kind(trophy):
    """Classify by wording: story progression vs grind vs platinum."""
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    if trophy.get("tier") == "platinum":
        return "platinum"
    grind_words = ("250", "100%", "all ", "every ", "each type", "20 different",
                   "gold medal in all", "at every location")
    if any(w in text for w in grind_words):
        return "grind"
    if "complete chapter" in text or "final mission" in text or "complete " in text:
        return "story"
    return "other"


KIND_SLOPE = {"story": 1.0, "other": 0.85, "grind": 0.6, "platinum": 0.45}
KIND_SIGMA = {"story": 1.15, "other": 1.0, "grind": 0.9, "platinum": 1.1}


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    zvals = [NORM.inv_cdf(clamp(t, 0.001, 0.999)) for t in levels]

    at_ts = parse_time(case["at"])
    target_ts = parse_time(case["target"])
    horizon_days = max((target_ts - at_ts) / 86400.0, 0.01)

    game = case.get("game") or {}
    try:
        release_ts = parse_time(game.get("release_date", "2026-11-19"))
    except Exception:
        release_ts = at_ts - 86400.0

    def hours_since_release(ts):
        return max((ts - release_ts) / 3600.0, 0.5)

    # Distinct page states only: between Exophase refreshes every read
    # is identical, and duplicates would fake a long flat history.
    states = []
    for read in history:
        key = (read.get("players_tracked"), tuple(
            -1.0 if r is None else r for r in read.get("rates") or []))
        if states and states[-1][0] == key:
            states[-1] = (key, read)
        else:
            states.append((key, read))
    reads = [r for _, r in states] if states else list(history)
    n_states = len(states)

    times = []
    players = []
    for read in reads:
        slot = read.get("slot") or read.get("at")
        times.append(parse_time(slot) if slot else at_ts)
        p = read.get("players_tracked")
        players.append(float(p) if p else 0.0)
    if not times:
        times = [at_ts]
        players = [0.0]
    t_last = times[-1]
    x_last = math.log(hours_since_release(t_last))
    x_target = math.log(hours_since_release(target_ts))

    def recency_weights(ts_list):
        return [0.5 ** ((t_last - t) / (36.0 * 3600.0)) for t in ts_list]

    # --- tracked-player pool: log-log growth, clamped to sane factors
    p_last = players[-1] if players[-1] > 0 else 1.0
    p_target = p_last
    if len(times) >= 2 and all(p > 0 for p in players):
        xs = [math.log(hours_since_release(t)) for t in times]
        ys = [math.log(p) for p in players]
        a, b, _ = wls_line(xs, ys, recency_weights(times))
        b = clamp(b, 0.0, 2.0)
        p_target = math.exp(a + b * x_target)
        p_target = clamp(p_target, p_last * 0.98, p_last * 6.0)

    rows = []
    for idx, trophy in enumerate(trophies):
        series = []
        for t, p, read in zip(times, players, reads):
            rates = read.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                series.append((t, p, float(rates[idx])))
        kind = trophy_kind(trophy)
        if not series:
            rows.append({"id": trophy["id"], "q": [0.4, 0.9, 1.8, 3.6, 8.0]})
            continue
        r_last = series[-1][2]
        l_last = logit_pct(r_last)
        ws = recency_weights([t for t, _, _ in series])
        xs = [math.log(hours_since_release(t)) for t, _, _ in series]
        damp = KIND_SLOPE[kind]

        # --- progression view: rate = cap * Phi(a + b * log time).
        # A plain probit line runs to 100%, but every trophy has a
        # ceiling well under it (most players never reach a given
        # chapter, let alone a grind), so the ceiling is fitted too:
        # try a grid of ceilings and keep the best weighted fit.
        # The fitted move away from the last rate is damped once, by
        # kind: a grind's curve bends far less than a chapter's.
        distinct_rates = len({r for _, _, r in series})
        if distinct_rates >= 3 and len(series) >= 3:
            r_max = max(r for _, _, r in series)
            best = None
            for cap_mult in (1.05, 1.15, 1.3, 1.5, 1.8, 2.2, 3.0, 4.5, 7.0, 12.0):
                cap = min(r_max * cap_mult, 100.0)
                if cap <= r_max * 1.01:
                    continue
                ys = []
                for _, _, r in series:
                    frac = clamp(r / cap, 0.0002, 0.9998)
                    ys.append(NORM.inv_cdf(frac))
                a, b, resid = wls_line(xs, ys, ws)
                sse = resid * resid
                if best is None or sse < best[0]:
                    best = (sse, cap, a, clamp(b, -1.0, 2.5), resid)
            _, cap, a, b, resid = best
            frac_t = clamp(NORM.cdf(a + b * x_target), 0.0002, 0.9998)
            pred_probit = cap * frac_t
            l_probit = l_last + damp * (logit_pct(pred_probit) - l_last)
            resid_logit = resid * 1.7
        else:
            l_probit = l_last
            resid_logit = 0.35

        # --- decomposition view: earners grow, pool dilutes
        earners = [r * p / 100.0 for _, p, r in series]
        if len(series) >= 3 and all(e >= 0 for e in earners):
            ys = [math.log(e + 0.5) for e in earners]
            a, b, _ = wls_line(xs, ys, ws)
            b = clamp(b, 0.0, 2.5)
            e_target = math.exp(a + b * x_target) - 0.5
            e_target = clamp(e_target, earners[-1] * 0.9, p_target)
            r_dec = clamp(100.0 * e_target / p_target, EPS, CEIL) if p_target > 0 else r_last
            l_dec = l_last + damp * (logit_pct(r_dec) - l_last)
        else:
            l_dec = l_last

        # --- blend the views in logit space
        l_med = 0.45 * l_probit + 0.35 * l_dec + 0.20 * l_last

        # A page that never refreshed inside the history can jump at
        # the next refresh: the pool swells and rates dilute.  Shift
        # the median a little down and let the spread carry the risk.
        stale = n_states <= 2
        if stale and r_last < 99.9:
            l_med -= 0.30

        # Safety rail: however strong a fitted curve looks, a rate
        # only moves so far in one horizon.  The cap on the logit
        # move grows slowly with the horizon and is widest for story
        # trophies, tightest for the platinum and the grinds.
        move_cap = 0.55 + 0.35 * math.log10(1.0 + horizon_days)
        move_cap *= {"story": 1.3, "other": 1.0, "grind": 0.7,
                     "platinum": 0.6}[kind]
        if stale:
            move_cap += 0.35
        l_med = clamp(l_med, l_last - move_cap, l_last + move_cap)

        sigma = 0.34 * math.sqrt(horizon_days) + 0.12
        sigma *= KIND_SIGMA[kind]
        sigma += clamp(0.5 * resid_logit, 0.0, 0.4)
        if stale:
            sigma += 0.55
        sigma = clamp(sigma, 0.15, 1.6)

        qs = []
        prev = 0.0
        for z in zvals:
            q = inv_logit_pct(l_med + z * sigma)
            q = max(q, prev)
            qs.append(round(q, 4))
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
