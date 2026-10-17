"""Keeps a 1-1.5 h meeting smooth on a MacBook (also a fanless Air): when the Mac gets warm, runs
on battery or slows down, the *optional* live work is reduced - the grey preview refreshes less
often, the English preview of long sentences pauses, and in the worst case only the real lines
are made. The real lines (German + English) and the models are never touched.

Signals (polled every few seconds): macOS thermal pressure + thermal state, Low Power Mode,
battery, memory pressure, the translator's speed (tokens/s) against its first minutes (a Mac that
throttles gets slower before macOS says so), and for fanless Macs a budget of GPU-busy time.
(Speech-pass seconds are not used for speed: they depend on how long the sentence was.)
"""
from __future__ import annotations

import math
import re
import statistics
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    level: int
    name: str
    preview_slowdown: float          # x the normal live-preview interval (inf: no previews)
    en_preview_min_s: float | None   # English preview for lines at least this long (None: off)
    en_preview_every_s: float
    early: bool                      # transcribe during the speaker's pause
    recheck: bool                    # ask the translator again when a negation / number got lost


PLANS = (
    Plan(0, "full", 1.0, 4.0, 2.5, True, True),
    Plan(1, "eco", 1.8, 6.0, 4.0, True, True),
    Plan(2, "cool", 3.0, None, 0.0, True, True),
    Plan(3, "minimal", math.inf, None, 0.0, False, False),
)

PRESSURE_LEVEL = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}     # notify thermal pressure -> plan level
STATE_LEVEL = {0: 0, 1: 1, 2: 2, 3: 3}              # NSProcessInfo thermal state -> plan level


class Governor:
    TICK_S = 5.0
    BUSY_WINDOW_S = 120.0
    COOL_DOWN_S = {True: 180.0, False: 90.0}         # fanless / with fan: wait this long before easing off
    BUDGET = {(True, "ac"): 0.50, (True, "battery"): 0.40, (False, "ac"): 0.75, (False, "battery"): 0.55}

    def __init__(self, signals, performance: str = "auto", clock=time.monotonic):
        self.signals = signals
        self.performance = performance
        self.clock = clock
        self.level = 0
        self.stretch = 1.0
        self.reason = ""
        self._last_change = clock()
        self._bad = 0
        self._slow_level = 0          # the level slowness asked for (held until the speed recovers)
        self._hot = False             # macOS reports heat
        self._speed_clean = True      # no heat, Low Power Mode or memory pressure: speeds count for the baseline
        self._lpm = None
        self._busy: deque = deque()
        self._llm = deque(maxlen=12)  # recent translator tok/s
        self._llm_base: list = []     # tok/s of the first clean samples on a cool Mac
        self._fanless: bool | None = None
        self._power = (None, None)
        self._power_at = -1e9
        self.last = {}

    # ------------------------------------------------------------------ measurements
    def note_work(self, seconds: float) -> None:
        """Seconds the GPU spent on a speech pass or a translation."""
        if seconds > 0:
            self._busy.append((self.clock(), float(seconds)))

    def note_llm(self, tok_s: float) -> None:
        if tok_s and tok_s > 0:
            self._llm.append(tok_s)
            # the baseline: a Mac that is not hot, not in Low Power Mode, not short of memory and not known
            # to be slow (battery / saver levels are fine: they only change the optional previews)
            if self._speed_clean and not self._slow_level and len(self._llm_base) < 10:
                self._llm_base.append(tok_s)

    def reset_baselines(self) -> None:
        """A different speech or translation model is slower or faster by itself - not a sign of heat."""
        self._llm.clear()
        self._llm_base.clear()
        self._bad = 0
        self._slow_level = 0

    def speed_ratio(self) -> float:
        """1.0 = as fast as in the first minutes; 0.6 = 40 % slower (throttling or memory)."""
        if len(self._llm_base) >= 5 and len(self._llm) >= 3:
            return statistics.median(list(self._llm)[-5:]) / statistics.median(self._llm_base)
        return 1.0

    def busy_fraction(self) -> float:
        now = self.clock()
        while self._busy and now - self._busy[0][0] > self.BUSY_WINDOW_S:
            self._busy.popleft()
        return min(1.0, sum(s for _, s in list(self._busy)) / self.BUSY_WINDOW_S)   # (other threads append)

    # ------------------------------------------------------------------ decisions
    def fanless(self) -> bool:
        if self._fanless is None:
            self._fanless = bool(self.signals.is_fanless())
        return self._fanless

    def tick(self) -> bool:
        """Re-evaluate; True if the plan changed."""
        now = self.clock()
        sig = self.signals
        pressure, state, lpm, mem = (sig.thermal_pressure(), sig.thermal_state(), sig.low_power_mode(),
                                     sig.memory_pressure())
        if now - self._power_at >= 60:
            self._power, self._power_at = sig.power_source(), now
        src, pct = self._power
        fanless = self.fanless()

        reasons = []
        os_floor = max(PRESSURE_LEVEL.get(pressure or 0, 3), STATE_LEVEL.get(state or 0, 3))
        if self._hot and not os_floor:
            # the heat explained the slowness: judge the speed afresh, from samples taken after it
            self._slow_level, self._bad = 0, 0
            self._llm.clear()
        if self._lpm is not None and bool(lpm) != self._lpm:
            # Low Power Mode changes the translator's speed by itself (like another model): measure again
            self._llm.clear()
            self._llm_base.clear()
            self._slow_level, self._bad = 0, 0
        self._lpm = bool(lpm)
        self._hot = os_floor > 0
        self._speed_clean = not os_floor and not lpm and mem not in (2, 4)
        if os_floor:
            reasons.append("the Mac is warm")
        floor = os_floor
        if self.performance != "max":
            if lpm:
                floor = max(floor, 1)
                reasons.append("Low Power Mode")
            # a MacBook Pro keeps its speed on battery; a fanless Air heats up and drains faster
            low = pct is not None and pct < 30
            if src == "battery" and (fanless or low):
                floor = max(floor, 2 if (fanless and pct is not None and pct < 25) else 1)
                reasons.append("on battery" + (f" ({pct} %)" if pct is not None else ""))
            if mem == 2:
                floor = max(floor, 1)
                reasons.append("memory is tight")
            elif mem == 4:
                floor = max(floor, 2)
                reasons.append("memory is very tight")
        if self.performance == "saver":
            floor = max(floor, 1)
            reasons.append("battery saver")

        speed = self.speed_ratio()
        self._bad = self._bad + 1 if speed < 0.8 else 0
        if speed >= 0.9 or self.performance == "max":
            self._slow_level = 0              # fast again (or the user wants full speed): slowness asks nothing
        elif self._bad >= 3:
            # slowing down alone never goes beyond "cool": the real lines keep their early pass and checks
            self._slow_level = max(self._slow_level, min(2, self.level + (2 if speed < 0.6 else 1)))
            self._bad = 0
        if self._slow_level:
            reasons.append(f"it got {round((1 - speed) * 100)} % slower")
        target = max(floor, self._slow_level)

        old = (self.level, round(self.stretch, 2), re.sub(r"\d+", "#", self.reason))
        if target > self.level:
            self.level, self._last_change = target, now
        elif target < self.level and now - self._last_change >= self.COOL_DOWN_S[fanless]:
            self.level, self._last_change = self.level - 1, now     # one step per cool-down

        # a fanless Mac heats up from sustained work long before macOS says so: keep to a budget
        busy = self.busy_fraction()
        budget = self.BUDGET[(fanless, src or "ac")]
        if self.performance == "max" or not (fanless or self.performance == "saver"):
            self.stretch = 1.0
        elif busy > budget:
            self.stretch = min(4.0, self.stretch * 1.2)
            reasons.append("keeping the GPU load down")
        else:
            self.stretch = max(1.0, self.stretch * 0.9)

        self.reason = ", ".join(dict.fromkeys(reasons))
        self.last = {"pressure": pressure, "state": state, "lpm": lpm, "power": src, "battery": pct, "mem": mem,
                     "speed": round(speed, 2), "busy": round(busy, 2), "budget": budget, "fanless": fanless}
        # (a changed reason counts - "the Mac is warm" gone - but not a changed number in it)
        return (self.level, round(self.stretch, 2), re.sub(r"\d+", "#", self.reason)) != old

    def set_performance(self, performance: str) -> None:
        """The user's choice counts at once (no cool-down); what was measured (speed, GPU load) stays."""
        self.performance = performance
        self.level, self._last_change = 0, self.clock()
        self.tick()

    def plan(self) -> Plan:
        return PLANS[self.level]

    def preview_slowdown(self) -> float:
        return self.plan().preview_slowdown * self.stretch

    def reset(self) -> None:
        """A new meeting: start from full speed (the signals bring it back down if needed). The
        speeds measured at the end of the last meeting say nothing about a Mac that has cooled down."""
        self.level, self.stretch, self._bad, self._slow_level, self.reason = 0, 1.0, 0, 0, ""
        self._last_change = self.clock()
        self._busy.clear()
        self._llm.clear()

    def status(self) -> dict:
        p = self.plan()
        return {"level": p.level, "mode": p.name, "reason": self.reason, "stretch": round(self.stretch, 2),
                "performance": self.performance, **self.last}
