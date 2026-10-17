"""What macOS says about heat, power and memory (read-only, no admin rights, never raises).

* thermal pressure  - notify(3) key "com.apple.system.thermalpressurelevel": 0 nominal, 1 moderate,
                      2 heavy, 3 trapping, 4 sleeping (Apple Libc OSThermalNotification.h)
* thermal state     - NSProcessInfo.thermalState: 0 nominal, 1 fair, 2 serious, 3 critical (pyobjc)
* Low Power Mode    - NSProcessInfo.isLowPowerModeEnabled (macOS 12+)
* power source      - `pmset -g batt`: "Now drawing from 'AC Power'" / "'Battery Power'" + percent
* memory pressure   - sysctl kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical

Every getter returns None when the value is unknown (other systems, older macOS). Tests and
`toggle.sh doctor` can pretend with LT_FAKE_THERMAL=0..4, LT_FAKE_STATE=0..3, LT_FAKE_LPM=1,
LT_FAKE_POWER=battery:20, LT_FAKE_MEM=2, LT_FAKE_FANLESS=1.
"""
from __future__ import annotations

import ctypes
import logging
import os
import platform
import re
import subprocess

log = logging.getLogger("lt.signals")


def _fake(name: str):
    v = os.environ.get(name)
    return None if v in (None, "") else v


class MacSignals:
    def __init__(self):
        self.is_mac = platform.system() == "Darwin"
        self._sys = None
        self._tok = None
        self._pi = None
        if not self.is_mac:
            return
        try:
            s = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            s.notify_register_check.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
            s.notify_register_check.restype = ctypes.c_uint32
            s.notify_get_state.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
            s.notify_get_state.restype = ctypes.c_uint32
            s.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                       ctypes.c_void_p, ctypes.c_size_t]
            s.sysctlbyname.restype = ctypes.c_int
            tok = ctypes.c_int(0)
            if s.notify_register_check(b"com.apple.system.thermalpressurelevel", ctypes.byref(tok)) == 0:
                self._tok = tok.value
            self._sys = s
        except Exception:  # noqa: BLE001
            log.info("thermal pressure not readable", exc_info=True)
        try:
            from Foundation import NSProcessInfo  # pyobjc (installed with pyobjc-framework-AVFoundation)

            self._pi = NSProcessInfo.processInfo()
        except Exception:  # noqa: BLE001
            self._pi = None

    def thermal_pressure(self) -> int | None:
        f = _fake("LT_FAKE_THERMAL")
        if f is not None:
            return int(f)
        if self._sys is None or self._tok is None:
            return None
        v = ctypes.c_uint64(0)
        try:
            return int(v.value) if self._sys.notify_get_state(self._tok, ctypes.byref(v)) == 0 else None
        except Exception:  # noqa: BLE001
            return None

    def thermal_state(self) -> int | None:
        f = _fake("LT_FAKE_STATE")
        if f is not None:
            return int(f)
        try:
            return int(self._pi.thermalState()) if self._pi is not None else None
        except Exception:  # noqa: BLE001
            return None

    def low_power_mode(self) -> bool | None:
        f = _fake("LT_FAKE_LPM")
        if f is not None:
            return f == "1"
        try:
            if self._pi is not None and hasattr(self._pi, "isLowPowerModeEnabled"):
                return bool(self._pi.isLowPowerModeEnabled())
        except Exception:  # noqa: BLE001
            pass
        return None

    def memory_pressure(self) -> int | None:
        f = _fake("LT_FAKE_MEM")
        if f is not None:
            return int(f)
        if self._sys is None:
            return None
        v = ctypes.c_int(0)
        n = ctypes.c_size_t(ctypes.sizeof(v))
        try:
            rc = self._sys.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(v), ctypes.byref(n),
                                        None, 0)
        except Exception:  # noqa: BLE001
            return None
        return v.value if rc == 0 else None

    def power_source(self) -> tuple[str | None, int | None]:
        """("ac" | "battery" | None, battery percent | None)."""
        f = _fake("LT_FAKE_POWER")
        if f is not None:
            src, _, pct = f.partition(":")
            return src, (int(pct) if pct else None)
        if not self.is_mac:
            return None, None
        try:
            out = subprocess.run(["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True, timeout=3).stdout
        except Exception:  # noqa: BLE001
            return None, None
        m = re.search(r"drawing from '([^']+)'", out)
        p = re.search(r"(\d+)%", out)
        src = None if not m else ("battery" if "Battery" in m.group(1) else "ac")
        return src, (int(p.group(1)) if p else None)

    def model_name(self) -> str:
        if not self.is_mac:
            return platform.machine()
        try:
            out = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.model"], capture_output=True, text=True,
                                 timeout=3).stdout.strip()
            return out
        except Exception:  # noqa: BLE001
            return ""

    def is_fanless(self) -> bool:
        """MacBook Air (no fan): heat builds up, so it gets a tighter budget. Unknown Mac -> careful."""
        f = _fake("LT_FAKE_FANLESS")
        if f is not None:
            return f == "1"
        if not self.is_mac:
            return False
        try:
            out = subprocess.run(["/usr/sbin/system_profiler", "SPHardwareDataType"], capture_output=True,
                                 text=True, timeout=15).stdout
        except Exception:  # noqa: BLE001
            return True
        if "MacBook Pro" in out or "Mac mini" in out or "Mac Studio" in out or "iMac" in out or "Mac Pro" in out:
            return False
        return True
