"""system_control.py — Windows input injection for system-wide zoom control.

This module turns a zoom "action" into real Windows input using only the
standard library (ctypes + SendInput), so there are no extra dependencies.

Two zoom back-ends are provided:

* ``zoom_app(direction, steps)`` — sends ``Ctrl + MouseWheel`` to whatever
  application is under the mouse cursor.  Chrome, Edge, Firefox, Office, PDF
  viewers, image editors, VS Code, terminals ... all zoom with Ctrl+wheel.

* ``zoom_magnifier(zoom_in)`` — sends ``Win + Plus`` / ``Win + Minus`` which
  drives the built-in Windows Magnifier overlay (also launches it on first
  use).  Use this for applications that do not support Ctrl+wheel zoom
  (File Explorer, games, the desktop itself).

* ``zoom_reset()`` — sends ``Ctrl + 0`` to reset zoom to 100 % in apps that
  support it.

Every helper releases every modifier key it presses, even on error, so the
Ctrl / Win keys can never be left stuck.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

user32 = ctypes.windll.user32

# --------------------------------------------------------------------------- #
# Win32 constants
# --------------------------------------------------------------------------- #
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

KEYEVENTF_KEYUP = 0x0002

MOUSEEVENTF_WHEEL = 0x0800
WHEEL_DELTA = 120  # one wheel "notch"

VK_CONTROL = 0x11
VK_LWIN = 0x5B
VK_OEM_PLUS = 0xBB   # main-row '+'  (Win+Plus  => Magnifier zoom in / launch)
VK_OEM_MINUS = 0xBD  # main-row '-'  (Win+Minus => Magnifier zoom out)
VK_0 = 0x30          # Ctrl+0       => reset zoom to 100 %


# --------------------------------------------------------------------------- #
# SendInput structures (64-bit safe: dwExtraInfo is a pointer-sized field)
# --------------------------------------------------------------------------- #
class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    )


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    )


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _fields_ = (("type", wintypes.DWORD), ("union", _INPUTUNION))


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
_failure_warned = False


def _send(inputs: list) -> int:
    """Inject a batch of INPUT events. Returns the number of events accepted."""
    count = len(inputs)
    arr = (INPUT * count)(*inputs)
    return user32.SendInput(count, ctypes.byref(arr), ctypes.sizeof(INPUT))


def _warn_if_blocked(sent: int) -> None:
    """SendInput returns 0 when UIPI blocks injection (e.g. target is an
    elevated/admin process).  Warn once so the user knows why zoom is dead."""
    global _failure_warned
    if sent == 0 and not _failure_warned:
        _failure_warned = True
        print(
            "[warn] Windows blocked the simulated zoom input (SendInput returned 0). "
            "If the application you are zooming runs as Administrator, run this "
            "script as Administrator too."
        )


def _key(vk: int, down: bool) -> INPUT:
    flags = 0 if down else KEYEVENTF_KEYUP
    return INPUT(
        type=INPUT_KEYBOARD,
        union=_INPUTUNION(
            ki=KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=None)
        ),
    )


def _wheel(delta: int) -> INPUT:
    # mouseData is a 32-bit field; wrap negatives so the bit pattern is right.
    wrapped = delta & 0xFFFFFFFF
    return INPUT(
        type=INPUT_MOUSE,
        union=_INPUTUNION(
            mi=MOUSEINPUT(
                dx=0, dy=0, mouseData=wrapped,
                dwFlags=MOUSEEVENTF_WHEEL, time=0, dwExtraInfo=None,
            )
        ),
    )


def _hold_modifier(mod_vk: int, actions: list, settle: float = 0.012) -> None:
    """Press a modifier key, inject `actions`, then always release the key."""
    try:
        _send([_key(mod_vk, True)])
        time.sleep(settle)
        _warn_if_blocked(_send(actions))
        time.sleep(settle)
    finally:
        _send([_key(mod_vk, False)])


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def zoom_app(direction: int, steps: int = 1) -> None:
    """Send Ctrl+wheel.  direction: +1 = scroll up (zoom in), -1 = zoom out."""
    if not direction:
        return
    steps = max(1, int(steps))
    wheels = [_wheel(WHEEL_DELTA * direction) for _ in range(steps)]
    _hold_modifier(VK_CONTROL, wheels)


def zoom_magnifier(zoom_in: bool) -> None:
    """Send Win+Plus / Win+Minus (drives / launches the Windows Magnifier)."""
    vk = VK_OEM_PLUS if zoom_in else VK_OEM_MINUS
    _hold_modifier(VK_LWIN, [_key(vk, True), _key(vk, False)])


def zoom_reset() -> None:
    """Send Ctrl+0 -> reset zoom to 100 % (Chrome, Office, browsers, ...)."""
    _hold_modifier(VK_CONTROL, [_key(VK_0, True), _key(VK_0, False)])


def release_all_keys() -> None:
    """Make sure no modifier key we ever press is left stuck (called on exit)."""
    for vk in (VK_CONTROL, VK_LWIN):
        _send([_key(vk, False)])
