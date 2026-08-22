"""Cross-platform line input with Ctrl+C and double-Escape cancellation."""
from __future__ import annotations

import sys
import time

if sys.platform == "win32":
    import msvcrt
else:
    import select
    import termios
    import tty


class CancelledInput(Exception):
    """Raised when the user cancels the current prompt."""


DOUBLE_ESC_TIMEOUT = 0.3


def read_input(prompt: str) -> str:
    """Read one line of input. Raises CancelledInput on double Escape, KeyboardInterrupt on Ctrl+C."""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    if sys.platform == "win32":
        return _read_windows()
    return _read_unix()


def _read_windows() -> str:
    chars: list[str] = []
    last_esc = 0.0
    while True:
        ch = msvcrt.getch()
        if ch == b"\x03":
            sys.stdout.write("\r\n")
            sys.stdout.flush()
            raise KeyboardInterrupt
        if ch == b"\x1b":
            now = time.monotonic()
            if last_esc and (now - last_esc) <= DOUBLE_ESC_TIMEOUT:
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                raise CancelledInput
            last_esc = now
            continue
        if ch in (b"\r", b"\n"):
            sys.stdout.write("\r\n")
            sys.stdout.flush()
            return "".join(chars)
        if ch == b"\x08" or ch == b"\x7f":
            if chars:
                chars.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            continue
        if ch in (b"\x00", b"\xe0"):
            try:
                msvcrt.getch()
            except Exception:
                pass
            continue
        if ch[0] < 32:
            continue
        try:
            c = ch.decode("utf-8", errors="ignore")
        except Exception:
            continue
        chars.append(c)
        sys.stdout.write(c)
        sys.stdout.flush()


def _read_unix() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    chars: list[str] = []
    last_esc = 0.0
    try:
        tty.setraw(fd, termios.TCSADRAIN)
        while True:
            select.select([sys.stdin], [], [])
            ch = sys.stdin.read(1)
            if not ch:
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                raise EOFError
            if ch == "\x03":
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                raise KeyboardInterrupt
            if ch == "\x1b":
                now = time.monotonic()
                if last_esc and (now - last_esc) <= DOUBLE_ESC_TIMEOUT:
                    sys.stdout.write("\r\n")
                    sys.stdout.flush()
                    raise CancelledInput
                last_esc = now
                extra = _read_unix_escape_sequence(fd)
                if extra:
                    continue
                continue
            if ch in ("\r", "\n"):
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return "".join(chars)
            if ch == "\x7f" or ch == "\x08":
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if ord(ch) < 32:
                continue
            chars.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _read_unix_escape_sequence(fd: int) -> str:
    """Peek for additional escape-sequence bytes and discard them."""
    end = time.monotonic() + 0.05
    collected = ""
    while time.monotonic() < end:
        r, _, _ = select.select([sys.stdin], [], [], 0.01)
        if r:
            collected += sys.stdin.read(1)
            if len(collected) >= 2:
                break
    return collected
