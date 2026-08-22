"""Modern terminal UI helpers built on rich."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterator

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from rich.markup import escape


def setup_console() -> None:
    """Enable UTF-8 output and ANSI/VT processing on Windows consoles."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


setup_console()
console = Console(highlight=True)

LOGO = [
    " █████╗   ██╗  ██████╗   ██╗",
    "██╔══██╗  ██║  ██╔══██╗  ██║",
    "███████║  ██║  ██████╔╝  ██║",
    "██╔══██║  ██║  ██╔══██╗  ██║",
    "██║  ██║  ██║  ██║  ██║  ██║",
    "╚═╝  ╚═╝  ╚═╝  ╚═╝  ╚═╝  ╚═╝",
]

_LOGO_STYLES = ["bold bright_blue"] * (len(LOGO) // 2) + ["bold grey74"] * (len(LOGO) - len(LOGO) // 2)


def _logo_text() -> Text:
    text = Text()
    for index, line in enumerate(LOGO):
        text.append(line, style=_LOGO_STYLES[index])
        if index < len(LOGO) - 1:
            text.append("\n")
    return text


def print_banner(version: str) -> None:
    console.print()
    console.print(
        Panel(
            Group(
                _logo_text(),
                Text("multi-agent personal assistant", style="dim", justify="center"),
            ),
            border_style="cyan",
            padding=(1, 3),
            expand=False,
        ),
        justify="center",
    )
    console.print(Text(f"v{version}", style="dim", justify="center"))
    console.print(
        Text("Type /help for commands · exit to quit", style="dim", justify="center")
    )
    console.print()


def print_help() -> None:
    table = Table(
        box=None,
        show_header=False,
        padding=(0, 2),
        title="Commands",
        title_style="bold cyan",
    )
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="dim")
    rows = [
        ("/add <path>", "Add and index a document (PDF, TXT, MD)"),
        ("/list", "List indexed documents"),
        ("/debug enable|disable", "Toggle tracing of routing and tool calls"),
        ("/setup", "Interactive configuration wizard"),
        ("/status", "Show current configuration and services"),
        ("/model", "Switch to a different LLM model"),
        ("/clear", "Clear screen and reset the display"),
        ("/help", "Show this help"),
        ("exit", "Quit the assistant"),
    ]
    for command, description in rows:
        table.add_row(command, description)
    console.print(table)


def print_assistant(text: str) -> None:
    console.print()
    console.print(Rule(Text("airi", style="bold cyan"), style="dim", align="left"))
    console.print(Markdown(text))
    console.print()


def type_out(text: str) -> None:
    """Print assistant markdown with a human-like typing effect."""
    console.print()
    console.print(Rule(Text("airi", style="bold cyan"), style="dim", align="left"))
    if not text:
        console.print()
        return

    live = Live(
        Markdown(""),
        console=console,
        refresh_per_second=60,
        vertical_overflow="ellipsis",
    )
    live.start()
    try:
        step = max(1, len(text) // 160)
        for end in range(step, len(text) + step, step):
            live.update(Markdown(text[:end]))
            time.sleep(0.015)
        live.update(Markdown(text))
    finally:
        live.stop()


def print_debug(label: str, detail: str, label_style: str = "dim", detail_style: str = "") -> None:
    text = Text("  ")
    text.append(label, style=label_style)
    text.append(" ")
    text.append(detail, style=detail_style or "")
    console.print(text)


def info(message: str) -> None:
    console.print(f"[cyan]{message}[/cyan]")


def success(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def warn(message: str) -> None:
    console.print(f"[yellow]![/yellow] {message}")


def error(message: str) -> None:
    console.print(f"[red]✗[/red] {message}")


def clear_screen() -> None:
    """Clear the terminal screen."""
    console.clear()


@contextmanager
def status(message: str) -> Iterator[None]:
    with console.status(f"[cyan]{message}[/cyan]", spinner="dots") as spinner:
        yield spinner
