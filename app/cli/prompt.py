"""Interactive prompt built on prompt_toolkit."""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.styles import Style

from app.cli.input import CancelledInput
from app.user_data import history_file

COMMANDS = ["/add", "/list", "/debug", "/help", "/setup", "/status", "/clear", "exit", "quit"]

_STYLE = Style.from_dict(
    {
        "completion-menu.completion": "bg:#222222 #eeeeee",
        "completion-menu.completion.current": "bg:#008787 #ffffff",
        "completion-menu.meta.completion": "bg:#444444 #ffffff",
        "scrollbar.background": "bg:#222222",
        "scrollbar.button": "bg:#00aaaa",
    }
)


class CommandCompleter(Completer):
    """Autocomplete slash commands and /debug options."""

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        stripped = text.lstrip()
        if stripped.startswith("/debug "):
            parts = text.split(None, 1)
            if len(parts) == 2 and " " not in parts[1].lstrip():
                prefix = parts[1]
                for option in ("enable", "disable"):
                    if option.startswith(prefix):
                        yield Completion(option, start_position=-len(prefix))
            return
        if " " in text:
            return
        word = document.get_word_before_cursor(WORD=True)
        for command in COMMANDS:
            if command.startswith(word):
                yield Completion(command, start_position=-len(word))


def _key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("escape")
    def _cancel(event: KeyPressEvent) -> None:
        raise CancelledInput

    return bindings


def _bottom_toolbar() -> HTML:
    return HTML(
        "<ansibrightblack>↑↓ history · Tab complete · Ctrl+C cancel · Ctrl+D exit"
        "</ansibrightblack>"
    )


def create_session() -> PromptSession:
    return PromptSession(
        history=FileHistory(str(history_file())),
        completer=CommandCompleter(),
        key_bindings=_key_bindings(),
        style=_STYLE,
    )


def read_prompt(session: PromptSession) -> str:
    return session.prompt(
        HTML("<b><ansicyan>You</ansicyan></b> <ansigreen>❯</ansigreen> "),
        bottom_toolbar=_bottom_toolbar(),
    )
