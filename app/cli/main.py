"""Interactive CLI for the personal AI assistant."""

import argparse
import ctypes
import importlib.metadata
import json
import re
import sys
import threading
import time
from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage

try:
    __version__ = importlib.metadata.version("ai-research-assistant")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.2.0"

from app.agents.automation_agent import AutomationAgent
from app.agents.user_context_agent import UserContextAgent
from app.agents.web_search_agent import WebSearchAgent
from app.chat_opencode import ChatOpenCode
from app.config import settings
from app.graph.workflow import build_assistant_graph
from app.services.embedding import Embeder
from app.services.vector_store import VectorStore
from app.services.document_service import DocumentService
from app.services.email_service import EmailService, SmtpConfig
from app.services.memory_service import MemoryService
from app.services.user_profile_service import UserProfileService
from app.tools.web_search_client import WebSearchClient
from app.user_data import config_file, history_file, storage_dir
from app.cli.setup import run_setup

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
BLUE = "\033[34m"

_COMMANDS = ["/add", "/list", "/debug", "/help", "/setup", "/status", "exit"]


def _c(code: str, text: str) -> str:
    return f"{code}{text}{RESET}"


def _setup_console() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _to_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in content
    )


def _plain_text(text: str) -> str:
    """Strip Markdown emphasis/headings/bullets from assistant output."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*([^*\n]+)\*", r"\1", text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[-*+]\s+", "", line)
        lines.append(line)
    return "\n".join(lines)


def _setup_history() -> None:
    """Enable command history with readline."""
    try:
        import readline

        hist = history_file()
        try:
            readline.read_history_file(str(hist))
        except (FileNotFoundError, OSError):
            pass
        readline.set_history_length(500)

        def completer(text, state):
            options = [cmd for cmd in _COMMANDS if cmd.startswith(text)]
            if state < len(options):
                return options[state]
            return None

        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")
    except ImportError:
        pass


def _save_history() -> None:
    """Persist command history to disk."""
    try:
        import readline

        readline.write_history_file(str(history_file()))
    except Exception:
        pass


@dataclass
class Services:
    """Composition root for the assistant's services."""

    memory: MemoryService
    documents: DocumentService
    email: EmailService
    profile: UserProfileService


def create_services() -> Services:
    """Build all services with their dependencies injected."""
    embeddings = Embeder.get_embedding_function()
    return Services(
        memory=MemoryService(embedding_function=embeddings),
        documents=DocumentService(vector_store=VectorStore()),
        email=EmailService(
            SmtpConfig(
                host=settings.smtp_host,
                port=settings.smtp_port,
                username=settings.smtp_username,
                password=settings.smtp_password,
                sender=settings.smtp_sender,
                imap_host=settings.imap_host,
                imap_port=settings.imap_port,
                imap_username=settings.imap_username,
                imap_password=settings.imap_password,
                imap_folder=settings.imap_folder,
            )
        ),
        profile=UserProfileService(),
    )


def create_assistant(llm: ChatOpenCode, services: Services):
    """Build the agents and compile the assistant graph."""
    agents = [
        WebSearchAgent(llm, WebSearchClient()),
        UserContextAgent(llm, services.memory, services.documents, services.profile),
        AutomationAgent(llm, services.email),
    ]
    return build_assistant_graph(llm, agents)


class DebugCallbackHandler(BaseCallbackHandler):
    """Prints tool calls and their results as they happen."""

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        name = serialized.get("name", "unknown")
        print(f"  {_c(DIM, '[tool]')} {_c(CYAN, name)}({_truncate(str(input_str), 100)})")

    def on_tool_end(self, output, **kwargs) -> None:
        content = getattr(output, "content", output)
        print(f"  {_c(DIM, '[result]')} <- {_truncate(str(content), 100)}")

    def on_tool_error(self, error, **kwargs) -> None:
        print(f"  {_c(RED, '[error]')} {_truncate(str(error), 100)}")


def _run(graph, history: list[BaseMessage], debug: bool):
    """Run the graph once. With debug, trace routing and tool activity."""
    if not debug:
        result = graph.invoke({"messages": history})
        return result["messages"][-1]

    final_answer = None
    handler = DebugCallbackHandler()
    for update in graph.stream(
        {"messages": history},
        stream_mode="updates",
        config={"callbacks": [handler]},
    ):
        for node, value in update.items():
            if node == "supervisor":
                nxt = value.get("next") if isinstance(value, dict) else None
                if nxt:
                    print(f"  {_c(DIM, '[route]')} {_c(BOLD, 'supervisor')} -> {_c(MAGENTA, str(nxt))}")
            elif isinstance(value, dict) and value.get("messages"):
                final_answer = value["messages"][-1]
    return final_answer


def _spin(stop_event: threading.Event, label: str) -> None:
    frames = ["|", "/", "-", "\\"]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r  {_c(DIM, label)} {_c(CYAN, frames[i % 4])}  ")
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(label) + 8) + "\r")
    sys.stdout.flush()


def _print_banner() -> None:
    bar = _c(DIM, "─" * 62)
    print(bar)
    print(f"  {_c(BOLD + CYAN, 'AI Research Assistant')}")
    print(f"  {_c(DIM, 'multi-agent personal assistant')}")
    print(bar)
    print(f"  {_c(DIM, 'Type')} {_c(BOLD, '/help')} {_c(DIM, 'for commands,')} {_c(BOLD, 'exit')} {_c(DIM, 'to quit.')}")
    print()


def _print_help() -> None:
    print(_c(BOLD + CYAN, "Commands:"))
    rows = [
        ("/add <path>", "Add and index a document (PDF, TXT, MD)"),
        ("/list", "List indexed documents"),
        ("/debug enable|disable", "Toggle tracing of routing and tool calls"),
        ("/setup", "Interactive configuration wizard"),
        ("/status", "Show current configuration and services"),
        ("/help", "Show this help"),
        ("exit", "Quit the assistant"),
    ]
    for cmd, desc in rows:
        print(f"  {_c(BOLD, cmd.ljust(22))} {_c(DIM, desc)}")


def _handle_add(document_service: DocumentService, raw_path: str) -> None:
    if not raw_path:
        print(f"  {_c(YELLOW, 'Usage:')} /add <document_path>")
        return
    try:
        info = document_service.add_document(raw_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  {_c(RED, 'Could not add document:')} {exc}")
        return
    print(f"  {_c(GREEN, 'Indexed')} '{info.name}' ({info.chunks} chunks).")


def _handle_list(document_service: DocumentService) -> None:
    documents = document_service.list_documents()
    if not documents:
        print(f"  {_c(DIM, 'No documents indexed yet. Use /add <document_path>.')}")
        return
    for doc in documents:
        added = doc.added_at or "unknown date"
        print(f"  {_c(BOLD, doc.name)}  {_c(DIM, f'({doc.chunks} chunks, added {added})')}")


def _handle_debug(argument: str, current: bool) -> bool:
    state = argument.lower()
    if state in {"enable", "on", "1"}:
        print(f"  Debug mode {_c(GREEN, 'enabled')} - routing and tool activity will be shown.")
        return True
    if state in {"disable", "off", "0"}:
        print(f"  Debug mode {_c(YELLOW, 'disabled')}.")
        return False
    status = _c(GREEN, "enabled") if current else _c(YELLOW, "disabled")
    print(f"  Debug mode is {status}. Use /debug enable or /debug disable.")
    return current


def _handle_status(services: Services) -> None:
    """Show current configuration and service status."""
    print(_c(BOLD + CYAN, "Status:"))
    print(f"  {_c(DIM, 'Config file:')} {config_file()}")
    print(f"  {_c(DIM, 'Storage:')} {storage_dir()}")
    print(f"  {_c(DIM, 'Model:')} {settings.opencode_model}")
    print(f"  {_c(DIM, 'API key:')} {'set' if settings.opencode_api_key else _c(RED, 'not set')}")
    print(f"  {_c(DIM, 'Search API:')} {'set' if settings.lang_search_api else _c(YELLOW, 'not set')}")
    print(f"  {_c(DIM, 'Ollama:')} {settings.ollama_url}")
    print(f"  {_c(DIM, 'SMTP:')} {'configured' if services.email.configured else _c(YELLOW, 'not configured')}")

    docs = services.documents.list_documents()
    print(f"  {_c(DIM, 'Documents:')} {len(docs)} indexed")

    try:
        memories = services.memory.search("test", k=1)
        print(f"  {_c(DIM, 'Memory service:')} {_c(GREEN, 'connected')}")
    except Exception:
        print(f"  {_c(DIM, 'Memory service:')} {_c(RED, 'error')}")


def _handle_command(user_input: str, services: Services, debug: bool) -> bool:
    command, _, argument = user_input.partition(" ")
    argument = argument.strip().strip('"')
    if command == "/add":
        _handle_add(services.documents, argument)
    elif command == "/list":
        _handle_list(services.documents)
    elif command == "/help":
        _print_help()
    elif command == "/debug":
        debug = _handle_debug(argument, debug)
    elif command == "/setup":
        run_setup()
    elif command == "/status":
        _handle_status(services)
    else:
        print(f"  {_c(RED, 'Unknown command:')} {command}. Type /help for available commands.")
    return debug


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="airesearch",
        description="Multi-agent CLI personal assistant with web search, document memory, and automation.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--config-dir",
        action="store_true",
        help="Print the user data directory and exit",
    )
    args = parser.parse_args()

    if args.config_dir:
        print(storage_dir().parent)
        return

    _setup_console()
    _setup_history()
    _print_banner()

    services = create_services()

    if not settings.opencode_api_key:
        print(f"  {_c(YELLOW, 'No API key configured.')} Run {_c(BOLD, '/setup')} to configure your settings.\n")

    try:
        llm = ChatOpenCode(
            api_key=settings.opencode_api_key,
            model_name=settings.opencode_model,
        )
        graph = create_assistant(llm, services)
    except Exception as exc:
        print(f"  {_c(RED, 'Failed to initialize assistant:')} {exc}")
        print(f"  {_c(DIM, 'Run /setup to configure your API key.')}")
        graph = None

    history: list[BaseMessage] = []
    debug = False
    print(f"  {_c(GREEN, 'Ready.')}\n")

    while True:
        try:
            user_input = input(_c(BOLD + BLUE, "You") + " > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        _save_history()

        if not user_input:
            continue
        if user_input.startswith("/"):
            debug = _handle_command(user_input, services, debug)
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print(_c(DIM, "Goodbye."))
            break

        if graph is None:
            print(f"  {_c(RED, 'Assistant not initialized.')} Run {_c(BOLD, '/setup')} first.\n")
            continue

        history.append(HumanMessage(content=user_input))

        if debug:
            answer = _run(graph, history, debug=True)
        else:
            stop = threading.Event()
            spinner = threading.Thread(target=_spin, args=(stop, "Thinking"), daemon=True)
            spinner.start()
            try:
                answer = _run(graph, history, debug=False)
            finally:
                stop.set()
                spinner.join()

        if answer is None:
            print(f"\n  {_c(RED, 'No response produced.')}\n")
            continue

        history.append(answer)
        text = _plain_text(_to_text(answer.content))
        print(f"\n  {_c(BOLD + GREEN, 'Assistant:')} {text}\n")


if __name__ == "__main__":
    main()
