"""Interactive CLI for the personal AI assistant."""

import argparse
import importlib.metadata
import logging
from typing import Optional

import httpx

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage

try:
    __version__ = importlib.metadata.version("airi")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.2.0"

logger = logging.getLogger(__name__)

from app.agents.automation_agent import AutomationAgent
from app.agents.user_context_agent import UserContextAgent
from app.agents.web_search_agent import WebSearchAgent
from app.chat_opencode import ChatOpenCode
from app.cli.input import CancelledInput
from app.cli.prompt import create_session, read_prompt
from app.cli.setup import run_setup
from app.cli import ui
from app.config import settings
from app.graph.workflow import build_assistant_graph
from app.services.embedding import Embeder
from app.services.vector_store import VectorStore
from app.services.document_service import DocumentService
from app.services.email_service import EmailService, SmtpConfig
from app.services.memory_service import MemoryService
from app.services.user_profile_service import UserProfileService
from app.tools.web_search_client import WebSearchClient
from app.user_data import config_file, storage_dir


_MODELS_URL = "https://opencode.ai/zen/go/v1/models"


def _fetch_models() -> list[str] | None:
    """Fetch available model IDs from the OpenCode API."""
    try:
        resp = httpx.get(_MODELS_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]
    except Exception:
        return None


def _test_model(api_key: str, model_name: str) -> tuple[bool, str]:
    """Send a minimal test call to verify a model works."""
    try:
        llm = ChatOpenCode(
            api_key=api_key,
            model_name=model_name,
            temperature=0,
            max_tokens=10,
        )
        response = llm.invoke([HumanMessage(content="Say OK")])
        return True, response.content.strip()[:50]
    except Exception as exc:
        return False, str(exc)[:150]


def _handle_model(
    services: Services,
    current_model: str,
) -> str | None:
    """Handle /model command. Returns the new model name or None if unchanged."""
    api_key = settings.opencode_api_key
    if not api_key:
        ui.warn("Set an API key first via /setup.")
        return None

    ui.console.print("[bold cyan]Available Models[/bold cyan]")

    models = _fetch_models()
    if models:
        ui.info(f"Fetched {len(models)} models from OpenCode.")
        table = ui.Table(box=None, show_header=False, padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(style="dim")
        table.add_column(style="green")
        for index, model_id in enumerate(models, 1):
            marker = "● current" if model_id == current_model else ""
            table.add_row(str(index), model_id, marker)
        ui.console.print(table)
        ui.console.print()
        ui.console.print("[dim]Enter a number or model name (empty to cancel).[/dim]")
    else:
        ui.warn("Could not fetch model list from OpenCode.")
        ui.console.print("[dim]Enter a model name manually, or press Enter to cancel.[/dim]")
        ui.console.print(f"[dim]Current: [bold]{current_model}[/bold][/dim]")

    try:
        choice = input("  Model> ").strip()
    except (EOFError, KeyboardInterrupt):
        ui.console.print()
        return None

    if not choice:
        ui.info("Model unchanged.")
        return None

    if choice.isdigit() and models:
        index = int(choice) - 1
        if 0 <= index < len(models):
            model_id = models[index]
        else:
            ui.error(f"Invalid number. Choose 1-{len(models)}.")
            return None
    else:
        model_id = choice

    if model_id == current_model:
        ui.info(f"Already using [bold]{model_id}[/bold].")
        return None

    ui.console.print(f"\n  [dim]Testing [bold]{model_id}[/bold]...[/dim]")
    ok, msg = _test_model(api_key, model_id)
    if ok:
        ui.success(f"[bold]{model_id}[/bold] is working. ({msg})")
        return model_id
    ui.error(f"[bold]{model_id}[/bold] failed: {msg}")
    ui.console.print("[dim]Switching cancelled.[/dim]")
    return None


def _truncate(text: str, limit: int) -> str:
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


class Services:
    """Composition root for the assistant's services."""

    def __init__(
        self,
        memory: MemoryService,
        documents: DocumentService,
        email: EmailService,
        profile: UserProfileService,
    ) -> None:
        self.memory = memory
        self.documents = documents
        self.email = email
        self.profile = profile


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
        WebSearchAgent(llm, WebSearchClient(), services.memory, services.profile),
        UserContextAgent(llm, services.memory, services.documents, services.profile),
        AutomationAgent(llm, services.email, services.memory, services.profile),
    ]
    return build_assistant_graph(llm, agents)


class DebugCallbackHandler(BaseCallbackHandler):
    """Prints tool calls and their results as they happen."""

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        name = serialized.get("name", "unknown")
        ui.print_debug(
            "tool",
            f"{name}({_truncate(str(input_str), 100)})",
            detail_style="cyan",
        )

    def on_tool_end(self, output, **kwargs) -> None:
        content = getattr(output, "content", output)
        ui.print_debug("result", f"<- {_truncate(str(content), 100)}")

    def on_tool_error(self, error, **kwargs) -> None:
        ui.print_debug("error", _truncate(str(error), 100), label_style="red")


def _run(graph, history: list[BaseMessage], debug: bool):
    """Run the graph once. With debug, trace routing and tool activity."""
    try:
        if not debug:
            result = graph.invoke({"messages": history})
            return result["messages"][-1]

        final_answer: Optional[BaseMessage] = None
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
                        ui.print_debug(
                            "route",
                            f"supervisor -> {nxt}",
                            detail_style="magenta",
                        )
                elif isinstance(value, dict) and value.get("messages"):
                    final_answer = value["messages"][-1]
        return final_answer
    except Exception as exc:
        logger.debug("Graph invocation failed: %s", exc, exc_info=True)
        ui.console.print()
        ui.error("A temporary error occurred. Please try again.")
        return None


def _handle_add(document_service: DocumentService, raw_path: str) -> None:
    if not raw_path:
        ui.warn("Usage: /add <document_path>")
        return
    try:
        info = document_service.add_document(raw_path)
    except (FileNotFoundError, ValueError) as exc:
        ui.error(f"Could not add document: {ui.escape(str(exc))}")
        return
    ui.success(f"Indexed '{ui.escape(info.name)}' ({info.chunks} chunks).")


def _handle_list(document_service: DocumentService) -> None:
    documents = document_service.list_documents()
    if not documents:
        ui.info("No documents indexed yet. Use /add <document_path>.")
        return
    for doc in documents:
        added = doc.added_at or "unknown date"
        ui.console.print(
            f"  [bold]{ui.escape(doc.name)}[/bold]  "
            f"[dim]({doc.chunks} chunks, added {ui.escape(added)})[/dim]"
        )


def _handle_debug(argument: str, current: bool) -> bool:
    state = argument.lower()
    if state in {"enable", "on", "1"}:
        ui.success("Debug mode enabled — routing and tool activity will be shown.")
        return True
    if state in {"disable", "off", "0"}:
        ui.warn("Debug mode disabled.")
        return False
    status = "[green]enabled[/green]" if current else "[yellow]disabled[/yellow]"
    ui.console.print(f"  Debug mode is {status}. Use /debug enable or /debug disable.")
    return current


def _handle_status(services: Services) -> None:
    ui.console.print("[bold cyan]Status:[/bold cyan]")
    ui.console.print(f"  [dim]Config file:[/dim] {ui.escape(str(config_file()))}")
    ui.console.print(f"  [dim]Storage:[/dim] {ui.escape(str(storage_dir()))}")
    ui.console.print(f"  [dim]Model:[/dim] {ui.escape(settings.opencode_model)}")
    api_key = "set" if settings.opencode_api_key else "[red]not set[/red]"
    ui.console.print(f"  [dim]API key:[/dim] {api_key}")
    search_api = "set" if settings.lang_search_api else "[yellow]not set[/yellow]"
    ui.console.print(f"  [dim]Search API:[/dim] {search_api}")
    ui.console.print(f"  [dim]Ollama:[/dim] {ui.escape(settings.ollama_url)}")
    smtp = "configured" if services.email.configured else "[yellow]not configured[/yellow]"
    ui.console.print(f"  [dim]SMTP:[/dim] {smtp}")

    docs = services.documents.list_documents()
    ui.console.print(f"  [dim]Documents:[/dim] {len(docs)} indexed")

    try:
        services.memory.search("test", k=1)
        ui.console.print("  [dim]Memory service:[/dim] [green]connected[/green]")
    except Exception:
        ui.console.print("  [dim]Memory service:[/dim] [red]error[/red]")


def _handle_command(user_input: str, services: Services, debug: bool) -> bool:
    command, _, argument = user_input.partition(" ")
    argument = argument.strip().strip('"')
    if command == "/add":
        _handle_add(services.documents, argument)
    elif command == "/list":
        _handle_list(services.documents)
    elif command == "/help":
        ui.print_help()
    elif command == "/debug":
        debug = _handle_debug(argument, debug)
    elif command == "/setup":
        run_setup()
    elif command == "/status":
        _handle_status(services)
    elif command == "/clear":
        ui.clear_screen()
        ui.print_banner(__version__)
        ui.success("Chat history preserved. Ready.")
    else:
        ui.error(f"Unknown command: {ui.escape(command)}. Type /help for available commands.")
    return debug


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="airi",
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

    ui.print_banner(__version__)

    services = create_services()

    if not settings.opencode_api_key:
        ui.warn("No API key configured. Run [bold]/setup[/bold] to configure your settings.")
        ui.console.print()

    try:
        llm = ChatOpenCode(
            api_key=settings.opencode_api_key,
            model_name=settings.opencode_model,
        )
        graph = create_assistant(llm, services)
        current_model: str = settings.opencode_model
    except Exception as exc:
        ui.error(f"Failed to initialize assistant: {ui.escape(str(exc))}")
        ui.info("Run /setup to configure your API key.")
        graph = None
        current_model: str = settings.opencode_model or "kimi-k2.6"

    history: list[BaseMessage] = []
    debug = False
    session = create_session()
    ui.success("Ready.")
    ui.console.print()

    while True:
        try:
            user_input = read_prompt(session, model=current_model).strip()
        except EOFError:
            ui.console.print()
            break
        except CancelledInput:
            ui.warn("Prompt cancelled.")
            continue
        except KeyboardInterrupt:
            ui.warn("Prompt cancelled.")
            continue

        if not user_input:
            continue
        if user_input == "/model":
            new_model = _handle_model(services, current_model)
            if new_model and new_model != current_model and settings.opencode_api_key:
                current_model = new_model
                try:
                    llm = ChatOpenCode(
                        api_key=settings.opencode_api_key,
                        model_name=current_model,
                    )
                    graph = create_assistant(llm, services)
                    ui.success(f"Switched to [bold]{current_model}[/bold].")
                except Exception as exc:
                    ui.error(f"Failed to create assistant: {ui.escape(str(exc))}")
                    graph = None
            continue
        if user_input.startswith("/"):
            debug = _handle_command(user_input, services, debug)
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            ui.console.print("[dim]Goodbye.[/dim]")
            break

        if graph is None:
            ui.error("Assistant not initialized. Run [bold]/setup[/bold] first.")
            ui.console.print()
            continue

        history.append(HumanMessage(content=user_input))

        try:
            if debug:
                answer = _run(graph, history, debug=True)
            else:
                with ui.status("Thinking"):
                    answer = _run(graph, history, debug=False)
        except KeyboardInterrupt:
            ui.console.print()
            ui.warn("Task cancelled.")
            continue

        if answer is None:
            ui.console.print()
            ui.error("No response produced.")
            ui.console.print()
            continue

        history.append(answer)
        ui.print_assistant(_to_text(answer.content))


if __name__ == "__main__":
    main()
