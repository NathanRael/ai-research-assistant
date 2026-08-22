import copy
import logging
from typing import Annotated, Callable, Sequence, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agents.base_agent import BaseAgent
from app.agents.supervisor_agent import GENERAL_CHOICE, SupervisorAgent

logger = logging.getLogger(__name__)

MAX_TOOL_CONTENT = 4000
MAX_OTHER_CONTENT = 8000
MAX_HISTORY_MESSAGES = 20
TRUNCATION_MARKER = "[...truncated...]"


class AssistantState(TypedDict):
    """State shared across the graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    next: str


def _trim_text(text: str, max_len: int) -> str:
    """Shorten text by keeping the start and end with a marker in the middle."""
    if len(text) <= max_len:
        return text
    keep = (max_len - len(TRUNCATION_MARKER)) // 2
    if keep <= 0:
        return text[:max_len]
    return text[:keep] + TRUNCATION_MARKER + text[-keep:]


def _trim_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Limit history length and shrink oversized message contents in place.

    Tool results are often the largest, so they get the smallest cap. Other
    messages are capped at a higher threshold. The original message objects are
    not mutated; trimmed copies are returned.
    """
    messages = list(messages)
    if messages and isinstance(messages[0], SystemMessage):
        kept = [copy.deepcopy(messages[0]), *[copy.deepcopy(m) for m in messages[-MAX_HISTORY_MESSAGES:]]]
    else:
        kept = [copy.deepcopy(m) for m in messages[-MAX_HISTORY_MESSAGES:]]

    for msg in kept:
        if isinstance(msg, ToolMessage):
            msg.content = _trim_text(str(msg.content), MAX_TOOL_CONTENT)
        else:
            msg.content = _trim_text(str(msg.content), MAX_OTHER_CONTENT)
    return kept


def build_assistant_graph(llm: BaseChatModel, agents: Sequence[BaseAgent]):
    """Compile the supervisor graph: START -> supervisor -> one specialist -> END.

    New specialists only need to be added to the agents list.
    """
    supervisor = SupervisorAgent(llm, agents)
    compiled_agents = {agent.name: agent.build() for agent in agents}

    def supervisor_node(state: AssistantState) -> dict:
        messages = _trim_messages(state["messages"])
        return {"next": supervisor.route(messages)}

    def general_node(state: AssistantState) -> dict:
        try:
            messages = _trim_messages(state["messages"])
            return {"messages": [supervisor.answer_general(messages)]}
        except Exception as exc:
            logger.debug("General node failed: %s", exc, exc_info=True)
            return {
                "messages": [
                    AIMessage(
                        content="A temporary failure occurred while generating a response. Please try again."
                    )
                ]
            }

    def make_agent_node(agent_name: str) -> Callable[[AssistantState], dict]:
        compiled = compiled_agents[agent_name]

        def node(state: AssistantState, config: RunnableConfig) -> dict:
            try:
                messages = _trim_messages(state["messages"])
                result = compiled.invoke({"messages": messages}, config=config)
                return {"messages": [result["messages"][-1]]}
            except Exception as exc:
                logger.debug("Agent %s failed: %s", agent_name, exc, exc_info=True)
                return {
                    "messages": [
                        AIMessage(
                            content=f"A temporary failure occurred in the {agent_name} agent. Please try again."
                        )
                    ]
                }

        return node

    graph = StateGraph(AssistantState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node(GENERAL_CHOICE, general_node)
    for name in compiled_agents:
        graph.add_node(name, make_agent_node(name))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {**{name: name for name in compiled_agents}, GENERAL_CHOICE: GENERAL_CHOICE},
    )
    graph.add_edge(GENERAL_CHOICE, END)
    for name in compiled_agents:
        graph.add_edge(name, END)

    return graph.compile()
