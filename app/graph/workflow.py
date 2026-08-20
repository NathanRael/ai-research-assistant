from typing import Annotated, Callable, Sequence, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agents.base_agent import BaseAgent
from app.agents.supervisor_agent import GENERAL_CHOICE, SupervisorAgent


class AssistantState(TypedDict):
    """State shared across the graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    next: str


def build_assistant_graph(llm: BaseChatModel, agents: Sequence[BaseAgent]):
    """Compile the supervisor graph: START -> supervisor -> one specialist -> END.

    New specialists only need to be added to the agents list.
    """
    supervisor = SupervisorAgent(llm, agents)
    compiled_agents = {agent.name: agent.build() for agent in agents}

    def supervisor_node(state: AssistantState) -> dict:
        return {"next": supervisor.route(state["messages"])}

    def general_node(state: AssistantState) -> dict:
        return {"messages": [supervisor.answer_general(state["messages"])]}

    def make_agent_node(agent_name: str) -> Callable[[AssistantState], dict]:
        compiled = compiled_agents[agent_name]

        def node(state: AssistantState, config: RunnableConfig) -> dict:
            result = compiled.invoke({"messages": state["messages"]}, config=config)
            return {"messages": [result["messages"][-1]]}

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
