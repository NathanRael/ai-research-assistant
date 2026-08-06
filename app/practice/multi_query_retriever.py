from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
import logging

from app.practice.embedding import Embeder

documents = [
    Document(
        page_content="""
AI agents are autonomous software systems that can perceive their environment,
reason about goals, make decisions, and execute actions without constant human
intervention. Unlike traditional programs that follow fixed instructions,
AI agents can adapt their behavior based on context and feedback.

Modern AI agents often combine large language models, external tools, memory,
and planning mechanisms to accomplish complex tasks.
""",
        metadata={"type": "ai_agents_introduction"},
    ),

    Document(
        page_content="""
Large Language Model agents (LLM agents) are systems built around language
models that can understand user requests and decide which actions to perform.
They can call APIs, execute code, search databases, interact with applications,
and use external tools to complete objectives.

An LLM agent is different from a simple chatbot because it can take actions
instead of only generating text responses.
""",
        metadata={"type": "llm_agents"},
    ),

    Document(
        page_content="""
Agent memory allows AI systems to store and recall information from previous
interactions. Short-term memory maintains the current conversation context,
while long-term memory stores persistent knowledge about users, tasks, and
past experiences.

Memory mechanisms improve personalization and allow AI assistants to maintain
continuity across multiple sessions.
""",
        metadata={"type": "agent_memory"},
    ),

    Document(
        page_content="""
AI agent planning is the process of breaking complex objectives into smaller
steps and deciding the best sequence of actions. Planning techniques include
task decomposition, reasoning chains, goal management, and decision making.

Planning enables agents to solve multi-step problems instead of responding
with a single generated answer.
""",
        metadata={"type": "agent_planning"},
    ),

    Document(
        page_content="""
Tool-using AI agents extend their capabilities by interacting with external
resources. Tools can include web search engines, databases, APIs, calculators,
file systems, and software execution environments.

By selecting and calling the appropriate tools, an agent can perform tasks
that a language model alone cannot complete.
""",
        metadata={"type": "agent_tools"},
    ),

    Document(
        page_content="""
Multi-agent systems consist of multiple AI agents working together to achieve
a shared objective. Each agent may have a specialized role such as researcher,
planner, programmer, reviewer, or coordinator.

Communication between agents allows complex workflows to be distributed and
solved collaboratively.
""",
        metadata={"type": "multi_agent_systems"},
    ),

    Document(
        page_content="""
Reinforcement learning agents learn through interaction with an environment.
They improve their behavior by receiving rewards or penalties based on their
actions.

Although modern LLM agents often use language models instead of traditional
reinforcement learning algorithms, both approaches focus on autonomous
decision making and goal-oriented behavior.
""",
        metadata={"type": "reinforcement_agents"},
    ),

    Document(
        page_content="""
AI assistants powered by agent architectures can automate workflows such as
customer support, software development, data analysis, and business operations.

These systems combine reasoning capabilities, external knowledge retrieval,
memory, and action execution to complete tasks with minimal human input.
""",
        metadata={"type": "agent_applications"},
    ),
]


logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

def create_base_vector_store():
    return Chroma.from_documents(
        documents=documents,
        embedding=Embeder.get_embedding_function()
    )


def multi_query_retriever(question):
    vector_store = create_base_vector_store()
    llm = ChatOllama(model="qwen3.5:4b", temperature=0.3)

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
        llm=llm
    )


    print(f"\nOriginal query: {question}")

    generated_queries = retriever.llm_chain.invoke(
        {"question": question}
    )

    print("\nGenerated queries:")
    print(generated_queries)

    docs = retriever.invoke(question)


    print("Result")

    for i, doc in enumerate(docs[:2]):
        print(f"\n{i+1} [{doc.metadata.get("type","N/A")} ]  {doc.page_content}")



if __name__ == "__main__":
    multi_query_retriever("What are AI agents?")