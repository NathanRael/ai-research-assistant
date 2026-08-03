from pprint import pprint
from typing import Optional

from langchain_core.messages import HumanMessage

from app.practice.vector_store import VectorStore
from deepagents import create_deep_agent
from langchain.tools import tool


models = {
    "qwen" : "ollama:qwen3.5:9b",
    "ornith" : "ollama:ornith:9b",
    "gemma" : "ollama:gemma4:e2b",
}

PROMPT = """
You are a document question-answering assistant.

Your job is to answer questions about the provided documents.

Rules:
1. Always use the search_information tool before answering document questions.
2. Answer the user's question directly while using his query language (ex : fr, en). Do not give unnecessary summaries.
3. Use ONLY information found in the retrieved documents.
4. Never invent information.
6. Answer in the same language as the user's question.
7. Do not ask follow-up questions or offer additional help.
8. Always include the source and page when available.

Example:

Question:
"Quel est l'objectif du contrat ?"

Good answer:
"L'objectif du contrat est la réalisation de prestations de services entre BCI France et le prestataire. (Source: contrat.pdf, page X)"

Bad answer:
"Here is a contract overview..."
"Would you like me to help with..."
"""

@tool(parse_docstring=True)
def search_information(query: str) -> str:
    """Search the knowledge base for relevant information.

       Args:
           query: Natural language search query.
           query: Natural language search query.

       Returns:
           Matching document chunks with their source metadata, or a message if
           no relevant documents are found.
       """
    contexts = VectorStore().search(query, top_k=3)
    if not contexts:
        return "No relevant documents found in the knowledge base"
    contexts_string = ""

    for context in contexts:
        metadata = context.metadata
        contexts_string += f"\n\n# Source : {metadata.get('source', 'unknown')}, page : {metadata.get('page', 'unknown')},Content : {context.page_content}"
    return contexts_string


class RagPipeline:
    def __init__(self, model : Optional[str] = models.get("qwen")):
        self.agent = create_deep_agent(
            model=model,
            system_prompt=PROMPT,
            tools=[search_information]
        )

    def query(self, query: str):
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]}
        )
        messages = result.get("messages", [])[-1].content
        return messages


if __name__ == "__main__":
    rag = RagPipeline()
    result = rag.query("Quel est l'objectif du contrat  ?")
    pprint(result)
