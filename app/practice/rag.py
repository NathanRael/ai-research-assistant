from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from app.chat_opencode import OpenCodeModel, ChatOpenCode
from app.config import settings
from app.practice.vector_store import VectorStore
from deep_translator import GoogleTranslator

PROMPT = """
You are a document question-answering assistant.

Your job is to answer questions about the provided documents.

Rules:
- Answer on the user language (ex : in french, english,...)
- If no relevant information found, just reply that you do not have information


Answer using bullets for explanation and raw text format (DO NOT USE ANY MD FORMAT)
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
    contexts = VectorStore().search(query, top_k=5)
    if not contexts:
        return "No relevant documents found in the knowledge base"
    contexts_string = ""

    for context in contexts:
        metadata = context.metadata
        contexts_string += f"\n\n# Source : {metadata.get('source', 'unknown')}, page : {metadata.get('page', 'unknown')},Content : {context.page_content}"
    return contexts_string


class RagPipeline:
    def __init__(self, model: OpenCodeModel = OpenCodeModel.KIMI_K26):
        model = ChatOpenCode(api_key=settings.opencode_api_key, model_name=model)
        self.agent = create_agent(
            model=model,
            system_prompt=PROMPT,
            tools=[search_information],
        )

    def query(self, query: str):
        translated = GoogleTranslator(source="auto", target="fr").translate(query)
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=translated)]}
        )
        messages = result.get("messages", [])[-1].content
        return messages


if __name__ == "__main__":
    rag = RagPipeline()
    result = rag.query("What's the role of the contractor ?")
    print(result)