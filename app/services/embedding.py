from langchain_ollama import OllamaEmbeddings


class Embeder:
    def __init__(self):
        pass

    @staticmethod
    def get_embedding_function():
        return OllamaEmbeddings(model="nomic-embed-text-v2-moe:latest")
        # return OllamaEmbeddings(model="qwen3-embedding:0.6b")

    def embed_query(self, query: str):
        return self.get_embedding_function().embed_query(query)
