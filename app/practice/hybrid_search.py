from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.practice.embedding import Embeder

documents = [
    Document(
        page_content="""
Artificial Intelligence (AI) is the field of computer science focused on creating
systems capable of performing tasks that normally require human intelligence.
These tasks include reasoning, learning, perception, language understanding,
and decision-making. AI powers applications such as recommendation systems,
voice assistants, autonomous vehicles, and fraud detection.
""",
        metadata={"type": "introduction"},
    ),

    Document(
        page_content="""
Machine Learning is a subset of AI where algorithms learn patterns from data
instead of being explicitly programmed. Common learning paradigms include
supervised learning, unsupervised learning, and reinforcement learning.
Popular algorithms include decision trees, random forests, support vector
machines, and neural networks.
""",
        metadata={"type": "machine_learning"},
    ),

    Document(
        page_content="""
Large Language Models (LLMs) are deep learning models trained on massive
collections of text. They can generate human-like responses, summarize
documents, translate languages, write code, and answer questions.
Examples include GPT, Llama, Gemini, Claude, and Mistral. These models
rely on the Transformer architecture and attention mechanisms.
""",
        metadata={"type": "llm"},
    ),

    Document(
        page_content="""
Retrieval-Augmented Generation (RAG) combines information retrieval with
language models. Instead of relying only on the model's internal knowledge,
a retriever searches a knowledge base for relevant documents. The retrieved
context is then provided to the language model, reducing hallucinations and
improving factual accuracy.
""",
        metadata={"type": "rag"},
    ),

    Document(
        page_content="""
Vector Search represents documents and queries as numerical embeddings.
Documents with similar meanings are located close together in vector space.
Similarity is typically computed using cosine similarity or dot products.
Vector databases such as Chroma, Pinecone, Weaviate, and Milvus are commonly
used for semantic search in AI applications.
""",
        metadata={"type": "vector_search"},
    ),

    Document(
        page_content="""
BM25 is a lexical ranking algorithm widely used in traditional search engines.
It scores documents according to keyword frequency, inverse document frequency,
and document length normalization. BM25 excels at retrieving documents
containing exact words or phrases, making it effective for searching product
codes, technical documentation, error messages, and legal texts. Hybrid search
often combines BM25 with vector search to leverage both exact keyword matching
and semantic understanding.
""",
        metadata={"type": "bm25"},
    ),
]

vectorstore = Chroma.from_documents(
    documents,
    embedding=Embeder().get_embedding_function(),
    collection_name="hybrid_test"
)

print("vector store ready")

vector_retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
bm25_retriever = BM25Retriever.from_documents(documents, k=3)

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

print('Hybrid retriever ready')


def test_query(question, name, retriever: BaseRetriever):
    results = retriever.invoke(question)
    print(f'\n{name} - Query : \"{question}\"')
    for i, doc in enumerate(results[:2]):
        preview = doc.page_content[:80] + "..."
        print(f'\t{i+1}. {preview}')

    return results


test_queries = [
    "What is BM25?",  # Exact keyword → BM25 = Best
    "Transformer architecture",  # Exact technical term → BM25 ≈ Vector
    "RAG",  # Exact acronym → BM25 = Best
    "How can an AI answer questions using my documents?",  # Semantic (RAG) → Vector = Best
    "Search by meaning instead of matching words",  # Semantic (Vector Search) → Vector = Best
    "How does an AI assistant generate text?",  # Synonym for LLM → Vector = Best
    "cosine similarity",  # Exact phrase → BM25 = Best
]

if __name__ == '__main__':
    for query in test_queries:
        print("=" * 60)

        # Vector only
        vector_results = test_query(query, name="VECTOR", retriever=vector_retriever)

        # BM25 only
        bm25_results = test_query(query, name="BM25", retriever=bm25_retriever)

        # Hybrid
        hybrid_results = test_query(query, name="Hybrid Search", retriever=ensemble_retriever)
