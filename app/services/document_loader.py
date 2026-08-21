from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DEFAULT_DOC_PATH = Path(__file__).resolve().parent.parent.parent / "data"

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class DocManager:
    def __init__(self):
        pass

    @staticmethod
    def load_file(file: str | Path) -> list[Document]:
        """Load a single file into documents, validating its extension."""
        file = Path(file)
        suffix = file.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file type '{suffix or file.name}'. Supported types: {supported}"
            )

        if suffix == ".pdf":
            documents = []
            reader = PdfReader(str(file))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": str(file), "page": page_num},
                        )
                    )
            return documents

        text = file.read_text(encoding="utf-8")
        return [Document(page_content=text, metadata={"source": str(file)})]

    @staticmethod
    def load_documents(path: Optional[str] = DEFAULT_DOC_PATH) -> list[Document]:
        documents = []
        data_path = Path(path)

        for file in data_path.iterdir():
            if file.suffix.lower() in SUPPORTED_EXTENSIONS:
                documents.extend(DocManager.load_file(file))

        return documents

    @staticmethod
    def split_documents(documents: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        return splitter.split_documents(documents)
