from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DEFAULT_DOC_PATH = Path(__file__).resolve().parent.parent.parent / "data"

class DocManager:
    def __init__(self):
        pass

    @staticmethod
    def load_documents(path: Optional[str] = DEFAULT_DOC_PATH) -> list[Document]:
        documents = []
        data_path = Path(path)

        for file in data_path.iterdir():
            if file.suffix.lower() == ".pdf":
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

            elif file.suffix.lower() == ".txt":
                text = file.read_text(encoding="utf-8")
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(file)},
                    )
                )

        return documents


    @staticmethod
    def split_documents(documents: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        return splitter.split_documents(documents)