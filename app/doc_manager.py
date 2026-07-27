from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


class DocManager:
    def __init__(self):
        pass

    @staticmethod
    def load_documents(path: str) -> list[Document]:
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