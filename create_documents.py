import pypdf
from langchain_core.documents import Document


# Load PDF as LangChain documents.
def load_pdf_pages(file_path):
    reader = pypdf.PdfReader(file_path)

    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]
