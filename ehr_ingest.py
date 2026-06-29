import glob

from langchain_text_splitters import RecursiveCharacterTextSplitter

from create_documents import load_pdf_pages

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)


# Convert PDFs into text chunks.
def generate_chunks():
    all_chunks = []

    for file_path in glob.glob("data/**/*.pdf", recursive=True):
        docs = load_pdf_pages(file_path)
        chunks = text_splitter.split_documents(docs)
        all_chunks.extend(chunks)

    return all_chunks
