import glob

from langchain_text_splitters import RecursiveCharacterTextSplitter

from create_documents import load_pdf_pages

import chromadb
from sentence_transformers import SentenceTransformer

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



model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_chunks(chunks):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)

    return [
        {
            "text": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": embedding.tolist(),
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]


def save_to_chroma(embedded_chunks, db_path="chroma_db", collection_name="ehr_chunks"):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(collection_name)

    collection.upsert(
        ids=[str(i) for i in range(len(embedded_chunks))],
        documents=[chunk["text"] for chunk in embedded_chunks],
        metadatas=[chunk["metadata"] for chunk in embedded_chunks],
        embeddings=[chunk["embedding"] for chunk in embedded_chunks],
    )

    return collection


def search_chroma(question, db_path="chroma_db", collection_name="ehr_chunks", n_results=3):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)
    question_embedding = model.encode(question, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
    )

    return results["documents"][0]

