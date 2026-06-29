import chromadb
from sentence_transformers import SentenceTransformer


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
