from ehr_ingest import generate_chunks
from utils import embed_chunks, save_to_chroma


# Generate chunks
chunks = generate_chunks()

# Convert chunks to embeddings
embedded_chunks = embed_chunks(chunks)

# Save embedding to local ChromaDb
collection = save_to_chroma(embedded_chunks)


if __name__ == "__main__":
    print(f"Total number of chunks: {len(chunks)}")

    print(f"Total number of embedded chunks: {len(embedded_chunks)}")
    
    print(f"Saved in ChromaDB collection: {collection.name}")
