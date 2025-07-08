import os
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# === Config ===
CHROMA_PATH = "/Users/shivanshusoni/Desktop/Bhawna/logisense/chroma_store"
COLLECTION_NAME = "telemetry_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# === Load SentenceTransformer Model ===
model = SentenceTransformer(EMBEDDING_MODEL)

# === Connect to Persistent ChromaDB ===
client = PersistentClient(path=CHROMA_PATH)
available_collections = [c.name for c in client.list_collections()]
print("✅ Available collections:", available_collections)

# === Error if collection missing ===
if COLLECTION_NAME not in available_collections:
    raise ValueError(f"❌ Collection '{COLLECTION_NAME}' not found in ChromaDB. Found: {available_collections}")

# === Load Collection ===
collection = client.get_collection(name=COLLECTION_NAME)

# === Query Function ===
def retrieve_documents(query: str, top_k: int = 5):
    """
    Returns list of dicts:
    [
        {
            "summary": "...",              # The summary string used for reasoning
            "metadata": {
                "row": {...}               # The full original record
            }
        },
        ...
    ]
    """
    print(f"\n🔍 Query: {query}")
    query_embedding = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    # Return paired summary + metadata
    return [
        {"summary": doc, "metadata": {"row": meta}}
        for doc, meta in zip(documents, metadatas)
    ]
