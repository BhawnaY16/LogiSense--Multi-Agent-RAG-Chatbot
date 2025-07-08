import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# === Config ===
DATA_PATH = "data/master.csv"
CHROMA_PATH = "/Users/shivanshusoni/Desktop/Bhawna/logisense/chroma_store"
COLLECTION_NAME = "telemetry_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# === Load Data ===
df = pd.read_csv(DATA_PATH)
assert "summary" in df.columns, "Missing 'summary' column in input file."

# === Embeddings ===
print("🧠 Generating sentence embeddings...")
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(df["summary"].tolist(), show_progress_bar=True, batch_size=32)

# === Initialize ChromaDB Persistent Client ===
client = PersistentClient(path=CHROMA_PATH)

# (Re)create collection
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    client.delete_collection(name=COLLECTION_NAME)
collection = client.create_collection(name=COLLECTION_NAME)

# === Upload in Batches ===
print("📤 Uploading to ChromaDB...")
batch_size = 5000
for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i + batch_size]
    # Add during ingestion
collection.add(
    documents=batch["summary"].tolist(),
    embeddings=embeddings[i:i + batch_size],
    ids=[str(idx) for idx in batch.index],
    metadatas=batch.to_dict(orient="records")  # includes full row
)

print(f"✅ Ingested {len(df)} records into ChromaDB.")
