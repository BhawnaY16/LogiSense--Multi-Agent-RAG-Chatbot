import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# === Config ===
MASTER_PATH = "data/master.csv"         
NEW_DATA_PATH = "data/new_data.csv"     
CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "telemetry_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"



def classify(value, low=0.3, high=0.7):
    if pd.isna(value):
        return "unknown"
    if value < low:
        return "low"
    elif value < high:
        return "moderate"
    else:
        return "high"


def generate_summary(row):
    return (
        f"On {row['timestamp']}, a shipment near GPS ({row['vehicle_gps_latitude']:.2f}, {row['vehicle_gps_longitude']:.2f}) "
        f"experienced {classify(row['traffic_congestion_level'])} traffic congestion, "
        f"{classify(row['fuel_consumption_rate'])} fuel consumption, and "
        f"{classify(1 - row['fatigue_monitoring_score'])} driver fatigue. "
        f"Supplier reliability was {classify(row['supplier_reliability_score'])}, "
        f"with delay probability marked as {round(row['delay_probability'], 2)}. "
        f"Warehouse inventory was {classify(row['warehouse_inventory_level'])}, and shipping costs were {classify(row['shipping_costs'])}. "
        f"Weather conditions were {classify(row['weather_condition_severity'])}, and "
        f"customs clearance took approximately {round(row['customs_clearance_time'], 1)} hours. "
        f"Port congestion was {classify(row['port_congestion_level'])}, route risk level was {classify(row['route_risk_level'])}, and "
        f"driver behavior score was {classify(row['driver_behavior_score'])}. "
        f"The temperature was {round(row['iot_temperature'], 1)}°C and historical demand was {round(row['historical_demand'], 1)} units. "
        f"Lead time was {round(row['lead_time_days'], 1)} days and order fulfillment was {classify(row['order_fulfillment_status'])}. "
        f"The disruption likelihood was {classify(row['disruption_likelihood_score'])}. "
        f"The cargo condition was {classify(row['cargo_condition_status'])}, and handling equipment availability was {classify(row['handling_equipment_availability'])}. "
        f"Loading/unloading time was {classify(row['loading_unloading_time'])}. "
        f"Overall, the risk classification was marked as {row['risk_classification']}."
    )


def update_data():
    # Load new and master data
    new_df = pd.read_csv(NEW_DATA_PATH)
    if os.path.exists(MASTER_PATH):
        master_df = pd.read_csv(MASTER_PATH)
        combined_df = pd.concat([master_df, new_df]).drop_duplicates(subset=["timestamp"])
    else:
        combined_df = new_df

    # Generate summaries only for new data
    new_entries = combined_df[~combined_df["timestamp"].isin(master_df["timestamp"])] if 'master_df' in locals() else combined_df
    if new_entries.empty:
        print("⚠️ No new data found to ingest.")
        return

    print(f"📝 Generating summary for {len(new_entries)} new entries...")
    new_entries["summary"] = new_entries.apply(generate_summary, axis=1)

    # Update combined DataFrame
    updated_df = pd.concat([master_df, new_entries]) if 'master_df' in locals() else new_entries
    updated_df.to_csv(MASTER_PATH, index=False)
    print(f"✅ Master file updated: {MASTER_PATH}")

    # Generate embeddings
    print("🧠 Generating sentence embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(new_entries["summary"].tolist(), show_progress_bar=True, batch_size=32)

    # Initialize Chroma
    print("📦 Updating ChromaDB vector store...")
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = [f"{row['timestamp']}" for _, row in new_entries.iterrows()]
    collection.add(
        documents=new_entries["summary"].tolist(),
        embeddings=embeddings,
        ids=ids
    )
    print(f"✅ Ingested {len(new_entries)} new records into ChromaDB.")


if __name__ == "__main__":
    update_data()
