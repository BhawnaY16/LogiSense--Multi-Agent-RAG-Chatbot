import sys
import re
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import webbrowser

from session_context import load_session, save_session
from graph import supply_chain_graph
from agents.spatial_agent import plot_delay_clusters

load_dotenv()

# === Handle input ===
if len(sys.argv) < 2:
    print("❗ Usage: python3 multi_agent_rag.py 'your query here'")
    sys.exit(1)

# Normalize quotes and trim
user_query = sys.argv[1].strip().replace("“", "\"").replace("”", "\"")
print(f"\n🧠 Multi-Agent RAG System\n🔍 User Query: {user_query}")

# === Handle Follow-up Queries ===
if re.search(r"\bsupporting\b.*\b(records|rows)?\b", user_query.lower()) or \
   re.search(r"\bshow\b.*\b(records|rows)?\b", user_query.lower()) or \
   re.search(r"\btop\b.*\b(records|rows)?\b", user_query.lower()):

    session = load_session()
    last_query = session.get("last_query")
    last_documents = session.get("last_documents", [])

    if not last_documents:
        print("⚠️ No previous query context found. Please run a primary query first.")
        sys.exit(1)

    # Extract number of records (default = 5)
    match = re.search(r"\b(\d+)\b", user_query)
    top_n = int(match.group(1)) if match else 5

    # Collect metadata rows from documents
    rows = []
    for doc in last_documents[:top_n]:
        if isinstance(doc, dict) and "metadata" in doc and "row" in doc["metadata"]:
            rows.append(doc["metadata"]["row"])

    if rows:
        df = pd.DataFrame(rows)
        print(f"\n📄 Top {len(df)} Supporting Records:\n")
        print(df.to_markdown(index=False))
    else:
        print("⚠️ Could not extract full records from previous documents.")
    sys.exit(0)

# === LangGraph Execution ===
print(f"🔍 Query: {user_query}")
results = supply_chain_graph.invoke({"query": user_query})

route = results.get("route", "unknown")
summary = results.get("summary", "")
response = results.get("response", "")
documents = results.get("documents", [])

print(f"🧭 Route: {route}")

# Save context for follow-up queries
save_session(user_query, documents)

# === Only plot map if user asks ===
if "map" in user_query.lower():
    map_output_path = Path(__file__).parent / "static" / "map.html"
    plot_delay_clusters([doc["summary"] for doc in documents], output_path=str(map_output_path))


    if map_output_path.exists():
        webbrowser.open(map_output_path.resolve().as_uri())
    else:
        print("⚠️ Map file could not be saved.")

# === Final output ===
print(f"\n✅ Final Response:\n{response}")
print("\n✅ Powered by multi-agent RAG pipeline")
