import re
import pandas as pd
from graph import supply_chain_graph
from agents.spatial_agent import plot_delay_clusters
from session_context import load_session, save_session

print("✅ Available collections: ['telemetry_docs']")
print("🧠 Multi-Agent RAG Chatbot Ready")
print("💬 Type your query (or 'exit' to quit)")

# Load session if any
session = load_session()

while True:
    user_input = input("\n🧑 You: ").strip()
    if user_input.lower() in ("exit", "quit"):
        print("👋 Exiting chatbot.")
        break

    # ===== Check for record-related query =====
    record_match = re.search(r"\b(top|show|give)\b.*\b(record|row|shipment|entry)\b", user_input.lower())
    number_match = re.search(r"\btop\s*(\d+)", user_input.lower())
    top_n = int(number_match.group(1)) if number_match else 5

    if record_match:
        last_docs = session.get("last_documents", [])
        if last_docs:
            # ✅ Context exists — treat as follow-up
            df = pd.DataFrame(last_docs[:top_n])
            if df.empty:
                print("⚠️ No records found in previous response.")
            else:
                print(f"\n📄 Top {top_n} Supporting Records:\n")
                print(df.to_markdown(index=False))
                plot_delay_clusters(last_docs[:top_n])
            continue
        else:
            # ❌ No context — treat as new primary query
            user_input = f"show top {top_n} records"

    # ===== Normal LangGraph execution =====
    input_state = {"query": user_input}
    try:
        result = supply_chain_graph.invoke(input_state)
    except Exception as e:
        print(f"❌ LangGraph error: {e}")
        continue

    # 📝 Output response
    response = result.get("response")
    if response:
        print("\n✅ Final Response:\n" + response)

    # 💾 Save documents in session
    docs = result.get("documents", [])
    rows = [doc["metadata"]["row"] for doc in docs if isinstance(doc, dict) and "metadata" in doc and "row" in doc["metadata"]]
    save_session(user_input, rows)

    # 🗺️ Plot map if prompt requests it
    if "map" in user_input.lower() or "plot" in user_input.lower():
        if docs:
            try:
                plot_delay_clusters([doc["summary"] for doc in docs])
            except Exception as e:
                print(f"⚠️ Map generation error: {e}")
