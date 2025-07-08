import streamlit as st
import requests
import pandas as pd
import streamlit.components.v1 as components

# --- FastAPI Backend URL ---
API_URL = "http://localhost:8000/query"  # Adjust if hosted remotely
MAP_URL = "http://localhost:8000/static/map.html"  # Must match FastAPI static path

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Multi-Agent RAG Chatbot", layout="wide")

st.title("🧠 Multi-Agent RAG Chatbot")
st.markdown("Ask Logisense:")
st.markdown("- `Top reasons for delay`")
st.markdown("- `Plot map for high driver fatigue`")
st.markdown("- `Show top 3 high risk shipments`")


# --- User Input ---
user_input = st.chat_input("Type your query here...")

# --- Process Query ---
if user_input:
    with st.spinner("Thinking... 🤔"):
        try:
            response = requests.post(API_URL, json={"query": user_input})
            response.raise_for_status()
            result = response.json()

            st.session_state.chat_history.append({
                "user": user_input,
                "response": result["response"],
                "records": result.get("top_records", []),
                "map": result.get("map_generated", False)
            })

        except Exception as e:
            st.error(f"❌ Error from backend: {e}")

# --- Display Chat History ---
for i, msg in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(msg["user"])

    with st.chat_message("assistant"):
        st.markdown(f"📌 **Insight:**\n\n{msg['response']}")

        if msg["records"]:
            st.markdown("📄 **Supporting Records:**")
            df = pd.DataFrame(msg["records"])
            st.dataframe(df, use_container_width=True)

        if msg["map"]:
            st.markdown("🗺️ **Map Preview:**")
            try:
                components.html(f'<iframe src="{MAP_URL}" width="100%" height="500" style="border:none;"></iframe>', height=510)
                st.markdown(f"[🌐 Open full map]({MAP_URL})", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"⚠️ Unable to load map: {e}")
