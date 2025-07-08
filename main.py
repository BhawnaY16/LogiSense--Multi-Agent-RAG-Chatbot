from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any

from graph import supply_chain_graph
from session_context import save_session
from agents.spatial_agent import plot_delay_clusters

app = FastAPI()

# Serve static files (map.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local frontend or Streamlit access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response schemas
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    top_records: List[Dict[str, Any]]
    map_url: str | None

# API route
@app.post("/query", response_model=QueryResponse)
def process_query(req: QueryRequest):
    query = req.query
    result = supply_chain_graph.invoke({"query": query})

    response = result.get("response", "")
    documents = result.get("documents", [])
    rows = [doc["metadata"]["row"] for doc in documents if "metadata" in doc and "row" in doc["metadata"]]

    save_session(query, rows)

    # Check if map should be generated
    map_url = None
    if "map" in query.lower() or "plot" in query.lower():
        success = plot_delay_clusters([doc["summary"] for doc in documents], output_path="static/map.html")
        if success:
            map_url = "http://localhost:8000/static/map.html"

    return QueryResponse(response=response, top_records=rows, map_url=map_url)
