from langgraph.graph import StateGraph, END
from agents.router_agent import classify_query
from agents.retriever_agent import retrieve_documents
from agents.reasoning_agent import analyze_documents
from agents.response_agent import format_response

from typing import TypedDict, List, Dict, Any

# ✅ Define the state used across the graph
class SupplyChainState(TypedDict, total=False):
    query: str
    route: str
    documents: List[Dict[str, Any]]
    summary: str
    summaries: List[str]  # ✅ add this line
    response: str


# 🧭 Step 1: Route tool
def route_tool(state: SupplyChainState) -> Dict[str, str]:
    route = classify_query(state["query"])
    return {"route": route}

# 📄 Step 2: Retrieve documents with summary and metadata
def retrieve_tool(state: SupplyChainState) -> Dict[str, Any]:
    documents = retrieve_documents(state["query"])
    return {"documents": documents}

# 🧠 Step 3: Analyze summaries (pass only summary to LLM)
def reasoning_tool(state: SupplyChainState) -> Dict[str, Any]:
    # Extract the summaries from the top documents
    summaries = [doc["summary"] for doc in state["documents"]]
    summary = analyze_documents(state["query"], state["documents"])
    return {
        "summary": summary,
        "summaries": summaries  # ✅ add this for spatial plotting
    }


# 🗣️ Step 4: Format final response
def response_tool(state: SupplyChainState) -> Dict[str, str]:
    response = format_response(state["summary"])
    return {"response": response}

# ✅ Build LangGraph flow
graph = StateGraph(SupplyChainState)

graph.add_node("route", route_tool)
graph.add_node("documents", retrieve_tool)
graph.add_node("summary", reasoning_tool)
graph.add_node("response", response_tool)

graph.set_entry_point("route")
graph.add_edge("route", "documents")
graph.add_edge("documents", "summary")
graph.add_edge("summary", "response")
graph.add_edge("response", END)

# ✅ Compile graph
supply_chain_graph = graph.compile()
