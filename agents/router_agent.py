# agents/router_agent.py

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# ✅ OpenRouter-compatible LLM setup
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",  # or any other OpenRouter-compatible model
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"  # 🔥 required to prevent OpenAI error
)

# 🧭 Prompt template for routing
router_prompt = PromptTemplate.from_template("""
Classify the logistics query into one of the following categories:
DELAY, FATIGUE, FUEL, WEATHER, RISK, INVENTORY, GENERAL

Query: {query}

Category:
""")

def classify_query(query: str) -> str:
    response = llm.invoke(router_prompt.format(query=query))
    return response.content.strip().upper()
