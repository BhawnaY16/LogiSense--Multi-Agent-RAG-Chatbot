import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ✅ Use OpenRouter key (set in .env)
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",  # or another supported by OpenRouter
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# 🧠 Reasoning Prompt
reasoning_prompt = PromptTemplate.from_template("""
You are a reasoning expert analyzing logistics documents.
Given the user query and the following summaries, generate a brief, concise answer using the most relevant information.

User Query: {query}

Relevant Document Summaries:
{documents}

Your Answer:
""")

def analyze_documents(query: str, documents: list) -> str:
    """
    Accepts a list of document strings or dicts and returns an LLM-generated summary.
    """

    # Normalize: extract string if document is a dict with 'document' key
    normalized_docs = []
    for doc in documents:
        if isinstance(doc, dict) and "document" in doc:
            normalized_docs.append(str(doc["document"]))
        else:
            normalized_docs.append(str(doc))

    joined_docs = "\n".join(normalized_docs[:5])  # Limit to top 5 for reasoning
    response = llm.invoke(reasoning_prompt.format(query=query, documents=joined_docs))
    return response.content.strip()
