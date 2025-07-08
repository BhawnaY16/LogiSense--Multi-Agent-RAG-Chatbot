# session_context.py

last_query = None
last_documents = []

def save_session(query, documents):
    global last_query, last_documents
    last_query = query
    last_documents = documents

def load_session():
    return {"last_query": last_query, "last_documents": last_documents}

