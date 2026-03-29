import json
import chromadb
from chromadb.utils import embedding_functions

# ══════════════════════════════════════════════════════════════════
# RAG ENGINE — DY Patil International University Reception Bot
# Loads college_data.json into ChromaDB vector store once at startup
# Retrieves top-k relevant Q&A chunks for any incoming query
# ══════════════════════════════════════════════════════════════════

DATA_FILE       = "college_data.json"
COLLECTION_NAME = "dypiu_kb"
EMBED_MODEL     = "all-MiniLM-L6-v2"   # ~90MB, downloads once, runs fully local
TOP_K           = 3                     # number of chunks to retrieve per query

# Local sentence-transformer embedding (no internet needed after first download)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

# In-memory ChromaDB — no separate server needed
chroma_client = chromadb.Client()


def build_knowledge_base():
    """
    Reads college_data.json and loads all Q&A pairs into ChromaDB.
    Each document = "Q: <question>\nA: <answer>" for richer semantic matching.
    Call this once at server startup.
    """
    print("\n4. Building RAG Knowledge Base...")

    # Clean slate on every restart
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Combine Q+A as the document so embedding captures both context and answer
    documents = [
        f"Q: {item['question']}\nA: {item['answer']}"
        for item in data
    ]
    ids       = [str(item["id"]) for item in data]
    metadatas = [{"category": item["category"]} for item in data]

    # ChromaDB has a batch limit — add in chunks of 500
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            ids=ids[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    print(f"   -> {len(data)} entries loaded into vector store!")
    print(f"   -> Embedding model : {EMBED_MODEL}")
    print(f"   -> Retrieval top-k : {TOP_K}")
    return collection


def retrieve_context(collection, query: str, top_k: int = TOP_K) -> str:
    """
    Given a user query string, returns the top_k most semantically
    similar Q&A pairs from the knowledge base as a single string block
    ready to be injected into the Ollama prompt.
    """
    if not query or not query.strip():
        return ""

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    docs = results["documents"][0]   # list of top matching doc strings
    return "\n\n".join(docs)
