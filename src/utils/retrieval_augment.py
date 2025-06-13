from dotenv import load_dotenv
load_dotenv()
import os
import psycopg2
import psycopg2.extras

import numpy as np
from langchain.schema import Document
from langchain.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from src.utils.vector_store import PG_CONN_INFO

def pgvector_semantic_search(query, embedder, top_k=5):
    """
    Returns a list of langchain Document objects from pgvector, ranked by similarity.
    """
    conn = psycopg2.connect(**PG_CONN_INFO)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    query_emb = embedder.embed_query(query)
    if isinstance(query_emb, np.ndarray):
        query_emb = query_emb.tolist()
    cur.execute(
        """
        SELECT id, content, metadata
        FROM documents
        ORDER BY embedding <#> %s::vector
        LIMIT %s
        """,
        (query_emb, top_k)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    # Convert to langchain Document objects
    docs = []
    for row in rows:
        docs.append(Document(page_content=row['content'], metadata=row['metadata']))
    return docs

def retrieve_similar(docs, embedder, query, k):
    """
    docs: all loaded Document objects (for BM25)
    embedder: embedding model
    query: user query string
    k: top-k
    """
    # Semantic retrieval (from pgvector/Postgres)
    sem_docs = pgvector_semantic_search(query, embedder, top_k=k)

    # Keyword retrieval (BM25 over all docs)
    keyword_retriever = BM25Retriever.from_documents(docs, k=k)
    key_docs = keyword_retriever.invoke(query)

    # Hybrid merge & dedupe
    all_docs = sem_docs + key_docs
    seen = set()
    unique = []
    for d in all_docs:
        # Use content as unique id if .id missing
        doc_id = getattr(d, "id", None) or hash(d.page_content)
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(d)
    hybrid_docs = unique[:k]

    # Prompt to send to the LLM
    prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, search in google.

Question: {question}

Context: {context}

Answer:
"""
    prompt_template = ChatPromptTemplate.from_template(prompt)

    llm = ChatGroq(
        model_name="llama3-70b-8192",
        streaming=True,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=lambda x: "\n\n".join(d.page_content for d in x["context"]))
        | prompt_template
        | llm
    )

    rag_chain_with_source = RunnableParallel(
        {"context": lambda q: hybrid_docs, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source
