from dotenv import load_dotenv
load_dotenv()
import os
import psycopg2
import psycopg2.extras
from sentence_transformers import CrossEncoder

import numpy as np
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
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

    # FIXES:
    # 1) Read from the correct table: document_embeddings  
    # 2) Use cosine operator <=> to match your vector_cosine_ops index
    # 3) Build metadata dict from existing columns 
    cur.execute(
        """
        SELECT
            id,
            content,
            doc_id,
            source,
            source_id,
            page,
            chunk_index,
            text_length,
            embedding_model
        FROM document_embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_emb, top_k)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Convert to langchain Document objects with reconstructed metadata
    docs = []
    for row in rows:
        meta = {
            "db_id": row["id"],
            "doc_id": str(row["doc_id"]) if row["doc_id"] is not None else None,
            "source": row["source"],
            "source_id": row["source_id"],
            "page": row["page"],
            "chunk_index": row["chunk_index"],
            "text_length": row["text_length"],
            "embedding_model": row["embedding_model"],
        }
        docs.append(Document(page_content=row["content"], metadata=meta))
    return docs


def load_docs_for_bm25():
    """Fetch all stored chunks as LangChain Documents for BM25."""
    conn = psycopg2.connect(**PG_CONN_INFO)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
        SELECT
            id, content, doc_id, source, source_id, page, chunk_index, text_length, embedding_model
        FROM document_embeddings
        ORDER BY doc_id, page, chunk_index
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()

    docs = []
    for r in rows:
        meta = {
            "db_id": r["id"],
            "doc_id": str(r["doc_id"]) if r["doc_id"] else None,
            "source": r["source"],
            "source_id": r["source_id"],
            "page": r["page"],
            "chunk_index": r["chunk_index"],
            "text_length": r["text_length"],
            "embedding_model": r["embedding_model"],
        }
        docs.append(Document(page_content=r["content"], metadata=meta))
    return docs


def retrieve_similar(embedder, query, k):
    """
    docs: all loaded Document objects (for BM25)
    embedder: embedding model
    query: user query string
    k: top-k
    """
    # Semantic retrieval (from pgvector/Postgres)
    sem_docs = pgvector_semantic_search(query, embedder, top_k=k)
    # Keyword retrieval (BM25 over all docs from pgvector/Postgres)
    docs = load_docs_for_bm25()
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
    hybrid_docs = unique

    ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  
    pairs = [(query, d.page_content) for d in hybrid_docs]            
    scores = ce_model.predict(pairs)                                  
    hybrid_docs = [doc for _, doc in sorted(zip(scores, hybrid_docs), 
                                        key=lambda x: x[0],
                                        reverse=True)][:k]
    # Prompt to send to the LLM
    prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, stick to the context provided and say "I don't know the answer".

Question: {question}

Context: {context}

Answer:
"""
    prompt_template = ChatPromptTemplate.from_template(prompt)

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
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



# """
# retrieval_augment.py
# --------------------
# Hybrid retriever for RAG evaluation (LangChain-Community v0.3.24).

# Combines:
# 1️⃣ Semantic retrieval (pgvector / embeddings)
# 2️⃣ Keyword retrieval (BM25)
# 3️⃣ Weighted hybrid merge
# 4️⃣ Cross-Encoder reranking (ms-marco MiniLM)
# All components are fully open-source and run locally.
# """

# from dotenv import load_dotenv
# load_dotenv()

# import os
# import psycopg2
# import psycopg2.extras
# import numpy as np
# from langchain.schema import Document
# from langchain_community.retrievers import BM25Retriever
# from sentence_transformers import CrossEncoder
# from src.utils.vector_store import PG_CONN_INFO


# # ===============================================================
# # 🔹 1. PostgreSQL pgvector semantic search
# # ===============================================================
# def pgvector_semantic_search(query, embedder, top_k: int = 10):
#     """
#     Perform semantic retrieval using pgvector embeddings.
#     Returns a list of LangChain Document objects ordered by similarity.
#     """
#     conn = psycopg2.connect(**PG_CONN_INFO)
#     cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

#     query_emb = embedder.embed_query(query)
#     if isinstance(query_emb, np.ndarray):
#         query_emb = query_emb.tolist()

#     cur.execute(
#         """
#         SELECT id, content, metadata
#         FROM documents
#         ORDER BY embedding <#> %s::vector
#         LIMIT %s
#         """,
#         (query_emb, top_k),
#     )
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()

#     docs = [Document(page_content=r["content"], metadata=r["metadata"]) for r in rows]
#     return docs


# # ===============================================================
# # 🔹 2. Weighted hybrid merge (semantic + keyword)
# # ===============================================================
# def weighted_hybrid(sem_docs, key_docs, w_sem=0.7, w_bm25=0.3):
#     """
#     Combine semantic and BM25 results with simple weighted ranking.
#     """
#     scores = {}
#     for i, d in enumerate(sem_docs):
#         scores[d.page_content] = scores.get(d.page_content, 0) + w_sem * (len(sem_docs) - i)
#     for i, d in enumerate(key_docs):
#         scores[d.page_content] = scores.get(d.page_content, 0) + w_bm25 * (len(key_docs) - i)
#     ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#     return [Document(page_content=c) for c, _ in ranked]


# # ===============================================================
# # 🔹 3. Cross-Encoder reranker
# # ===============================================================
# _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

# def rerank_with_cross_encoder(query: str, docs, top_k: int = 5):
#     """
#     Rerank candidate documents for a query using a cross-encoder.
#     """
#     if not docs:
#         return []
#     pairs = [[query, d.page_content] for d in docs]
#     scores = _reranker.predict(pairs)
#     ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
#     return [doc for doc, _ in ranked[:top_k]]


# # ===============================================================
# # 🔹 4. Main hybrid retrieval function
# # ===============================================================
# def retrieve_similar(docs, embedder, query: str, k: int = 5):
#     """
#     Run pgvector + BM25 + weighted fusion + reranking.
#     Compatible with langchain-community v0.3.24.
#     """
#     final_docs = []

#     try:
#         # --- Stage 1: Semantic search ---
#         sem_docs = pgvector_semantic_search(query, embedder, top_k=10)

#        # --- Stage 2: BM25 keyword search (stable across 0.3.x) ---


#         key_docs = []
#         try:
#             keyword_retriever = BM25Retriever.from_documents(docs, k=10)
#             key_docs = keyword_retriever.get_relevant_documents(query)
#             print(f"✅ BM25Retriever fetched {len(key_docs)} documents")
#         except Exception as e:
#             print(f"⚠️ BM25Retriever failed: {e}")
#             key_docs = []

#         # --- Stage 3: Weighted hybrid merge ---
#         hybrid_candidates = weighted_hybrid(sem_docs, key_docs, w_sem=0.7, w_bm25=0.3)

#         # --- Stage 4: Cross-encoder reranking ---
#         try:
#             final_docs = rerank_with_cross_encoder(query, hybrid_candidates, top_k=k)
#         except Exception as e:
#             print(f"⚠️ Cross-encoder reranking failed: {e}")
#             final_docs = hybrid_candidates[:k]

#     except Exception as e:
#         print(f"⚠️ Retrieval pipeline error: {e}")
#         final_docs = []

#     return final_docs


# # ===============================================================
# # 🔹 5. Optional: manual smoke test
# # ===============================================================
# if __name__ == "__main__":
#     from sentence_transformers import SentenceTransformer
#     from src.utils.data_loading import load_data

#     embedder = SentenceTransformer("all-MiniLM-L6-v2")
#     KB = "../../knowledgebase"
#     files = [f"{KB}/model_S_owners_manual.pdf", f"{KB}/model_X_owners_manual.pdf"]
#     docs = load_data(files)
#     query = "How do I open the charge port?"

#     results = retrieve_similar(docs, embedder, query, k=3)
#     for i, d in enumerate(results, 1):
#         print(f"\n[{i}] {d.page_content[:200]}...")



