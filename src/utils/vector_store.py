import os
import psycopg2
import numpy as np
import json
PG_CONN_INFO = {
    "dbname":   "vectordb",   # or your chosen database name
    "user":     "postgres",
    "password": "password",   # must match POSTGRES_PASSWORD from Docker
    #"host":     "172.17.0.2",
    "host":     "localhost",
    "port":     5432
    }

def embed_chunks(docs, embedder):
    
    conn = psycopg2.connect(**PG_CONN_INFO)
    cur = conn.cursor()
    for doc in docs:
        embedding = embedder.embed_query(doc.page_content)
        # If embedding is a numpy array, convert to list
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        cur.execute(
            """
            INSERT INTO documents (content, embedding, metadata)
            VALUES (%s, %s, %s)
            """,
            (doc.page_content, embedding, json.dumps(getattr(doc, 'metadata', {})))
        )
    conn.commit()
    cur.close()
    conn.close()

# Usage:
# insert_documents(chunked_docs, embedder)















# # Store embeddings in the Pinecone vector store
# from langchain_pinecone import PineconeVectorStore
# import os
# from pinecone import  Pinecone, ServerlessSpec
# from dotenv import load_dotenv
# def embed_chunks(chunks,embedder):
#     load_dotenv()    
#     pinecone_api_key = os.getenv("PINECONE_API_KEY")
#     index_name = "customerbot2" 

# # Initialize Pinecone
#     pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#     if index_name not in pc.list_indexes().names():
#         pc.create_index(
#             name=index_name,
#             dimension=384,  # Huggingface embeddings = 384
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )

# #Create vectorstore
#     for i in range(0, len(chunks), 100):  # upload 100 at a time
#         vectorstore= PineconeVectorStore.from_documents(
#             chunks[i:i+100],
#             embedding=embedder,
#             index_name=index_name
#     )
    
#     return vectorstore
