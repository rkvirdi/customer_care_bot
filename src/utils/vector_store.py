import os
import psycopg2
import numpy as np
import json
import uuid
from typing import Iterable

PG_CONN_INFO = {
    "dbname":   "vectordb",   # or your chosen database name
    "user":     "postgres",
    "password": "password",   # must match POSTGRES_PASSWORD from Docker
    #"host":     "172.17.0.2",
    "host":     "localhost",
    "port":     5432
    }

# Global connection object
_conn = None

def get_db_connection():
    """Get or create a database connection.
    Returns the same connection if already exists."""
    global _conn
    try:
        if _conn is None or _conn.closed:
            print("Establishing new database connection...")
            _conn = psycopg2.connect(**PG_CONN_INFO)
            print("Database connection successful!")
        return _conn
    except psycopg2.Error as e:
        print(f"Database connection error: {str(e)}")
        raise

def close_db_connection():
    """Close the database connection explicitly when needed."""
    global _conn
    if _conn is not None and not _conn.closed:
        _conn.close()
        _conn = None

def setup_database():
    """Create or update the database schema with all required columns.
    Safe to run multiple times - will only add missing columns."""
    
    print("Setting up database...")
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create the extension if it doesn't exist
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Create table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_embeddings (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding vector(384),
            doc_id UUID,
            source TEXT,
            source_id TEXT,
            page INTEGER,
            chunk_index INTEGER,
            text_length INTEGER,
            embedding_model TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create indexes
    index_commands = [
        "CREATE INDEX IF NOT EXISTS idx_document_embeddings_doc_id ON document_embeddings(doc_id);",
        "CREATE INDEX IF NOT EXISTS idx_document_embeddings_source ON document_embeddings(source);",
        "CREATE INDEX IF NOT EXISTS idx_document_embeddings_page ON document_embeddings(page);",
        "CREATE INDEX IF NOT EXISTS idx_document_embeddings_embedding ON document_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
    ]
    
    for cmd in index_commands:
        try:
            cur.execute(cmd)
        except psycopg2.Error as e:
            print(f"Note: Index creation skipped: {str(e)}")
            # Continue even if index creation fails (might already exist)
            conn.rollback()
    
    conn.commit()
    cur.close()
    # Connection is kept open for reuse

def embed_chunks(docs, embedder):
    """Embed and insert chunk documents into Postgres.

    This enriches metadata with recommended fields while remaining
    backwards-compatible when `doc.metadata` isn't provided."""
    
    print(f"Embedding and inserting {len(docs)} documents...")

    conn = get_db_connection()
    cur = conn.cursor()
    for idx, doc in enumerate(docs):
        if idx > 0 and idx % 10 == 0:
            print(f"Progress: processed {idx}/{len(docs)} documents")
        
        # generate embedding
        try:
            embedding = embedder.embed_query(doc.page_content)
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding for document {idx}: {str(e)}")
            raise

        # Start with any existing metadata on the document
        base_meta = dict(getattr(doc, 'metadata', {}) or {})

        # Extract metadata values, using None if not present
        doc_id = base_meta.get("doc_id") or str(uuid.uuid4())
        source = base_meta.get("source") or getattr(doc, 'source', None) or base_meta.get('filename')
        source_id = base_meta.get("source_id")
        page = base_meta.get("page") or getattr(doc, 'page', None)
        # Convert page to int if possible
        try:
            page = int(page) if page is not None else None
        except (ValueError, TypeError):
            page = None
        chunk_index = base_meta.get("chunk_index") or base_meta.get("chunk") or idx
        text_length = base_meta.get("text_length") or len(doc.page_content or "")
        embedding_model = base_meta.get("embedding_model") or getattr(embedder, 'model_name', None)

        cur.execute(
            """
            INSERT INTO document_embeddings (
                content, 
                embedding, 
                doc_id,
                source,
                source_id,
                page,
                chunk_index,
                text_length,
                embedding_model
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                doc.page_content, 
                embedding, 
                doc_id,
                source,
                source_id,
                page,
                chunk_index,
                text_length,
                embedding_model
            )
        )

    conn.commit()
    cur.close()
    # Connection is kept open for reuse

def verify_database():
    """Verify database connection and table setup."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'document_embeddings'
            );
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            print("❌ Table 'document_embeddings' does not exist!")
            return False
            
        # Count rows
        cur.execute("SELECT COUNT(*) FROM document_embeddings;")
        row_count = cur.fetchone()[0]
        print(f"✅ Table exists with {row_count} documents")
        
        # Sample a row
        if row_count > 0:
            cur.execute("""
                SELECT id, doc_id, source, page 
                FROM document_embeddings 
                LIMIT 1;
            """)
            sample = cur.fetchone()
            print(f"Sample row - ID: {sample[0]}, Doc ID: {sample[1]}, Source: {sample[2]}, Page: {sample[3]}")
        
        cur.close()
        return True
    except Exception as e:
        print(f"❌ Database verification failed: {str(e)}")
        return False

# Usage:
# setup_database()  # Set up schema
# embed_chunks(chunked_docs, embedder)  # Insert documents
# verify_database()  # Verify setup and count documents





#PINECONE
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
