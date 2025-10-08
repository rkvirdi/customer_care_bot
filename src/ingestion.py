from pathlib import Path
from xml.parsers.expat import model
from src.utils.data_loading      import load_data
from src.utils.chunking          import chunk_text
from src.utils.embedding         import embedding_model
from src.utils.vector_store      import embed_chunks, setup_database, verify_database

def ingest(file_paths: list[str], chunk_size: int = 500, chunk_overlap: int = 50) -> None:
    """
    One-time (or on-demand) ingestion:
      1) Load raw docs
      2) Chunk
      3) Embed + write to your vector store (DB)
    """
    print(f"Loading documents from: {file_paths}")
    docs = load_data(file_paths)
    print(f"Loaded {len(docs)} documents")
    
    print(f"Chunking documents with size={chunk_size}, overlap={chunk_overlap}")
    chunks = chunk_text(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Created {len(chunks)} chunks")
    
    print("Loading embedding model...")
    model = embedding_model()
    
    print("Setting up database...")
    setup_database()  # Creates/updates the schema
    
    print("Verifying database setup...")
    if not verify_database():
        raise Exception("Database setup verification failed!")
    
    print("Embedding chunks and storing in database...")
    embed_chunks(chunks, model)   # persists to your store/DB
    
    print("\nVerifying document ingestion...")
    verify_database()

if __name__ == "__main__":
    # Compute project root (two levels up if this file lives in src/)
    BASE_DIR = Path(__file__).resolve().parent.parent
    KB_DIR   = BASE_DIR / "knowledgebase"

    file_paths = [
        str(KB_DIR / "model_S_owners_manual.pdf"),
        str(KB_DIR / "model_X_owners_manual.pdf"),
    ]
    print("Starting ingestion process...")
    ingest(file_paths)
    print("✅ Ingestion complete.")
