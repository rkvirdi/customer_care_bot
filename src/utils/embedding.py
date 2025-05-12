from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

def embedding_model():

# Set up the cache store
    store = LocalFileStore("./cache/")

# Initialize the Hugging Face embedding model
#def embedding_model():
    core_embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Use the cache-backed embedder with the Hugging Face model
    embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model,
        store,
        namespace=core_embeddings_model.model_name
)
    return embedder
