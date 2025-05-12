# Store embeddings in the Pinecone vector store
from langchain_pinecone import PineconeVectorStore
import os
from pinecone import  Pinecone, ServerlessSpec
from dotenv import load_dotenv
def embed_chunks(chunks,embedder):
    load_dotenv()    
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = "customerbot2" 

# Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # Huggingface embeddings = 384
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Create vectorstore
    for i in range(0, len(chunks), 100):  # upload 100 at a time
        vectorstore= PineconeVectorStore.from_documents(
            chunks[i:i+100],
            embedding=embedder,
            index_name=index_name
    )
    
    return vectorstore
