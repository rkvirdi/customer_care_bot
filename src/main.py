# # main.py

# # If you installed your project as a package (e.g. pip install -e .),
# # you can do: from utils import load_data, chunk_documents, …
# # Otherwise adjust PYTHONPATH or prepend src/ to sys.path.

# from utils.data_loading import load_data
# from utils.chunking import chunk_text
# from utils.embedding import embedding_model
# from utils.vector_store import embed_chunks
# from utils.retrieval_augment import retrieve_similar
# from utils.generation import generate_answer

# def main():
#    # List of manual files
#     file_paths = [
#     r"C:\Users\GursewakNeet\Documents\customer_care_chatbot\knowledgebase\model_S_owners_manual.pdf",
#     r"C:\Users\GursewakNeet\Documents\customer_care_chatbot\knowledgebase\model_X_owners_manual.pdf"
#     ]
#     docs = load_data(file_paths)

#     # 2. Break into chunks
#     chunks = chunk_text(docs, chunk_size=500, chunk_overlap=50)

#     # 3. Create and store embeddings
#     model=embedding_model()
#     vectorstore = embed_chunks(chunks,model)
#     # Query
#     # query = "how to open the door of model s"
#     # results = vectorstore.similarity_search(query)

#     # # Print top result
#     # print("Answer:", results[0].page_content)

#     # 4. Retrieve relevant context for a query
#     user_query = "what to do if parking break fault sign in on"
#     top_chunks = retrieve_similar(vectorstore)

#     # 5. Generate an answer using those chunks
#     answer = generate_answer(user_query,top_chunks)

# if __name__ == "__main__":
#     main()

# src/main.py
from pathlib import Path
from utils.data_loading      import load_data
from utils.chunking          import chunk_text
from utils.embedding         import embedding_model
from utils.vector_store      import embed_chunks
from utils.retrieval_augment import retrieve_similar
from utils.generation        import generate_answer

# 1) Define a reusable pipeline function
def run_pipeline(file_paths: list[str], user_query: str):
    docs        = load_data(file_paths)
    chunks      = chunk_text(docs, chunk_size=500, chunk_overlap=50)
    model       = embedding_model()
    vectorstore = embed_chunks(chunks, model)
    top_chunks  = retrieve_similar(vectorstore)
    answer      = generate_answer(user_query, top_chunks)
    return answer, top_chunks

# 2) Keep your CLI entrypoint
def main():
   # 1. Compute project root (two levels up if this file lives in src/)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # 2. Point at the knowledgebase folder
    KB_DIR = BASE_DIR / "knowledgebase"

    # 3. Manually list your PDFs
    file_paths = [
        str(KB_DIR / "model_S_owners_manual.pdf"),
        str(KB_DIR / "model_X_owners_manual.pdf"),
    ]
    q = "what to do if parking brake fault sign is on"
    answer, _ = run_pipeline(file_paths, q)
    print(answer)

if __name__ == "__main__":
    main()
