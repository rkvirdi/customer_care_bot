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
    top_chunks  = retrieve_similar(docs,vectorstore,user_query,k=5)
    answer, context      = generate_answer(user_query, top_chunks)
    return answer, context

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
    # q = "what to do if parking brake fault sign is on"
    # answer, context = run_pipeline(file_paths, q)
    # print(answer)

if __name__ == "__main__":
    main()
