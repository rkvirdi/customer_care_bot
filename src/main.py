from src.utils.embedding import embedding_model
from src.utils.vector_store import retrieve_similar
from src.utils.generation import generate_answer

def run_pipeline(q: str, k: int = 5):
    model = embedding_model()
    top_chunks = retrieve_similar(model, q, k=k)
    answer, context = generate_answer(q, top_chunks)
    return answer, context

if __name__ == "__main__":
    q = "what to do if parking brake fault sign is on"
    ans, ctx = run_pipeline(q)
    print("answer:", ans)
