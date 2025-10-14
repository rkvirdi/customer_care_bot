
from src.utils.embedding import embedding_model
from src.utils.retrieval_augment import retrieve_similar
from src.utils.generation import generate_answer

def run_pipeline(q: str, k: int = 5):
    model = embedding_model()
    top_chunks = retrieve_similar(model, q, k=k)
    answer, context = generate_answer(q, top_chunks)
    return answer, context

if __name__ == "__main__":
    q = "What should you do if a charge port light turns red during charging on Model X?"
    ans, ctx = run_pipeline(q)
    print("answer:", ans)
