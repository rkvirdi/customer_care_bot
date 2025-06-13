from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from .main import run_pipeline
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Customer Care Chatbot API",
    description="Ask questions about your vehicle's manuals.",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] for all origins (less secure)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class QueryRequest(BaseModel):
    question: str

# Pre-compute your file paths once
BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "knowledgebase"
FILE_PATHS = [
    str(KB_DIR / "model_S_owners_manual.pdf"),
    str(KB_DIR / "model_X_owners_manual.pdf"),
]

@app.post("/ask")
async def ask(request: QueryRequest):
    # Reuse your existing pipeline
    answer, context = run_pipeline(FILE_PATHS, request.question)
    # Extract plain text from the Document objects
    context_texts = [getattr(doc, "page_content", str(doc)) for doc in context]
    return {
        "question": request.question,
        "answer": answer,
        "context": context_texts
    }
