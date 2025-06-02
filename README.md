# Customer Care Bot

A Retrieval-Augmented Generation (RAG) chatbot for answering customer queries about Tesla vehicles using PDF owner manuals as a knowledge base. Powered by modern NLP: Hugging Face embeddings, Pinecone vector search, LangChain, and Streamlit.

---

## Features

- **PDF-based knowledge ingestion** (chunking & embedding)
- **Hybrid retrieval** (semantic & keyword search with BM25)
- **RAG question-answering with LLM**
- **Streamlit web interface**
- **Automated evaluation** (retrieval & generation metrics)
- **Dockerized deployment**

---

## Project Structure

customer_care_bot/
│
├── src/
│ ├── utils/
│ │ ├── chunking.py
│ │ ├── data_loading.py
│ │ ├── embedding.py
│ │ ├── generation.py
│ │ ├── retrieval_augment.py
│ │ └── vector_store.py
│ ├── main.py
│ ├── app.py
│ └── tests/
│ ├── retriever_evaluation.py
│ └── generation_evaluation.py
│
├── knowledgebase/
│ ├── model_S_owners_manual.pdf
│ └── model_X_owners_manual.pdf
│
├── ground_truth.json
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/customer_care_bot.git
cd customer_care_bot
2. Create & Activate a Virtual Environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
3. Install Python Requirements
pip install -r requirements.txt
4. Add Knowledge Base PDFs
Put your Tesla manuals in the knowledgebase/ folder.
Example: model_S_owners_manual.pdf, model_X_owners_manual.pdf.

Running the App
Locally (Streamlit)
streamlit run src/app.py
Then open http://localhost:8501 in your browser.

Docker

Build the Docker image:
docker-compose build
Run the container:
docker-compose up
Go to http://localhost:8501

If you change source code, rebuild with docker-compose build!

Evaluation
Run these from your project root:

Retrieval Metrics (Precision, Recall, MRR):
python src/tests/retriever_evaluation.py

Generation Metrics (BLEU, ROUGE-L):
python src/tests/generation_evaluation.py

Environment Variables
Store API keys in a .env file (auto-loaded):

PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
Troubleshooting
NLTK or package import errors:
Activate your .venv before running/installing.

PDF not found:
Check filenames and location in knowledgebase/.



Credits
Built with LangChain, Pinecone, HuggingFace Transformers, Streamlit, NLTK, and rouge-score.

License
MIT License


