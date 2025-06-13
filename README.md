# Customer Care Bot

A Retrieval-Augmented Generation (RAG) chatbot for answering customer queries about Tesla vehicles using PDF owner manuals as a knowledge base.

* **Backend:** FastAPI + Uvicorn
* **Frontend:** React (Create React App) + Axios
* **Database:** PostgreSQL with pgvector extension
* **Dockerized:** Compose for local development and production

## Features

* PDF-based knowledge ingestion (chunking & embedding)
* Hybrid retrieval (semantic search & BM25 keyword search)
* RAG question-answering with an LLM pipeline
* React web interface for interactive Q\&A
* Automated evaluation scripts for retrieval & generation metrics
* Full Docker Compose deployment

## Project Structure

```
customer_care_bot/                  # FastAPI backend
│   ├── src/
│   │   ├── api.py            # FastAPI application
│   │   ├── main.py           # RAG pipeline runner
│   │   └── tests
        └── utils/           # ingestion, embedding, retrieval logic
├── frontend/                  # React frontend root
│   ├── my-react-app/
│   │   ├── src/              # React source files
│   │   ├── public/           # CRA public assets
│   │   ├── package.json
│   │   ├── package-lock.json
│   │   └── Dockerfile        # Frontend Dockerfile
├── docker-compose.yml        # Compose to run backend, frontend, pgvector
├── .env                      # Environment variables (API keys)
├── ground_truth.json
├── requirements.txt      # Python dependencies
├── Dockerfile            # Backend Dockerfile
└── README.md                 # This file
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/customer_care_bot.git
cd customer_care_bot
```

### 2. Backend (FastAPI)

1. Create & activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .venv\\Scripts\\activate   # Windows
   ```
2. Install dependencies:

   ```bash
   pip install -r backend/requirements.txt
   ```
3. Configure environment variables in `.env` (e.g. API keys, DB URL).
4. Run the backend server:

   ```bash
   uvicorn src.api:app --reload --port 8000
   ```
5. Visit [http://localhost:8000/docs](http://localhost:8000/docs) to explore the API.

### 3. Frontend (React)

1. Change into the React app directory:

   ```bash
   cd frontend/my-react-app
   ```
2. Install dependencies:

   ```bash
   npm install
   ```
3. Start the dev server:

   ```bash
   npm start
   ```
4. Open [http://localhost:3000](http://localhost:3000) in your browser to chat with the bot.

## Docker Compose

Run both services and a pgvector-backed Postgres in one command:

```bash
docker-compose build
docker-compose up -d
```

* **Backend:** [http://localhost:8000](http://localhost:8000)
* **Frontend:** [http://localhost:3000](http://localhost:3000)

To view logs:

```bash
docker-compose logs -f backend
docker-compose logs -f frontend
```

## Evaluation

From the project root, run retrieval and generation tests:

```bash
python src/tests/retriever_evaluation.py
python src/tests/generation_evaluation.py
```

## Environment Variables

Add a `.env` file at the root:

```dotenv
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
```

## Troubleshooting

* **CORS errors:** Ensure frontend is requesting from `http://localhost:3000` and that FastAPI’s `CORSMiddleware` allows that origin.
* **Port conflicts:** Verify ports 3000, 8000, and 5432 are free or change them in `docker-compose.yml`.
* **Missing PDFs:** Place your `*.pdf` files under `backend/knowledgebase/`.

## Credits

Built with FastAPI, React, LangChain, Pinecone, Hugging Face, pgvector & Nginx.

## License

MIT License
