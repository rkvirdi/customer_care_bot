# Customer Care Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for answering questions based on Tesla Model S and Model X owner’s manuals.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [Usage](#usage)

   * [Command-Line Interface](#command-line-interface)
   * [Streamlit Web App](#streamlit-web-app)
7. [Docker Deployment](#docker-deployment)
8. [Configuration](#configuration)
9. [Contributing](#contributing)

---

## Project Overview

This project implements a RAG pipeline to:

1. **Load** PDF manuals from a `knowledgebase/` directory.
2. **Chunk** the text into smaller segments.
3. **Embed** each chunk into vector representations.
4. **Store** and search embeddings in a vector store.
5. **Retrieve** relevant chunks for a user query.
6. **Generate** a final answer using an LLM based on retrieved context.

It provides:

* A **CLI** interface (`src/main.py`).
* A **Streamlit** UI (`src/app.py`).
* **Docker** and **Docker Compose** setup for containerization.

---

## Features

* **PDF Data Loading**: Automatically loads all PDFs in `knowledgebase/`.
* **Chunking**: Splits text into overlapping chunks.
* **Embeddings**: Configurable embedding model.
* **Vector Store**: Efficient similarity search.
* **Retrieval**: Gets top-k relevant chunks for a query.
* **Generation**: Uses an LLM to craft answers from context.
* **Dual Interfaces**: CLI and Streamlit web app.
* **Dockerized**: Easy deployment via Docker.

---

## Prerequisites

* **Python** 3.11+
* **pip**
* **Docker** & **Docker Compose** (for containerized deployment)

---

## Installation

1. **Clone the repository**

   ```bash
   git clone <REPO_URL>
   cd customer_care_chatbot
   ```

2. **(Optional) Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\\Scripts\\activate   # Windows
   ```

3. **Install Python dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Project Structure

```plaintext
customer_care_chatbot/
├── knowledgebase/                # PDF manuals
│   ├── model_S_owners_manual.pdf
│   └── model_X_owners_manual.pdf
├── src/
│   ├── main.py                   # CLI entrypoint
│   ├── app.py                    # Streamlit UI
│   └── utils/                    # Utility modules
│       ├── data_loading.py
│       ├── chunking.py
│       ├── embedding.py
│       ├── vector_store.py
│       ├── retrieval_augment.py
│       └── generation.py
├── requirements.txt              # Python libraries
├── Dockerfile                    # Docker image recipe
├── docker-compose.yml            # Compose setup
└── README.md                     # This document
```

---

## Usage

### Command-Line Interface

Run the pipeline from the CLI:

```bash
cd src
python main.py
```

This loads manuals, processes a default query, and prints the answer.

### Streamlit Web App

1. **Run** the Streamlit app:

   ```bash
   cd src
   streamlit run app.py
   ```
2. **Open** your browser at [http://localhost:8501](http://localhost:8501).

Use the input box to type a question, then click **Ask** to get an answer and view context chunks.

---

## Docker Deployment

1. **Build and start** services:

   ```bash
   docker-compose up --build
   ```
2. **Access** the Streamlit UI:
   Navigate to [http://localhost:8501](http://localhost:8501).

Docker Compose maps port **8501** by default.

---

## Configuration

* **Knowledgebase Path**: By default, PDF manuals are read from `knowledgebase/` next to `src/`. To change, update `KB_PATH` in `.env` or modify the path resolution in code (`app.py` and `main.py`).
* **Environment Variables**: Create a `.env` file in the project root for custom settings.

---

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/xyz`)
3. Commit your changes (`git commit -m "Add feature xyz"`)
4. Push (`git push origin feature/xyz`)
5. Open a Pull Request

---


