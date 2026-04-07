# NexaAI Knowledge Assistant

NexaAI is a production-oriented AI knowledge retrieval system built using a **Self-Reflective Retrieval-Augmented Generation (Self-RAG)** architecture. Unlike traditional RAG systems that blindly trust retrieved documents, NexaAI employs iterative loops to evaluate relevance, detect hallucinations, and ensure every response is grounded in verifiable evidence.

---

## 🚀 Key Features

* **Self-Reflective Loops:** Dynamically evaluates whether retrieval is necessary and filters out irrelevant context.
* **Hallucination Detection:** Built-in grounding checks to ensure the LLM doesn't "invent" information.
* **State-Based Workflows:** Powered by LangGraph to manage complex, non-linear AI decision-making.
* **End-to-End Observability:** Integrated with LangSmith for token-level tracing and debugging.
* **Iterative Query Rewriting:** Automatically refines user queries to improve retrieval quality when initial searches fail.

---

## 🧠 Architecture: Why Self-RAG?

Traditional RAG often suffers from "noise" in retrieved documents. NexaAI solves this through a sophisticated state-graph workflow:

1.  **Retrieval Decision:** Determines if external knowledge is actually required.
2.  **Relevance Evaluation:** A dedicated node filters out documents that don't match the query intent.
3.  **Query Rewriting:** If the context is insufficient, the system rewrites the query and tries again.
4.  **Grounding Check:** Validates that the generated answer is supported by the retrieved facts.
5.  **Usefulness Check:** Final validation to ensure the response solves the user's specific problem.

---

## 🛠️ Tech Stack

* **Language:** Python >= 3.11
* **Orchestration:** LangChain & LangGraph
* **Model:** Google Gemini (via `langchain-google-genai`)
* **Observability:** LangSmith
* **Vector Store:** ChromaDB
* **UI Framework:** Streamlit
* **Document Processing:** PyMuPDF (fitz)

---

## 📋 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/NexaAI-Knowledge-Assistant.git
cd NexaAI-Knowledge-Assistant
```

### 2. Install Dependencies
This project uses a `pyproject.toml` structure. You can install the required packages using pip:
```bash
pip install dotenv langchain-chroma langchain-community langchain-core langchain-google-genai langchain-text-splitters langgraph langsmith pymupdf streamlit
```

### 3. Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
GOOGLE_API_KEY=your_gemini_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=NexaAI-Self-RAG
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 📊 Observability with LangSmith

NexaAI leverages **LangSmith** to provide a "glass box" view of the AI's internal reasoning. This allows developers to:
* Debug graph execution paths (Conditional execution & Loops).
* Analyze retrieval performance and document ranking.
* Track token usage and latency for every node in the graph.
* Evaluate model behavior across different prompt iterations.

---
# 🚀 NexaAI Knowledge Assistant  
**Self-RAG Powered AI Knowledge Retrieval System**

NexaAI Knowledge Assistant is a **production-oriented AI knowledge retrieval system** built using a **Self-Reflective Retrieval-Augmented Generation (Self-RAG)** architecture.  
It delivers **grounded, verifiable, and context-aware responses** while minimizing hallucinations through iterative reasoning and validation workflows.

---

# 📌 Project Overview

Traditional RAG pipelines often retrieve documents blindly and trust retrieved content without validation, which leads to hallucinations and irrelevant responses.

**NexaAI solves this problem** using a **Self-RAG architecture** that introduces intelligent decision loops, validation nodes, and grounded answer verification.

The system uses **LangGraph-based state workflows** to create deterministic AI pipelines instead of unreliable black-box chains.

---

# 🧠 Core Architecture: Self-RAG Workflow

```text
User Query
     │
     ▼
Retrieval Decision Node
     │
     ▼
Document Retrieval
     │
     ▼
Relevance Evaluation Node
     │
     ▼
Query Rewriting Node (if needed)
     │
     ▼
Answer Generation
     │
     ▼
Grounding & Hallucination Detection
     │
     ▼
Answer Usefulness Check
     │
     ▼
Retry or Final Response

NexaAI-Knowledge-Assistant/

├── chroma_langchain_db/     # Vector database storage
├── documents/               # Source knowledge documents
├── ingestions/              # Document ingestion pipeline
├── models/                  # LLM configuration & schemas
│   ├── llm.py
│   ├── StateSchema.py
│
├── nodes/                   # Self-RAG decision nodes
├── workflow/                # LangGraph workflow logic
│
├── main.py                  # Application entry point
├── README.md
├── pyproject.toml
├── uv.lock
└── .gitignore

## 🔮 Roadmap

- [ ] **Multi-source Ingestion:** Expand beyond PDFs to include Databases and live APIs.
- [ ] **RBAC:** Implement Role-Based Access Control for enterprise security.
- [ ] **Hybrid Search:** Integrate FAISS with Chroma for optimized vector/keyword retrieval.
- [ ] **Cost-Aware Routing:** Implement logic to route queries to smaller models when high-tier LLMs aren't necessary.

---

## 💡 Engineering Takeaways

* **Deterministic Workflows:** LangGraph transforms "black-box" chains into manageable, stateful graphs.
* **Reliability:** Self-correction mechanisms significantly reduce hallucinations compared to standard RAG architectures.
* **Monitoring:** Observability isn't optional; it's the only way to safely deploy multi-step agents in production.

