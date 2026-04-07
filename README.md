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

🔧 Tech Stack
Core Technologies
Python 3.11+ — Backend logic
LangChain — LLM orchestration
LangGraph — State-based AI workflow
LangSmith — Observability & tracing
Streamlit — Interactive UI
ChromaDB — Vector storage
PyMuPDF — Document parsing
