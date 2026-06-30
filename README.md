# DocuMind – AI-Powered Document Research Assistant

**DocuMind** is a production‑ready, autonomous research agent that lets you upload documents (PDF, Excel, TXT) and ask natural‑language questions about them. It uses **Google Gemini** for reasoning, **RAG (Retrieval-Augmented Generation)** for document context, and a modular tool system to search, analyze, and synthesize answers.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Google%20Gemini-886FBF?style=for-the-badge&logo=googlebard&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)

---

## Features

- **Multi‑format support** – Upload PDFs, Excel (`.xlsx`/`.xls`), and text files.
- **Smart RAG** – ChromaDB vector store with persistent disk storage for fast semantic search.
- **Modular Tools** – Automatically selects the right tool:
  - `DocumentSearchTool` – search uploaded documents.
  - `WebSearchTool` – fetch latest info via DuckDuckGo.
  - `PDFAnalyzerTool` – extract and query PDF content.
  - `ExcelAnalyzerTool` – read, summarise, and analyse spreadsheets with pandas + Gemini.
- **Gemini Integration** – Uses the new `google-genai` SDK with automatic model fallback and exponential‑backoff retry for rate limits (429/503).
- **Premium UI** – Beautiful, responsive Streamlit interface with drag‑and‑drop, chat history, and real‑time progress.
- **Persistent Storage** – Uploaded files, vector DB, and logs are all saved locally.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit (custom CSS) |
| **LLM** | Google Gemini (`google-genai` SDK) |
| **Vector DB** | ChromaDB (PersistentClient) |
| **Data Processing** | Pandas, PyPDF2 |
| **Web Search** | DuckDuckGo (`duckduckgo-search`) |
| **Language** | Python 3.10+ |

---

