# research_agent.py Documentation

## Overview
A **Streamlit-based Deep Research Agent** that searches ArXiv for academic papers, downloads PDFs, and lets users either:
- **Quick Search (RAG)**: Ask specific questions answered via BM25 retrieval.
- **Deep Study (Map-Reduce)**: Summarize the first 10 pages using LangChain's map-reduce strategy.

---

## Core Idea
| Concept | Description |
|---------|-------------|
| **ArXiv Search** | Queries arxiv.org for papers matching a topic. |
| **PyMuPDF (fitz)** | Extracts text from downloaded PDF files. |
| **BM25Retriever** | Keyword-based retriever for fast, token-cheap Q&A. |
| **Map-Reduce Chain** | Splits text into chunks, summarizes each (map), then combines (reduce). |
| **Streamlit** | Provides the interactive web UI. |

---

## Architecture

```
┌──────────────┐      ┌─────────────┐      ┌───────────────────┐
│  User Query  │─────▶│ ArXiv API   │─────▶│ Paper Selection   │
└──────────────┘      └─────────────┘      └───────────────────┘
                                                    │
                       ┌────────────────────────────┴────────────────────────────┐
                       │                                                          │
                       ▼                                                          ▼
              ┌─────────────────┐                                      ┌─────────────────┐
              │ Quick Search    │                                      │ Deep Study      │
              │ (BM25 + LLM)    │                                      │ (Map-Reduce)    │
              └─────────────────┘                                      └─────────────────┘
                       │                                                          │
                       └──────────────────────┬───────────────────────────────────┘
                                              ▼
                                     ┌─────────────────┐
                                     │ Chat Interface  │
                                     └─────────────────┘
```

---

## Key Syntax

### 1. Search ArXiv
```python
import arxiv

def search_arxiv(query, max_results=3):
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    return [{"title": r.title, "pdf_url": r.pdf_url, ...} for r in client.results(search)]
```

### 2. Extract text from PDF (limit pages)
```python
import fitz

def download_and_extract_text(pdf_url):
    # ... download PDF ...
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc[:10]:  # First 10 pages only
        text += page.get_text()
    return text
```

### 3. Map-Reduce summarization
```python
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=200)
docs = [Document(page_content=x) for x in text_splitter.split_text(text_content)]

chain = load_summarize_chain(llm, chain_type="map_reduce")
result = chain.invoke(docs)
```

### 4. BM25 Retrieval for Q&A
```python
from langchain_community.retrievers import BM25Retriever

retriever = BM25Retriever.from_documents(docs)
retriever.k = 5
relevant_docs = retriever.invoke(user_question)
```

---

## How to Run
```bash
streamlit run project/research_agent.py
```
Enter your Groq API key in the sidebar, search for a topic, select a paper, and analyze.

---

## Dependencies
- `streamlit`
- `arxiv`
- `pymupdf` (fitz)
- `langchain-groq`
- `langchain-core`
- `langchain-text-splitters`
- `langchain-community`
- `langchain-classic`
