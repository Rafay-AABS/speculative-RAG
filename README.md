# speculative-RAG


speculative-rag/
│
├── data/
│   ├── raw/                 # raw PDFs, txt, etc. (optional)
│   └── processed/           # processed .txt or chunked files
│
├── vector_store/
│   ├── index.faiss          # FAISS index written automatically
│   └── embeddings.npy
│
├── models/
│   ├── draft_model.py
│   └── target_model.py
│
├── src/
│   ├── chunker.py
│   ├── embedder.py
│   ├── retriever.py
│   ├── rag_prompt.py
│   ├── speculative_decoder.py
│   └── pipeline.py
│
├── main.py
└── requirements.txt