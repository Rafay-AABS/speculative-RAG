# speculative-RAG

A Retrieval-Augmented Generation (RAG) system with speculative decoding using Groq API - **no local model downloads required!**

## Quick Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Get free Groq API key:**
   - Visit https://console.groq.com
   - Sign up and get your API key (free tier available)

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

4. **Add data:**
   - Place your .txt files in `data/raw/`

5. **Run:**
```bash
python main.py
```

## Architecture

```
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
│   ├── draft_model.py       # Fast Llama-3.1-8B via Groq API
│   └── target_model.py      # Larger Llama-3.1-70B via Groq API
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
```