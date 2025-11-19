# 🚀 Speculative RAG

A production-ready **Retrieval-Augmented Generation (RAG)** system with **Speculative Decoding** powered by Groq API. Get faster, more accurate responses without downloading massive language models!

## ✨ Features

- 🔥 **No Model Downloads** - Uses Groq's ultra-fast API (300+ tokens/sec)
- ⚡ **Speculative Decoding** - Draft model generates quickly, target model verifies for quality
- 🎯 **Semantic Search** - FAISS vector store for efficient document retrieval
- 🆓 **Free Tier Available** - Generous free tier on Groq API
- 📝 **Simple Setup** - 5 minutes from clone to running

## 🎯 What is Speculative Decoding?

Speculative decoding is an optimization technique that uses two models:
1. **Draft Model** (Llama-3.1-8B) - Fast, generates candidate responses
2. **Target Model** (Llama-3.3-70B) - Accurate, verifies and refines

This gives you near-large-model quality at small-model speeds! ⚡

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Free Groq API Key

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up (free tier available)
3. Copy your API key

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Add Your Documents

Place your `.txt` files in `data/raw/`:

```bash
data/raw/document1.txt
data/raw/document2.txt
```

### 5. Run

```bash
python main.py
```

## 📁 Project Structure

```
speculative-rag/
│
├── data/
│   └── raw/                 # Place your .txt files here
│
├── vector_store/            # Auto-generated FAISS index
│   ├── index.faiss
│   └── embeddings.npy
│
├── models/
│   ├── draft_model.py       # Fast Llama-3.1-8B via Groq
│   └── target_model.py      # Accurate Llama-3.3-70B via Groq
│
├── src/
│   ├── chunker.py           # Document chunking
│   ├── embedder.py          # Sentence embeddings
│   ├── retriever.py         # Vector search
│   ├── rag_prompt.py        # Prompt formatting
│   ├── speculative_decoder.py  # Speculative decoding logic
│   └── pipeline.py          # Main RAG pipeline
│
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
└── .env                     # Your API keys (not in git)
```

## 🔧 How It Works

```mermaid
graph LR
    A[User Query] --> B[Retrieve Docs]
    B --> C[Build Prompt]
    C --> D[Draft Model]
    D --> E[Target Model]
    E --> F[Verified Answer]
```

1. **Document Chunking** - Splits documents into semantic chunks
2. **Vector Embedding** - Creates embeddings using sentence-transformers
3. **FAISS Indexing** - Builds fast similarity search index
4. **Query Processing** - Retrieves relevant chunks for user query
5. **Speculative Decoding**:
   - Draft model generates quick response
   - Target model verifies and refines
6. **Return Answer** - High-quality, contextually accurate response

## 🎮 Usage

### Basic Usage

```python
from src.pipeline import SpeculativeRAG
from src.retriever import Retriever

retriever = Retriever()
pipeline = SpeculativeRAG(retriever)

query = "What are the main leadership principles?"
answer = pipeline.run(query, chunks)
print(answer)
```

### Customize Models

Edit `models/draft_model.py` or `models/target_model.py`:

```python
class DraftModel:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        # Change to any Groq model
        ...
```

Available Groq models:
- `llama-3.1-8b-instant` (fastest)
- `llama-3.3-70b-versatile` (most accurate)
- `mixtral-8x7b-32768` (long context)

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Speed** | ~300 tokens/sec (Groq API) |
| **Latency** | ~500ms first token |
| **Accuracy** | 70B-class quality |
| **Cost** | Free tier: 14,400 req/day |

## 🛠️ Configuration

### Token Limits

Adjust in `models/target_model.py`:

```python
max_tokens=1024  # Increase for longer responses
temperature=0.3  # 0.0-1.0 (lower = more deterministic)
```

### Chunk Size

Adjust in `src/chunker.py`:

```python
chunk_text(text, chunk_size=500, overlap=50)
```

## 🐛 Troubleshooting

### "GROQ_API_KEY not found"
- Make sure `.env` file exists in project root
- Check API key is correctly set: `GROQ_API_KEY=gsk_...`

### "No text files found"
- Add `.txt` files to `data/raw/`
- Check file permissions

### Incomplete answers
- Increase `max_tokens` in model files (currently 1024)

## 📚 Dependencies

- **groq** - Groq API client
- **faiss-cpu** - Vector similarity search
- **sentence-transformers** - Text embeddings
- **numpy** - Numerical operations
- **python-dotenv** - Environment management

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

MIT License - feel free to use in your projects!

## 🔗 Resources

- [Groq Documentation](https://console.groq.com/docs)
- [FAISS Documentation](https://faiss.ai/)
- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)

## 💡 Tips

- **Start small** - Test with 1-2 documents first
- **Monitor usage** - Check Groq dashboard for API limits
- **Experiment** - Try different model combinations
- **Cache results** - Vector store is reused across runs

---

Built with ❤️ using Groq API and FAISS