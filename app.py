from fastapi import FastAPI

app = FastAPI(
    title="Speculative RAG API",
    description="Fast RAG over uploaded PDFs",
    version="1.0.0",
)

@app.get("/")
async def root():
    return {"message": "Speculative RAG API running"}