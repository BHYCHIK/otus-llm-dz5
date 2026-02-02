from fastapi import FastAPI
from rag.embedder import get_embedding
from sentence_transformers import util

app = FastAPI()

@app.post("/generate_embeddings")
def generate_embeddings():
    return {}

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/test")
def test_endpoint():
    embedding1= get_embedding("Пес ходила на лужайке")
    embedding2 = get_embedding("Собака бегала на газоне")
    embedding3 = get_embedding("Василий бухал на лужайке")
    cosim1 = util.cos_sim(embedding1, embedding2)
    cosim2 = util.cos_sim(embedding1, embedding3)
    print(cosim1)
    print(cosim2)
    return {"status": "ok"}