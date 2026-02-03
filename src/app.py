from fastapi import FastAPI
from rag.embedder import Embedder
from sentence_transformers import util
from rag.dataset import ArxivDataset
from rag.vector_store import ChromaStore

app = FastAPI()

embedder = Embedder(model='BAAI/bge-m3')
store = ChromaStore(embedder=embedder.get_model())

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/test")
def test_endpoint():
    embedding1= embedder.get_embedding("Пес ходила на лужайке")
    embedding2 = embedder.get_embedding("Собака бегала на газоне")
    embedding3 = embedder.get_embedding("Василий бухал на лужайке")
    cosim1 = util.cos_sim(embedding1, embedding2)
    cosim2 = util.cos_sim(embedding1, embedding3)
    print(cosim1)
    print(cosim2)
    return {"status": "ok"}