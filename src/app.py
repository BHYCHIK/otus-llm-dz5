from fastapi import FastAPI
from rag.embedder import Embedder
from sentence_transformers import util
from rag.vector_store import ChromaStore

app = FastAPI()

embedder = Embedder(model='BAAI/bge-m3')
#store = ChromaStore(embedder=embedder.get_model(), construction_ef=4, M=2, search_ef=1)
store = ChromaStore(embedder=embedder.get_model(), construction_ef=100, M=16, search_ef=10)

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/test")
def test_endpoint():
    splits = store.find_splits("Which parameters help predict oil consumption?", 5)
    for split in splits:
        print(split.page_content)
    return {"status": "ok"}