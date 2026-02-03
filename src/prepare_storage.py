from rag.embedder import Embedder
from rag.dataset import ArxivDataset
from rag.vector_store import ChromaStore
import time

embedder = Embedder(model='BAAI/bge-m3')
store = ChromaStore(embedder=embedder.get_model())

ITER_SIZE = 50

def generate_embeddings():
    print("generate embeddings")

    arxiv_dataset = ArxivDataset('../data').load().split()
    splits = arxiv_dataset.get_splits()

    splits_size = len(splits)
    i = 0
    while i < splits_size:
        i += ITER_SIZE
        start = time.time()
        store.store_splits(splits[i:min(i+ITER_SIZE, splits_size)])
        end = time.time()
        print(f"Vectorized {i}. Chunk of {ITER_SIZE} vectorized for {end-start} seconds")
    arxiv_dataset.clean()
    print("done")

if __name__ == "__main__":
    generate_embeddings()