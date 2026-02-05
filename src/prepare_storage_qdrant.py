import dotenv

from rag.embedder import Embedder
from rag.dataset import ArxivDataset
from rag.vector_store import QdrantStore
import time
import dotenv

print('loading dotenv')
dotenv.load_dotenv('../.env', verbose=True)

embedder = Embedder(model='BAAI/bge-m3')
store = QdrantStore(embedder=embedder.get_model(), construction_ef=4, M=2, search_ef=1, need_setup=True)
#store = QdrantStore(embedder=embedder.get_model(), construction_ef=100, M=16, search_ef=10, need_setup=True)

ITER_SIZE = 50

def generate_embeddings():
    print("Prepare storage")
    store.setup_collection()
    print("generate embeddings")

    arxiv_dataset = ArxivDataset('../data').load().split()
    splits = arxiv_dataset.get_splits()

    splits_size = len(splits)
    i = 0
    while i < splits_size:
        start = time.time()
        store.store_splits(splits[i:min(i+ITER_SIZE, splits_size)])
        end = time.time()
        print(f"Vectorized {i}. Chunk of {ITER_SIZE} vectorized for {end-start} seconds")
        i += ITER_SIZE
    arxiv_dataset.clean()
    print("done")

if __name__ == "__main__":
    generate_embeddings()