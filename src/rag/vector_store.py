from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore
from qdrant_client import qdrant_client, models

class VectorStore:
    def __init__(self, embedder, space='cosine', construction_ef=100, M=16, search_ef=10):
        self._embedder = embedder
        self._space = space
        self._construction_ef = construction_ef
        self._M = M
        self._search_ef = search_ef
        self._vector_store = None

    def get_embedder(self):
        return self._embedder

    def store_splits(self, splits):
        self._vector_store.add_documents(splits)

    def find_splits(self, query: str, limit: int=100):
        return self._vector_store.similarity_search_with_score(query, limit)

class ChromaStore(VectorStore):
    def __init__(self, embedder, space='cosine', construction_ef=100, M=16, search_ef=10):
        super().__init__(embedder=embedder, space=space, construction_ef=construction_ef, M=M, search_ef=search_ef)
        self._vector_store = Chroma(collection_name='arxiv',
                                    persist_directory=f'../chroma_{space}_{construction_ef}_{M}_{search_ef}/',
                                    embedding_function=self.get_embedder(),
                                    collection_metadata={
                                        'hnsw:space': self._space,
                                        'hnsw:construction_ef': self._construction_ef,
                                        'hnsw:M': self._M,
                                        'hnsw:search_ef': self._search_ef
                                    })

    def get_retriever(self, limit: int=100, fetch_limit: int=100, search_type: str='similarity'):
        search_kwargs = {
            'k': limit,
        }
        if search_type == 'mmr':
            search_kwargs['fetch_k'] = fetch_limit

        return self._vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

class QdrantStore(VectorStore):
    def __init__(self, embedder, space='cosine', construction_ef=100, M=16, search_ef=10):

        if space == 'cosine':
            local_space = models.Distance.COSINE
        elif space == 'euclid':
            local_space = models.Distance.EUCLID
        elif space == 'dot':
            local_space = models.Distance.DOT
        elif space == 'manhattan':
            local_space = models.Distance.MANHATTAN
        else:
            raise Exception("Space must be one of 'cosine', 'euclid', 'dot', 'manhattan'")

        super().__init__(embedder=embedder, space=space, construction_ef=construction_ef, M=M, search_ef=search_ef)

        self._client = qdrant_client.QdrantClient()
        self._collection_name = f'arxiv_{space}_{construction_ef}_{M}_{search_ef}'
        self._vector_store = QdrantVectorStore(
            collection_name=self._collection_name,
            distance=local_space,
            client=self._client,
        )

    def setup_collection(self):
        return