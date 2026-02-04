from langchain_chroma import Chroma
from langchain_qdrant import Qdrant

class VectorStore:
    def __init__(self, embedder, space='cosine', construction_ef=100, M=16, search_ef=10):
        self._embedder = embedder
        self._space = space
        self._construction_ef = construction_ef
        self._M = M
        self._search_ef = search_ef

    def get_embedder(self):
        return self._embedder

    def store_splits(self, splits):
        self._vector_store.add_documents(splits)

    def find_splits(self, query: str, limit: int=100):
        return self._vector_store.similarity_search_with_score(query, limit)

#TODO add configuration of HSNW and make it as name to dir
class ChromaStore(VectorStore):
    def __init__(self, embedder, space='cosine', construction_ef=100, M=16, search_ef=10):
        super().__init__(embedder=embedder, space='cosine', construction_ef=100, M=16, search_ef=10)
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