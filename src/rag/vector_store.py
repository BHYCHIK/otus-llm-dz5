from langchain_chroma import Chroma

class VectorStore:
    def __init__(self, embedder):
        self._embedder = embedder

    def get_embedder(self):
        return self._embedder

#TODO add configuration of HSNW and make it as name to dir
class ChromaStore(VectorStore):
    def __init__(self, embedder, space='cosine', construction_ef=100, M=16, search_ef=10):
        super().__init__(embedder=embedder)
        self._vector_store = Chroma(collection_name='arxiv',
                                    persist_directory=f'../chroma_{space}_{construction_ef}_{M}_{search_ef}/',
                                    embedding_function=self.get_embedder(),
                                    collection_metadata={
                                        'hnsw:space': space,
                                        'hnsw:construction_ef': construction_ef,
                                        'hnsw:M': M,
                                        'hnsw:search_ef': search_ef
                                    })

    def store_splits(self, splits):
        self._vector_store.add_documents(splits)