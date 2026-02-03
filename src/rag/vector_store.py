from langchain_chroma import Chroma

class VectorStore:
    def __init__(self, embedder):
        self._embedder = embedder

    def get_embedder(self):
        return self._embedder


class ChromaStore(VectorStore):
    def __init__(self, embedder):
        super().__init__(embedder=embedder)
        self._vector_store = Chroma(collection_name='arxiv',
                                    persist_directory='../chroma/',
                                    embedding_function=self.get_embedder())

    def store_splits(self, splits):
        self._vector_store.add_documents(splits)