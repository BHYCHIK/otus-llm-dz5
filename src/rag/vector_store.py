from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

import os

from qdrant_client.models import HnswConfigDiff, VectorParams


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

    def find_splits(self, query: str, limit: int=100, categories: list[str] = None):
        return self._vector_store.similarity_search_with_score(query, limit)

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

    def find_splits(self, query: str, limit: int=100, categories: list[str] = None):
        if categories is None:
            return super().find_splits(query=query, limit=limit)

        return self._vector_store.similarity_search_with_score(query=query, k=limit, filter={'metadata.loaded_category': {'$in': categories}})


class QdrantStore(VectorStore):
    def _get_distance(self):
        if self._space == 'cosine':
            return models.Distance.COSINE
        elif self._space == 'euclid':
            return models.Distance.EUCLID
        elif self._space == 'dot':
            return models.Distance.DOT
        elif self._space == 'manhattan':
            return models.Distance.MANHATTAN
        else:
            raise Exception("Space must be one of 'cosine', 'euclid', 'dot', 'manhattan'")

    def __init__(self, embedder, space='cosine', construction_ef=100, M=16, search_ef=10, need_setup=False):

        super().__init__(embedder=embedder, space=space, construction_ef=construction_ef, M=M, search_ef=search_ef)

        self._client = QdrantClient(url=os.environ['QDRANT_URL'])
        self._collection_name = f'arxiv_{space}_{construction_ef}_{M}_{search_ef}'

        if need_setup:
            self.setup_collection()

        self._vector_store = QdrantVectorStore(
            collection_name=self._collection_name,
            distance=self._get_distance(),
            client=self._client,
            embedding=self.get_embedder()
        )

    def find_splits(self, query: str, limit: int=100, categories: list[str] = None):
        if categories is None:
            return super().find_splits(query=query, limit=limit)

        return self._vector_store.similarity_search_with_score(query=query, k=limit,
                                                               filter={
                                                                   'must': [
                                                                       {
                                                                           'key': 'metadata.loaded_category',
                                                                           'match': {'any': categories},
                                                                       },
                                                                   ],
                                                                }
                                                               )

    def setup_index(self):
        self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name='metadata.category',
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True
        )

    def setup_collection(self):
        if self._client.collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)

        vector_dim = self.get_embedder()._client.get_sentence_embedding_dimension()

        print(f"Prepare collection {self._collection_name} with size {vector_dim}")

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=self._get_distance()
            )
        )

        self.setup_index()

        self._client.update_collection(
            collection_name=self._collection_name,
            hnsw_config=HnswConfigDiff(
                m=self._M,
                ef_construct=self._construction_ef,
            )
        )

        self._client.update_collection(
            collection_name=self._collection_name,
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )
        )

