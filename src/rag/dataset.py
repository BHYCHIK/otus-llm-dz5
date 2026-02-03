import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

class SubsetMetadata():
    _metadata = {}

    def __init__(self, path):
        metas = json.load(open(path + os.sep + 'meta.json', 'r'))
        self._metadata = {}
        for m in metas:
            self._metadata[m["id"]] = {
                'title': m["title"],
                'link': m["link"],
                'category': m["category"],
                'all_authors': m["authors"],
            }

    def get_metadata_of_doc(self, id):
        return self._metadata[id]

    def get_all_metadata(self):
        return self._metadata

class ArxivDataset:
    _path: str = '../data'
    _docs: list[Document] = []
    _docs_splits: list[Document] = []

    def __init__(self, path='../data'):
        self._path = path

    def _load_directory(self, subdir):
        print("loading directory {}".format(subdir))
        dir_metadata = SubsetMetadata(subdir)
        print(dict(dir_metadata.get_all_metadata()))

        loader = PyPDFDirectoryLoader(subdir, recursive=True)
        self._docs.extend(loader.load())
        print("Now loaded {} documents".format(len(self._docs)))
        return self

    def load(self):
        dirs = os.listdir(self._path)
        for dir in dirs:
            self._load_directory(self._path + os.sep + dir)
            break #TODO: delete
        print("loaded {} documents".format(len(self._docs)))
        return self

    def split(self, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._docs_splits =text_splitter.split_documents(self._docs)
        print("split {} documents".format(len(self._docs_splits)))

        sources = {d.metadata.get("source") for d in self._docs}
        print("unique sources:", len(sources))
        print(list(sorted(sources))[:10])

        return self

    def clean_docs(self):
        self._docs.clear()
        return self

    def clean_splits(self):
        self._docs_splits.clear()
        return self

    def clean(self):
        self.clean_docs()
        self.clean_splits()
        return self

    def get_splits(self):
        return self._docs_splits
