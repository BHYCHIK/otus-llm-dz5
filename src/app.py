import time
import os
import dotenv

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.globals import set_debug

set_debug(False)

from rag.embedder import Embedder
from rag.vector_store import ChromaStore

from prompts.prompts import get_basic_rag_prompt

dotenv.load_dotenv('../.env')
app = FastAPI()

llm = ChatOpenAI(
    api_key=os.environ['API_KEY'],
    base_url=os.environ['API_BASE_URL'],
    temperature=0.0,
    model='qwen-3-32b'
)

embedder = Embedder(model='BAAI/bge-m3')
store_chroma_bad = ChromaStore(embedder=embedder.get_model(), construction_ef=4, M=2, search_ef=1)
store_chroma_good = ChromaStore(embedder=embedder.get_model(), construction_ef=100, M=16, search_ef=10)

def _get_vector_store():
    return store_chroma_good

def _get_bad_vector_store():
    return store_chroma_bad

app = FastAPI()

def simple_llm(query:str):
    return (ChatPromptTemplate.from_messages([HumanMessage(query)]) | llm | StrOutputParser()).invoke({})

def simple_rag(query: str, vector_storage=None):
    if vector_storage is None:
        vector_storage = _get_vector_store()
    splits = vector_storage.find_splits(query, 15)
    context = ''.join([f"<document>{doc.page_content}</document>" for (doc, score) in splits])
    return (get_basic_rag_prompt() | llm | StrOutputParser()).invoke({'context': context, 'query': query})

def hallucinations_check(query:str):
    return simple_rag(query, _get_bad_vector_store())

def simple_rag_mmr(query: str):
    retriever = _get_vector_store().get_retriever(search_type='mmr', limit=15, fetch_limit=70)
    splits = retriever.invoke(query)
    context = ''.join([f"<document>{doc.page_content}</document>" for doc in splits])
    return (get_basic_rag_prompt() | llm | StrOutputParser()).invoke({'context': context, 'query': query})

def _get_hyde_output(query:str):
    return (ChatPromptTemplate.from_messages([HumanMessage(query)])  | llm | StrOutputParser()).invoke({})


def _format_context(splits):
    context = ''.join([f"<document>{doc.page_content}</document>" for doc in splits])
    return context

def rag_with_hyde(query: str):
    retriever = _get_vector_store().get_retriever(limit=15)

    chain = (
            {
                "query": RunnablePassthrough(),
                "context": _get_hyde_output | retriever | _format_context,
            }
            | get_basic_rag_prompt()
            | llm
            | StrOutputParser()
    )
    return chain.invoke(query)

def rag_with_hyde_mmr(query: str):
    retriever = _get_vector_store().get_retriever(search_type='mmr', limit=15, fetch_limit=70)

    chain = (
            {
                "query": RunnablePassthrough(),
                "context": _get_hyde_output | retriever | _format_context,
            }
            | get_basic_rag_prompt()
            | llm
            | StrOutputParser()
    )
    return chain.invoke(query)

def naive_search_good(query:str, limit:int=5, cycles=10):
    splits = []
    start = time.time()
    for _ in range(cycles):
        splits = _get_vector_store().find_splits(query, limit)
    end = time.time()
    resp = []
    for i, (document, score) in enumerate(splits):
        print(document, score)
        resp.append({'document': document, 'score': score})
    return {'documents': resp, 'timing': end - start}

def naive_search_bad(query:str, limit:int=5, cycles=10):
    splits = []
    start = time.time()
    for _ in range(cycles):
        splits =  _get_bad_vector_store().find_splits(query, limit)
    end = time.time()
    resp = []
    for i, (document, score) in enumerate(splits):
        resp.append({'document': document, 'score': score})
    return {'documents': resp, 'timing': end - start}

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/test")
def test_endpoint():
    query = 'Which parameters help predict oil consumption?'
    return {'status': 'ok',
            'query': query,
            'chroma_naive_good': naive_search_good(query),
            'chroma_naive_bad': naive_search_bad(query),
            'simple_llm_answer': simple_llm(query),
            'simple_rag_answer': simple_rag(query),
            'simple_rag_mmr_answer': simple_rag_mmr(query),
            'rag_with_hyde_answer': rag_with_hyde(query),
            'rag_with_hyde_mmr_answer': rag_with_hyde_mmr(query),
            'hallucinations_check': hallucinations_check(query),
            }
