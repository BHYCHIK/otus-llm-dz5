import time
import os
import uuid

import dotenv

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.globals import set_debug

from langfuse import get_client, propagate_attributes

from rag.embedder import Embedder
from rag.vector_store import ChromaStore, QdrantStore

from prompts.prompts import get_basic_rag_prompt, get_query_cat_prompt

from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from pydantic import BaseModel, Field
from enum import Enum

set_debug(False)

dotenv.load_dotenv('../.env')

langfuse_client = get_client()

llm = ChatOpenAI(
    api_key=os.environ['API_KEY'],
    base_url=os.environ['API_BASE_URL'],
    temperature=0.0,
    model='qwen-3-32b'
)

embedder = Embedder(model='BAAI/bge-m3')
store_chroma_bad = ChromaStore(embedder=embedder.get_model(), construction_ef=4, M=2, search_ef=1)
store_chroma_good = ChromaStore(embedder=embedder.get_model(), construction_ef=100, M=16, search_ef=10)
store_qdrant_bad  = QdrantStore(embedder=embedder.get_model(), construction_ef=4, M=2, search_ef=1, need_setup=False)
store_qdrant_good = QdrantStore(embedder=embedder.get_model(), construction_ef=100, M=16, search_ef=10, need_setup=False)

def _get_vector_store():
    return store_chroma_good

def _get_bad_vector_store():
    return store_chroma_bad

app = FastAPI()

def simple_llm(query:str, llm_cfg):
    return (ChatPromptTemplate.from_messages([HumanMessage(query)]) | llm | StrOutputParser()).invoke({}, config=llm_cfg)

def simple_rag(query: str, llm_cfg, vector_storage=None):
    if vector_storage is None:
        vector_storage = _get_vector_store()
    splits = vector_storage.find_splits(query, 15)
    context = ''.join([f"<document>{doc.page_content}</document>" for (doc, score) in splits])
    return (get_basic_rag_prompt() | llm | StrOutputParser()).invoke({'context': context, 'query': query}, config=llm_cfg)

def hallucinations_check(query:str, llm_cfg):
    return simple_rag(query, llm_cfg, _get_bad_vector_store())

def simple_rag_mmr(query: str, llm_cfg):
    retriever = _get_vector_store().get_retriever(search_type='mmr', limit=15, fetch_limit=70)
    splits = retriever.invoke(query)
    context = ''.join([f"<document>{doc.page_content}</document>" for doc in splits])
    return (get_basic_rag_prompt() | llm | StrOutputParser()).invoke({'context': context, 'query': query}, config=llm_cfg)

def _format_context(splits):
    context = ''.join([f"<document>{doc.page_content}</document>" for doc in splits])
    return context

def rag_with_hyde(query: str, llm_cfg):
    retriever = _get_vector_store().get_retriever(limit=15)

    def _get_hyde_output(query: str):
        return (ChatPromptTemplate.from_messages([HumanMessage(query)]) | llm | StrOutputParser()).invoke({},
                                                                                                          config=llm_cfg)

    chain = (
            {
                "query": RunnablePassthrough(),
                "context": _get_hyde_output | retriever | _format_context,
            }
            | get_basic_rag_prompt()
            | llm
            | StrOutputParser()
    )
    return chain.invoke(query, config=llm_cfg)

def rag_with_hyde_mmr(query: str, llm_cfg):
    retriever = _get_vector_store().get_retriever(search_type='mmr', limit=15, fetch_limit=70)

    def _get_hyde_output(query: str):
        return (ChatPromptTemplate.from_messages([HumanMessage(query)]) | llm | StrOutputParser()).invoke({},
                                                                                                          config=llm_cfg)

    chain = (
            {
                "query": RunnablePassthrough(),
                "context": _get_hyde_output | retriever | _format_context,
            }
            | get_basic_rag_prompt()
            | llm
            | StrOutputParser()
    )
    return chain.invoke(query, config=llm_cfg)

def _naive_search(vector_store, query:str, limit:int=5, cycles=10):
    splits = []
    start = time.time()
    for _ in range(cycles):
        splits = vector_store.find_splits(query, limit)
    end = time.time()
    resp = []
    for i, (document, score) in enumerate(splits):
        resp.append({'document': document, 'score': score})
    return {'documents': resp, 'timing': end - start}

def naive_chroma_search_good(query:str, limit:int=5, cycles=10):
    return _naive_search(store_chroma_good, query, limit, cycles)

def naive_chroma_search_bad(query:str, limit:int=5, cycles=10):
    return _naive_search(store_chroma_bad, query, limit, cycles)

def naive_qdrant_search_good(query:str, limit:int=5, cycles=10):
    return _naive_search(store_qdrant_good, query, limit, cycles)

def naive_qdrant_search_bad(query:str, limit:int=5, cycles=10):
    return _naive_search(store_qdrant_bad, query, limit, cycles)

class Category(str, Enum):
    astro_ph = 'astro-ph'
    physics = 'physics'
    cond_mat = 'cond-mat'
    high_energy = 'high-energy'
    gr_qc = 'gr-qc'
    hep_ex = 'hep-ex'
    hep_ph = 'hep-ph'
    hep_lat = 'hep-lat'
    mat_ph = 'math-ph'
    econ = 'econ'
    eess = 'eess'
    stat = 'stat'
    q_fin = 'q-fin'
    math = 'math'
    quant_ph = 'quant-ph'
    nlin = 'nlin'
    nucl_ex = 'nucl-ex'
    nucl_th = 'nucl-th'
    cs = 'cs'

class SearchCategories(BaseModel):
    categories: list[Category] = Field(description='Categories of query, which fit the most')

def rag_with_hybrid_search(query: str, llm_cfg):
    parser = PydanticOutputParser(pydantic_object=SearchCategories)

    categories = (get_query_cat_prompt() | llm | parser).invoke({
            'query': query,
            'format_instruction': parser.get_format_instructions()
        },
        config=llm_cfg)

    categories = [str(c).split('.')[1] for c in categories.categories]

    print(categories)
    split = _get_bad_vector_store().find_splits(query, limit=5, categories=categories)

    return split


@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/test")
def test_endpoint():
    session_id = uuid.uuid4().hex
    llm_cfg: RunnableConfig = {
        'configurable': {'thread_id': session_id},
        'callbacks': [LangfuseCallbackHandler()],
    }
    query = 'Which parameters help predict oil consumption?'
    #query = 'Which star is the closest to Earth?'
    with langfuse_client.start_as_current_observation(as_type='span', name='langchain_call'):
        with propagate_attributes(session_id=session_id):
            return {'status': 'ok',
                    'query': query,
                    'chroma_naive_good': naive_chroma_search_good(query),
                    'chroma_naive_bad': naive_chroma_search_bad(query),
                    'qdrant_naive_good': naive_qdrant_search_good(query),
                    'qdrant_naive_bad': naive_qdrant_search_bad(query),
                    #'simple_llm_answer': simple_llm(query, llm_cfg),
                    #'simple_rag_answer': simple_rag(query, llm_cfg),
                    #'simple_rag_mmr_answer': simple_rag_mmr(query, llm_cfg),
                    #'rag_with_hyde_answer': rag_with_hyde(query, llm_cfg),
                    #'rag_with_hyde_mmr_answer': rag_with_hyde_mmr(query, llm_cfg),
                    'rag_with_hybrid_search_answer_with_bad_index': rag_with_hybrid_search(query, llm_cfg),
                    #'hallucinations_check': hallucinations_check(query, llm_cfg),
                    }
