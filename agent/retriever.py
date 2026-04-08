import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_openai import OpenAIEmbeddings


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

load_dotenv(dotenv_path=ENV_FILE)

CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db_local")
EMBEDDING_MODEL = "text-embedding-3-small"
RERANK_MODEL = "rerank-multilingual-v3.0"
BASE_RETRIEVER_K = 15
RERANK_TOP_N = 5


def require_env(var_name: str):
    if not os.getenv(var_name):
        raise RuntimeError(
            f"{var_name} não encontrada. Verifique o arquivo .env na raiz do projeto."
        )


def normalize_collection_names(collection_names):
    if isinstance(collection_names, str):
        return [collection_names]
    return list(collection_names)


def build_embeddings():
    require_env("OPENAI_API_KEY")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def build_reranker(top_n: int = RERANK_TOP_N):
    require_env("COHERE_API_KEY")
    return CohereRerank(
        model=RERANK_MODEL,
        top_n=top_n,
    )


def build_vectorstore(collection_name: str):
    return Chroma(
        collection_name=collection_name,
        embedding_function=build_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def get_similarity_retriever(collection_name: str, k: int = BASE_RETRIEVER_K):
    vectorstore = build_vectorstore(collection_name)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_advanced_retriever(
    collection_name: str,
    k: int = BASE_RETRIEVER_K,
    top_n: int = RERANK_TOP_N,
):
    vectorstore = build_vectorstore(collection_name)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    compressor = build_reranker(top_n=top_n)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def retrieve_similarity_documents(collection_names, query: str, k: int = BASE_RETRIEVER_K):
    documents = []

    for collection_name in normalize_collection_names(collection_names):
        retriever = get_similarity_retriever(collection_name, k=k)
        for document in retriever.invoke(query):
            document.metadata["collection_name"] = collection_name
            documents.append(document)

    return documents


def retrieve_ranked_documents(
    collection_names,
    query: str,
    k: int = BASE_RETRIEVER_K,
    top_n: int = RERANK_TOP_N,
):
    documents = retrieve_similarity_documents(collection_names, query=query, k=k)
    if not documents:
        return []

    return list(build_reranker(top_n=top_n).compress_documents(documents, query=query))
