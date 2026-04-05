import os
import glob
import hashlib
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

# 1. Carrega as chaves do .env (OpenAI)
load_dotenv(dotenv_path=ENV_FILE)

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# --- CONFIGURAÇÕES ---
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db_local")
EMBEDDING_MODEL = "text-embedding-3-small"
TEXT_FILE_ENCODINGS = ("utf-8", "iso-8859-1", "latin-1", "cp1252")

# AGORA É UMA LISTA: Olhamos uma pasta para trás (../) e acessamos os Artigos e o Código
DIRECTORIES_TO_SCAN = [
    os.path.abspath(os.path.join(PROJECT_ROOT, "Artigos")),
    os.path.abspath(os.path.join(PROJECT_ROOT, "PPSUS-main/IC-master")),
]

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY não encontrada. Verifique o arquivo .env na raiz do projeto."
    )

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# O Guarda de Trânsito
FILE_ROUTING = {
    ".md": {"collection": "teoria", "language": Language.MARKDOWN},
    ".pdf": {"collection": "teoria", "language": None},
    
    ".py": {"collection": "logica", "language": Language.PYTHON},
    ".c": {"collection": "logica", "language": Language.C},
    ".h": {"collection": "logica", "language": Language.C},
    
    ".js": {"collection": "frontend", "language": Language.JS},
    ".html": {"collection": "frontend", "language": Language.HTML},
    ".css": {"collection": "frontend", "language": None}
}

FOLDERS_TO_IGNORE = ["saves", "codegen", "Matlab Code", "__pycache__", "mex", "html"]
COLLECTION_NAMES = sorted({config["collection"] for config in FILE_ROUTING.values()})

def should_ignore(filepath):
    normalized_path = filepath.replace("\\", "/")

    # Verifica pasta ignorada
    for folder in FOLDERS_TO_IGNORE:
        if f"/{folder}/" in normalized_path:
            return True
            
    # Verifica extensão
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in FILE_ROUTING:
        return True # Barrado na porta!
        
    return False

def get_text_splitter(ext):
    routing_info = FILE_ROUTING[ext]
    lang = routing_info["language"]
    
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=800, chunk_overlap=100
        )
    else:
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

def iter_files_to_scan():
    for directory in DIRECTORIES_TO_SCAN:
        if not os.path.isdir(directory):
            print(f"Diretório não encontrado, ignorando: {directory}")
            continue

        pattern = os.path.join(directory, "**", "*")
        for filepath in sorted(glob.iglob(pattern, recursive=True)):
            absolute_path = os.path.abspath(filepath)
            if os.path.isfile(absolute_path) and not should_ignore(absolute_path):
                yield absolute_path

def calculate_file_hash(filepath):
    digest = hashlib.sha256()

    with open(filepath, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()

def build_chunk_id(filepath, file_hash, chunk_index):
    raw_id = f"{filepath}:{file_hash}:{chunk_index}".encode("utf-8")
    return hashlib.sha256(raw_id).hexdigest()

def get_existing_entries(vector_store, filepath):
    return vector_store.get(where={"source": filepath}, include=["metadatas"])

def decide_ingest_action(existing_entries, file_hash):
    ids = existing_entries.get("ids", [])
    if not ids:
        return "insert", "novo arquivo"

    metadatas = [metadata or {} for metadata in existing_entries.get("metadatas", [])]
    stored_hashes = {metadata.get("file_hash") for metadata in metadatas if metadata.get("file_hash")}
    has_legacy_entries = any(not metadata.get("file_hash") for metadata in metadatas)

    if not stored_hashes:
        return "skip", "já indexado (sem hash legado)"

    if stored_hashes == {file_hash} and not has_legacy_entries:
        return "skip", "já indexado"

    return "update", "arquivo alterado"

def load_text_documents(filepath):
    last_error = None

    for encoding in TEXT_FILE_ENCODINGS:
        try:
            with open(filepath, encoding=encoding) as file:
                return [
                    Document(
                        page_content=file.read(),
                        metadata={"source": filepath, "encoding": encoding},
                    )
                ]
        except UnicodeDecodeError as error:
            last_error = error

    raise RuntimeError(f"Não foi possível decodificar o arquivo: {filepath}") from last_error

def load_documents(filepath, ext):
    if ext == ".pdf":
        documents = PyPDFLoader(filepath).load()
    else:
        documents = load_text_documents(filepath)

    return [
        document
        for document in documents
        if document.page_content and document.page_content.strip()
    ]

def split_documents(documents, ext):
    splitter = get_text_splitter(ext)
    chunks = splitter.split_documents(documents)

    return [
        chunk
        for chunk in chunks
        if chunk.page_content and chunk.page_content.strip()
    ]

def prepare_chunks(chunks, filepath, ext, file_hash):
    texts = []
    metadatas = []
    ids = []

    for chunk_index, chunk in enumerate(chunks):
        text = chunk.page_content.strip()
        if not text:
            continue

        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "source": filepath,
                "extension": ext,
                "file_hash": file_hash,
                "chunk_index": chunk_index,
            }
        )

        texts.append(text)
        metadatas.append(metadata)
        ids.append(build_chunk_id(filepath, file_hash, chunk_index))

    return texts, metadatas, ids

def build_vector_stores():
    return {
        collection_name: Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        for collection_name in COLLECTION_NAMES
    }

def ingest_file(vector_store, filepath):
    ext = os.path.splitext(filepath)[1].lower()
    file_hash = calculate_file_hash(filepath)
    existing_entries = get_existing_entries(vector_store, filepath)
    action, reason = decide_ingest_action(existing_entries, file_hash)

    if action == "skip":
        print(f"Ignorando: {filepath} -> {reason}")
        return "skipped"

    documents = load_documents(filepath, ext)
    if not documents:
        print(f"Ignorando: {filepath} -> arquivo sem conteúdo legível")
        return "empty"

    chunks = split_documents(documents, ext)
    texts, metadatas, ids = prepare_chunks(chunks, filepath, ext, file_hash)

    if not texts:
        print(f"Ignorando: {filepath} -> nenhum chunk válido gerado")
        return "empty"

    if action == "update":
        vector_store.delete(ids=existing_entries["ids"])
        print(f"Atualizando: {filepath}")
    else:
        print(f"Inserindo: {filepath}")

    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return action

def ingest_project_data():
    print("Iniciando varredura e ingestão de dados...")

    vector_stores = build_vector_stores()
    summary = {
        "insert": 0,
        "update": 0,
        "skipped": 0,
        "empty": 0,
        "errors": 0,
    }

    for filepath in iter_files_to_scan():
        ext = os.path.splitext(filepath)[1].lower()
        collection_name = FILE_ROUTING[ext]["collection"]

        try:
            result = ingest_file(vector_stores[collection_name], filepath)
            summary[result] += 1
        except Exception as error:
            summary["errors"] += 1
            print(f"Erro ao processar {filepath}: {error}")

    print("\nResumo da ingestão:")
    print(f"- Novos arquivos inseridos: {summary['insert']}")
    print(f"- Arquivos atualizados: {summary['update']}")
    print(f"- Arquivos já indexados: {summary['skipped']}")
    print(f"- Arquivos vazios/sem conteúdo: {summary['empty']}")
    print(f"- Arquivos com erro: {summary['errors']}")
    print(f"\nBanco vetorial persistido em: {CHROMA_PERSIST_DIR}")

if __name__ == "__main__":
    ingest_project_data()
