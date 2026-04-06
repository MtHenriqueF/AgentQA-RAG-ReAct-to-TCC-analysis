import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

# Carrega as chaves da OpenAI e Cohere do .env
load_dotenv(dotenv_path=ENV_FILE)

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever

# --- CONFIGURAÇÕES ---
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db_local")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY não encontrada. Verifique o arquivo .env na raiz do projeto."
    )

if not os.getenv("COHERE_API_KEY"):
    raise RuntimeError(
        "COHERE_API_KEY não encontrada. Verifique o arquivo .env na raiz do projeto."
    )

# Instanciamos o mesmo modelo de matemática para conseguirmos "ler" o banco
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def get_advanced_retriever(collection_name: str):
    """
    Cria um buscador avançado em duas etapas para uma coleção específica.
    Etapa 1: Pega os Top 15 resultados (ChromaDB)
    Etapa 2: Filtra e devolve os Top 3 exatos (Cohere Rerank)
    """
    
    # 1. A Rede Larga: Conecta ao banco vetorial existente
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    
    # Configura para trazer 15 documentos baseados na matemática pura
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    
    # 2. A Pinça: Configura o modelo de Rerank da Cohere
    # Usamos o modelo multilíngue pois seu TCC tem português e código (inglês)
    compressor = CohereRerank(
        model="rerank-multilingual-v3.0", 
        top_n=3  # Só queremos os 3 melhores textos no final
    )
    
    # 3. Une as duas etapas em uma ferramenta só
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever

# --- ÁREA DE TESTE LOCAL ---
if __name__ == "__main__":
    print("Testando o Retriever Avançado (Rerank)...")
    
    # Vamos testar buscando na gaveta de código (lógica)
    colecao = "logica" 
    pergunta = "Onde estão as definições de especialidade dos recursos materiais?"
    
    try:
        retriever = get_advanced_retriever(colecao)
        print(f"\nBuscando na coleção '{colecao}' pela pergunta: '{pergunta}'")
        
        # O .invoke dispara todo o pipeline (Rede + Pinça)
        resultados = retriever.invoke(pergunta)
        
        print("\n--- TOP 3 RESULTADOS ENCONTRADOS ---")
        for i, doc in enumerate(resultados):
            print(f"\nResultado {i+1} (Arquivo: {doc.metadata.get('source')}):")
            # Imprime os primeiros 200 caracteres para não poluir o terminal
            print(doc.page_content[:200] + "...\n")
            print("-" * 40)
            
    except Exception as e:
         print(f"Erro ao buscar: {e}")
