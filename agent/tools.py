from langchain_core.tools import tool

# Importe os retrievers que você acabou de criar
from agent.retriever import retrieve_ranked_documents, retrieve_similarity_documents

@tool
def search_project_knowledge(query: str) -> str:
    """
    Busca informações na base de conhecimento do projeto. 
    Use esta ferramenta SEMPRE que o usuário fizer perguntas técnicas.
    
    Exemplo de uso:
    Usuário: "Como o sistema valida o token JWT?"
    Ação: search_project_knowledge(query="validação token JWT auth.py")
    
    Args:
        query: A pergunta ou termos chave para a busca.
    """
    
    # Defina as coleções que o agente deve buscar por padrão. 
    # Adapte esses nomes para as coleções que você realmente possui no Chroma.
    collections = ["logica", "teoria"] 
    
    # Vamos usar o rerank porque você já provou que ele traz resultados mais limpos
    documents = retrieve_ranked_documents(collection_names=collections, query=query)
    
    if not documents:
        return "Nenhum documento relevante foi encontrado na base de conhecimento para esta consulta."
    
    # O Agente precisa de texto puro de volta. 
    # Vamos formatar os documentos em uma string bem estruturada.
    formatted_results = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Arquivo Desconhecido")
        content = doc.page_content
        formatted_results.append(f"--- Documento {index} | Fonte: {source} ---\n{content}")
        
    return "\n\n".join(formatted_results)

# (Opcional) Se você quiser dar ao agente a opção de fazer uma busca mais rápida sem rerank
@tool
def fast_similarity_search(query: str) -> str:
    """
    Realiza uma busca rápida e superficial por similaridade vetorial.
    Use apenas se 'search_project_knowledge' falhar ou se precisar de muito contexto difuso rapidamente.
    """
    collections = ["logica", "teoria"]
    documents = retrieve_similarity_documents(collection_names=collections, query=query, k=5)
    
    if not documents:
        return "Nenhum documento encontrado."
        
    formatted_results = [f"Fonte: {d.metadata.get('source', '?')}\n{d.page_content}" for d in documents]
    return "\n\n".join(formatted_results)
