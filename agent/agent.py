import os
import json
from dotenv import load_dotenv

from langchain.agents import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from tools import search_project_knowledge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=ENV_FILE)

def require_env(var_name: str):
    if not os.getenv(var_name):
        raise RuntimeError(f"Variável {var_name} não encontrada no .env")

# --- PROMPT TEMPLATE QA (JSON Obrigatório) ---
QA_SYSTEM_PROMPT = """Você é um assistente de IA especializado em Quality Assurance (QA) e Engenharia de Software.
Sua missão é responder a perguntas técnicas sobre as regras de negócio, arquitetura e código do projeto.

REGRAS OBRIGATÓRIAS:
1. Baseie-se SEMPRE nas informações fornecidas pelas ferramentas de busca.
2. Se a informação não estiver no contexto recuperado, diga explicitamente que não tem a informação. Não invente ou presuma dados.
3. Ao mencionar arquivos de código, explique brevemente a lógica associada a eles.
4. Você DEVE retornar a sua resposta final ESTRITAMENTE no formato JSON abaixo.
5. NÃO inclua blocos de formatação markdown (como ```json) ao redor da resposta. Apenas o JSON puro.

FORMATO DE SAÍDA ESPERADO:
{
  "status": "success",
  "answer": "Sua resposta detalhada aqui...",
  "sources_used": ["arquivo1.py", "arquivo2.c"]
}
"""

def get_llm(provider: str = "openai"):
    """
    Factory function para inicializar o LLM escolhido pelo Front-end.
    Opções: 'openai', 'claude', 'llama'
    """
    provider = provider.lower()
    
    if provider == "openai":
        require_env("OPENAI_API_KEY")
        return ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0,
            # Força o formato JSON nativamente na API da OpenAI
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
    elif provider == "claude":
        require_env("CLAUDE_API_KEY")
        # Claude 3.5 Haiku: O concorrente direto do 4o-mini em preço/velocidade.
        # A Anthropic lida muito bem com JSON apenas com o System Prompt.
        return ChatAnthropic(
            model_name="claude-3-5-haiku-20241022", 
            temperature=0,
            api_key=os.getenv("CLAUDE_API_KEY")
        )
        
    elif provider == "llama":
        # Para rodar Llama sem baixar o modelo via HuggingFace (pesado e lento), 
        # a Groq é a API em nuvem mais rápida e com tier gratuito.
        require_env("GROQ_API_KEY")
        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            # O Groq também suporta o response_format nativo para JSON
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
    else:
        raise ValueError(f"Provedor de LLM '{provider}' não suportado.")

def get_agent(provider: str = "openai"):
    llm = get_llm(provider)
    tools = [search_project_knowledge]
    
    return create_react_agent(
        llm=llm, 
        tools=tools, 
        state_modifier=QA_SYSTEM_PROMPT
    )

def test_agent_interactive():
    # Aqui você pode mudar para "claude" ou "llama" para testar
    provedor_atual = "openai" 
    
    agent_executor = get_agent(provider=provedor_atual)
    print(f"🤖 Agente QA [{provedor_atual.upper()}] inicializado. Digite 'sair' para encerrar.")
    print("-" * 50)
    
    while True:
        user_input = input("\nVocê: ")
        if user_input.lower() in ['sair', 'exit', 'quit']:
            break
            
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        print(f"\nProcessando com {provedor_atual.upper()}...")
        resultado = agent_executor.invoke(inputs)
        ultima_mensagem = resultado["messages"][-1].content
        
        try:
            json_output = json.loads(ultima_mensagem)
            print("\n🤖 Resposta (JSON Válido):")
            print(json.dumps(json_output, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print("\n⚠️ Erro de Parsing JSON. Retorno bruto:")
            print(ultima_mensagem)

if __name__ == "__main__":
    test_agent_interactive()