import os
import sys
from dotenv import load_dotenv

# Importações do LangGraph e LangChain
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Carrega as variáveis de ambiente (OpenAI, Cohere, etc.)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importe a ferramenta que criamos no passo anterior
from agent.tools import search_project_knowledge

load_dotenv(dotenv_path=ENV_FILE)

def require_env(var_name: str):
    if not os.getenv(var_name):
        raise RuntimeError(f"{var_name} não encontrada. Verifique o .env.")

# 1. Configurar o LLM (O "cérebro" do agente)
# Usamos temperature=0 para que o agente seja determinístico e não alucine código.
require_env("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# 2. Registrar as Ferramentas
tools = [search_project_knowledge]

# 3. Criar o agente usando a API atual do LangChain
agent_executor = create_agent(model=llm, tools=tools)

# 4. Função para testar no terminal
def test_agent_interactive():
    print("🤖 Agente inicializado. Digite 'sair' para encerrar.")
    print("-" * 50)
    
    while True:
        user_input = input("\nVocê: ")
        if user_input.lower() in ['sair', 'exit', 'quit']:
            break
            
        # O LangGraph trabalha recebendo e enviando listas de mensagens (Estado)
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        # O método stream permite ver os passos internos do agente no terminal
        print("\nProcessando", end="")
        for step in agent_executor.stream(inputs, stream_mode="values"):
            print(".", end="", flush=True)
            # A última mensagem no estado atual é a ação mais recente
            last_message = step["messages"][-1]
            
        print("\n\n🤖 Agente:")
        print(last_message.content)

if __name__ == "__main__":
    test_agent_interactive()
