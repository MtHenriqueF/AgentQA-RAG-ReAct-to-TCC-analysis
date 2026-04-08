from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# Importa o agente E o parser que criamos
from agent.agent import get_agent, parser, AgentResponse

app = FastAPI(title="TCC QA Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

agents = {
    "openai": get_agent("openai"),
    # "claude": get_agent("claude"), # Descomente quando a chave estiver no .env
}

class ChatRequest(BaseModel):
    session_id: str
    message: str
    provider: str = "openai"

@app.post("/api/chat", response_model=AgentResponse) # Usamos o próprio modelo Pydantic do agente
async def chat_endpoint(request: ChatRequest):
    provider = request.provider.lower()
    if provider not in agents:
        raise HTTPException(status_code=400, detail=f"Provedor '{provider}' não suportado.")
    
    agent_executor = agents[provider]
    config = {"configurable": {"thread_id": request.session_id}}
    inputs = {"messages": [HumanMessage(content=request.message)]}
    
    try:
        resultado = await agent_executor.ainvoke(inputs, config=config)
        ultima_mensagem_str = resultado["messages"][-1].content
        
        # O parser resolve markdown (```json) e valida os campos automaticamente!
        parsed_response = parser.parse(ultima_mensagem_str)
        
        return parsed_response
        
    except Exception as e:
        print(f"Erro ao processar: {e}")
        # Retorno de segurança para a API não quebrar
        raise HTTPException(status_code=500, detail="Erro interno ao processar a resposta do modelo.")

@app.get("/api/health")
async def health_check():
    return {"status": "online"}
