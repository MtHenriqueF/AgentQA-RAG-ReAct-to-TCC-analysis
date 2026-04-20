from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# Importa o agente E o parser que criamos
from agent.agent import get_agent, parser, AgentResponse

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(title="TCC QA Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/", include_in_schema=False)
async def root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return RedirectResponse(url="/docs")

SUPPORTED_PROVIDERS = {"openai", "claude", "llama"}
agents = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str
    provider: str = "openai"


def resolve_agent(provider: str):
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Provedor '{provider}' não suportado.")

    if provider not in agents:
        try:
            agents[provider] = get_agent(provider)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Provedor '{provider}' não está configurado no servidor.",
            ) from exc
        except Exception as exc:
            print(f"Erro ao inicializar provedor '{provider}': {exc}")
            raise HTTPException(
                status_code=500,
                detail="Erro interno ao inicializar o provedor solicitado.",
            ) from exc

    return agents[provider]

@app.post("/api/chat", response_model=AgentResponse) # Usamos o próprio modelo Pydantic do agente
async def chat_endpoint(request: ChatRequest):
    provider = request.provider.lower()
    agent_executor = resolve_agent(provider)
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
