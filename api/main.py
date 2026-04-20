import os
import secrets
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent.agent import AgentResponse, get_agent, parser

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)

PROVIDER_ORDER = ("llama", "openai", "claude")
PROVIDER_SETTINGS = {
    "llama": {
        "label": "Llama 3.1 via Groq",
        "env_var": "GROQ_API_KEY",
        "requires_password": False,
    },
    "openai": {
        "label": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "requires_password": True,
    },
    "claude": {
        "label": "Claude",
        "env_var": "CLAUDE_API_KEY",
        "requires_password": True,
    },
}
agents = {}

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


class ChatRequest(BaseModel):
    session_id: str
    message: str
    provider: str = "llama"


class AccessVerifyRequest(BaseModel):
    password: str


class AccessVerifyResponse(BaseModel):
    authorized: bool


class ProviderPublicConfig(BaseModel):
    id: str
    label: str
    requires_password: bool
    available: bool
    reason: str | None = None


class AppConfigResponse(BaseModel):
    default_provider: str
    providers: list[ProviderPublicConfig]


def get_app_access_password() -> str | None:
    password = os.getenv("APP_ACCESS_PASSWORD", "").strip()
    return password or None


def get_provider_config(provider: str) -> dict:
    normalized_provider = provider.lower().strip()
    if normalized_provider not in PROVIDER_SETTINGS:
        raise HTTPException(status_code=400, detail=f"Provedor '{provider}' não suportado.")
    return PROVIDER_SETTINGS[normalized_provider]


def get_provider_reason(provider: str) -> str | None:
    config = PROVIDER_SETTINGS[provider]

    if not os.getenv(config["env_var"]):
        return f"Configure {config['env_var']} no servidor para habilitar este provedor."

    if config["requires_password"] and not get_app_access_password():
        return "Configure APP_ACCESS_PASSWORD no servidor para liberar os provedores protegidos."

    return None


def provider_is_available(provider: str) -> bool:
    return get_provider_reason(provider) is None


def build_provider_public_config(provider: str) -> ProviderPublicConfig:
    config = PROVIDER_SETTINGS[provider]
    reason = get_provider_reason(provider)
    return ProviderPublicConfig(
        id=provider,
        label=config["label"],
        requires_password=config["requires_password"],
        available=reason is None,
        reason=reason,
    )


def get_default_provider() -> str:
    available_free_providers = [
        provider
        for provider in PROVIDER_ORDER
        if provider_is_available(provider) and not PROVIDER_SETTINGS[provider]["requires_password"]
    ]
    if available_free_providers:
        return available_free_providers[0]

    available_providers = [provider for provider in PROVIDER_ORDER if provider_is_available(provider)]
    if available_providers:
        return available_providers[0]

    return PROVIDER_ORDER[0]


def require_provider_access(provider: str, access_password: str | None) -> None:
    config = PROVIDER_SETTINGS[provider]
    if not config["requires_password"]:
        return

    expected_password = get_app_access_password()
    if not expected_password:
        raise HTTPException(
            status_code=503,
            detail="A senha do aplicativo não está configurada no servidor.",
        )

    candidate_password = (access_password or "").strip()
    if not candidate_password:
        raise HTTPException(
            status_code=403,
            detail="Este provedor exige a senha de acesso do aplicativo.",
        )

    if not secrets.compare_digest(candidate_password, expected_password):
        raise HTTPException(
            status_code=403,
            detail="Senha inválida para usar o provedor solicitado.",
        )


def resolve_agent(provider: str):
    normalized_provider = provider.lower().strip()
    get_provider_config(normalized_provider)

    if not provider_is_available(normalized_provider):
        raise HTTPException(
            status_code=503,
            detail=get_provider_reason(normalized_provider)
            or f"Provedor '{normalized_provider}' não está configurado no servidor.",
        )

    if normalized_provider not in agents:
        try:
            agents[normalized_provider] = get_agent(normalized_provider)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Provedor '{normalized_provider}' não está configurado no servidor.",
            ) from exc
        except Exception as exc:
            print(f"Erro ao inicializar provedor '{normalized_provider}': {exc}")
            raise HTTPException(
                status_code=500,
                detail="Erro interno ao inicializar o provedor solicitado.",
            ) from exc

    return agents[normalized_provider]


@app.get("/", include_in_schema=False)
async def root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return RedirectResponse(url="/docs")


@app.get("/api/config", response_model=AppConfigResponse)
async def app_config():
    return AppConfigResponse(
        default_provider=get_default_provider(),
        providers=[build_provider_public_config(provider) for provider in PROVIDER_ORDER],
    )


@app.post("/api/access/verify", response_model=AccessVerifyResponse)
async def verify_access(request: AccessVerifyRequest):
    if not get_app_access_password():
        raise HTTPException(
            status_code=503,
            detail="A senha do aplicativo não está configurada no servidor.",
        )

    candidate_password = request.password.strip()
    if not candidate_password:
        raise HTTPException(status_code=400, detail="Informe a senha para validar o acesso.")

    if not secrets.compare_digest(candidate_password, get_app_access_password() or ""):
        raise HTTPException(status_code=403, detail="Senha inválida.")

    return AccessVerifyResponse(authorized=True)


@app.post("/api/chat", response_model=AgentResponse)
async def chat_endpoint(
    request: ChatRequest,
    x_access_password: str | None = Header(default=None),
):
    provider = request.provider.lower().strip()
    get_provider_config(provider)

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="A mensagem não pode estar vazia.")

    require_provider_access(provider, x_access_password)
    agent_executor = resolve_agent(provider)
    config = {"configurable": {"thread_id": request.session_id}}
    inputs = {"messages": [HumanMessage(content=request.message)]}

    try:
        result = await agent_executor.ainvoke(inputs, config=config)
        last_message = result["messages"][-1].content
        parsed_response = parser.parse(last_message)
        return parsed_response
    except HTTPException:
        raise
    except Exception as exc:
        print(f"Erro ao processar com o provedor '{provider}': {exc}")
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao processar a resposta do modelo.",
        ) from exc


@app.get("/api/health")
async def health_check():
    return {"status": "online"}
