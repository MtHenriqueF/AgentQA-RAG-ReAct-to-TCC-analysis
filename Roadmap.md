- [X] Arquitetura e Fluxo : Definição do escopo, agentes e ferramentas.

- [X] Ingestão de Dados (Atual): Criar o script Python que varre a pasta do TCC, ignora lixos (.mat, .csv), faz o chunking correto por linguagem e salva no ChromaDB persistido no HD.

- [X] Módulo de Recuperação (RAG + Rerank): Criar as funções que buscam no banco vetorial e passam pela API da Cohere para filtrar os Top 3 resultados matemáticos mais exatos.

- [X] Construção das Tools: "Envelopar" as funções de busca do passo 3 usando o decorador @tool do LangChain para que o LLM possa chamá-las.

- [ ] Orquestração do Agente ReAct: Instanciar o LLM (OpenAI) e entregar as ferramentas para ele. Testar o agente via terminal (sem frontend).

- [ ] Desenvolvimento da API (FastAPI): Criar a rota de comunicação HTTP (/api/chat).

- [ ] Frontend (Integração): Desenvolver o HTML/JS e conectar à API.

- [ ] Dockerização: Criar o Dockerfile final apenas para encapsular o que já funciona.