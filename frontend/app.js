const STORAGE_KEYS = {
  sessionId: "tcc-qa.session-id",
  provider: "tcc-qa.provider",
  history: "tcc-qa.history",
};

const API_ENDPOINT =
  window.location.protocol === "file:"
    ? "http://127.0.0.1:8000/api/chat"
    : `${window.location.origin}/api/chat`;

const state = {
  sessionId: loadOrCreateSessionId(),
  provider: loadProvider(),
  messages: loadHistory(),
  isSending: false,
};

const elements = {
  chatForm: document.querySelector("#chat-form"),
  chatInput: document.querySelector("#chat-input"),
  sendButton: document.querySelector("#send-button"),
  chatHistory: document.querySelector("#chat-history"),
  providerSelect: document.querySelector("#provider-select"),
  sessionIdDisplay: document.querySelector("#session-id-display"),
  statusBanner: document.querySelector("#status-banner"),
  resetSessionButton: document.querySelector("#reset-session-button"),
};

bootstrap();

function bootstrap() {
  elements.providerSelect.value = state.provider;
  elements.sessionIdDisplay.textContent = state.sessionId;
  elements.sessionIdDisplay.title = state.sessionId;
  renderMessages();
  updateComposerState();
  autoResizeTextarea();

  elements.chatForm.addEventListener("submit", handleSubmit);
  elements.chatInput.addEventListener("input", autoResizeTextarea);
  elements.chatInput.addEventListener("keydown", handleTextareaKeydown);
  elements.providerSelect.addEventListener("change", handleProviderChange);
  elements.resetSessionButton.addEventListener("click", startNewSession);
}

async function handleSubmit(event) {
  event.preventDefault();

  const content = elements.chatInput.value.trim();
  if (!content || state.isSending) {
    return;
  }

  hideStatus();
  const userMessage = createMessage({
    role: "user",
    content,
  });

  const pendingMessage = createMessage({
    role: "assistant",
    content: "",
    pending: true,
    provider: state.provider,
  });

  state.messages.push(userMessage, pendingMessage);
  persistHistory();
  renderMessages();

  elements.chatInput.value = "";
  autoResizeTextarea();
  state.isSending = true;
  updateComposerState();

  try {
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_id: state.sessionId,
        message: content,
        provider: state.provider,
      }),
    });

    const payload = await parseJsonSafe(response);
    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, response.status));
    }

    replacePendingMessage(
      createMessage({
        role: "assistant",
        content: payload.answer || "A API respondeu sem conteúdo.",
        sourcesUsed: Array.isArray(payload.sources_used) ? payload.sources_used : [],
        exactChunks: Array.isArray(payload.exact_chunks) ? payload.exact_chunks : [],
        provider: state.provider,
      }),
    );
  } catch (error) {
    const errorMessage =
      error instanceof Error
        ? error.message
        : "Não foi possível se comunicar com o backend.";

    replacePendingMessage(
      createMessage({
        role: "assistant",
        content: errorMessage,
        isError: true,
        provider: state.provider,
      }),
    );

    showStatus(errorMessage);
  } finally {
    state.isSending = false;
    persistHistory();
    renderMessages();
    updateComposerState();
    elements.chatInput.focus();
  }
}

function handleTextareaKeydown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.chatForm.requestSubmit();
  }
}

function handleProviderChange(event) {
  state.provider = event.target.value;
  localStorage.setItem(STORAGE_KEYS.provider, state.provider);
}

function startNewSession() {
  if (state.isSending) {
    return;
  }

  state.sessionId = createSessionId();
  state.messages = [createWelcomeMessage()];
  localStorage.setItem(STORAGE_KEYS.sessionId, state.sessionId);
  persistHistory();
  elements.sessionIdDisplay.textContent = state.sessionId;
  elements.sessionIdDisplay.title = state.sessionId;
  hideStatus();
  renderMessages();
  elements.chatInput.focus();
}

function createWelcomeMessage() {
  return createMessage({
    role: "system",
    content:
      "Pergunte sobre o projeto e eu envio sua consulta para a API do agente. Use o seletor para trocar o provedor antes de enviar a mensagem.",
  });
}

function createMessage({
  role,
  content,
  pending = false,
  isError = false,
  sourcesUsed = [],
  exactChunks = [],
  provider = null,
}) {
  return {
    id:
      typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    role,
    content,
    pending,
    isError,
    sourcesUsed,
    exactChunks,
    provider,
    createdAt: new Date().toISOString(),
  };
}

function replacePendingMessage(nextMessage) {
  const index = state.messages.findIndex((message) => message.pending);
  if (index === -1) {
    state.messages.push(nextMessage);
    return;
  }

  state.messages.splice(index, 1, nextMessage);
}

function renderMessages() {
  elements.chatHistory.textContent = "";

  state.messages.forEach((message) => {
    const row = document.createElement("article");
    row.className = `message-row message-row--${message.role}`;

    const card = document.createElement("div");
    const visualRole = message.isError ? "error" : message.role;
    card.className = `message-card message-card--${visualRole}`;

    const meta = document.createElement("div");
    meta.className = "message-meta";

    const author = document.createElement("span");
    author.className = "message-author";
    author.textContent = getAuthorLabel(message);

    const timestamp = document.createElement("time");
    timestamp.dateTime = message.createdAt;
    timestamp.textContent = formatTimestamp(message.createdAt);

    meta.append(author, timestamp);

    const body = document.createElement("div");
    body.className = "message-body";

    if (message.pending) {
      const typingIndicator = document.createElement("div");
      typingIndicator.className = "typing-indicator";
      typingIndicator.setAttribute("aria-label", "Assistente digitando");
      typingIndicator.innerHTML = "<span></span><span></span><span></span>";
      body.appendChild(typingIndicator);
    } else if (message.role === "user") {
      body.textContent = message.content;
    } else {
      body.innerHTML = renderMarkdown(message.content);
    }

    card.append(meta, body);

    if (
      message.role === "assistant" &&
      !message.pending &&
      (message.sourcesUsed.length || message.exactChunks.length)
    ) {
      card.appendChild(buildSourcesAccordion(message));
    }

    row.appendChild(card);
    elements.chatHistory.appendChild(row);
  });

  elements.chatHistory.scrollTop = elements.chatHistory.scrollHeight;
  window.scrollTo({
    top: document.body.scrollHeight,
    behavior: "smooth",
  });
}

function buildSourcesAccordion(message) {
  const details = document.createElement("details");
  details.className = "message-sources";

  const summary = document.createElement("summary");
  summary.textContent = "Ver Fontes";

  const content = document.createElement("div");
  content.className = "sources-content";

  if (message.sourcesUsed.length) {
    const sourcesTitle = document.createElement("strong");
    sourcesTitle.textContent = "sources_used";

    const list = document.createElement("div");
    list.className = "sources-list";

    message.sourcesUsed.forEach((source) => {
      const chip = document.createElement("div");
      chip.className = "source-chip";
      chip.textContent = source;
      list.appendChild(chip);
    });

    content.append(sourcesTitle, list);
  }

  if (message.exactChunks.length) {
    const chunksTitle = document.createElement("strong");
    chunksTitle.textContent = "exact_chunks";

    const chunks = document.createElement("div");
    chunks.className = "chunks-list";

    message.exactChunks.forEach((chunk) => {
      const card = document.createElement("div");
      card.className = "chunk-card";
      card.textContent = chunk;
      chunks.appendChild(card);
    });

    content.append(chunksTitle, chunks);
  }

  details.append(summary, content);
  return details;
}

function renderMarkdown(markdown) {
  if (!markdown) {
    return "<p>Sem conteúdo retornado.</p>";
  }

  const codeBlocks = [];
  let html = escapeHtml(markdown).replace(
    /```([\w-]*)\n([\s\S]*?)```/g,
    (_, language, code) => {
      const token = `__CODE_BLOCK_${codeBlocks.length}__`;
      codeBlocks.push({
        language,
        code: code.trimEnd(),
      });
      return token;
    },
  );

  html = html.replace(/\r\n/g, "\n");

  const blocks = html
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean)
    .map((block) => renderBlock(block));

  let rendered = blocks.join("");
  rendered = rendered.replace(
    /__CODE_BLOCK_(\d+)__/g,
    (_, index) => renderCodeBlock(codeBlocks[Number(index)]),
  );

  return rendered;
}

function renderBlock(block) {
  if (/^__CODE_BLOCK_\d+__$/.test(block)) {
    return block;
  }

  if (/^#{1,3}\s/.test(block)) {
    const [, hashes, content] = block.match(/^(#{1,3})\s+(.*)$/) || [];
    const level = hashes ? hashes.length : 3;
    return `<h${level}>${formatInlineMarkdown(content || "")}</h${level}>`;
  }

  const listLines = block.split("\n");
  if (listLines.every((line) => /^[-*]\s+/.test(line))) {
    const items = listLines
      .map((line) => line.replace(/^[-*]\s+/, ""))
      .map((line) => `<li>${formatInlineMarkdown(line)}</li>`)
      .join("");
    return `<ul>${items}</ul>`;
  }

  if (listLines.every((line) => /^\d+\.\s+/.test(line))) {
    const items = listLines
      .map((line) => line.replace(/^\d+\.\s+/, ""))
      .map((line) => `<li>${formatInlineMarkdown(line)}</li>`)
      .join("");
    return `<ol>${items}</ol>`;
  }

  if (block.startsWith("&gt;")) {
    const quote = block
      .split("\n")
      .map((line) => line.replace(/^&gt;\s?/, ""))
      .join("<br>");
    return `<blockquote>${formatInlineMarkdown(quote)}</blockquote>`;
  }

  return `<p>${formatInlineMarkdown(block).replace(/\n/g, "<br>")}</p>`;
}

function formatInlineMarkdown(content) {
  return content
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function renderCodeBlock(block) {
  if (!block) {
    return "";
  }

  const languageClass = block.language ? ` class="language-${block.language}"` : "";
  return `<pre><code${languageClass}>${block.code}</code></pre>`;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function updateComposerState() {
  elements.chatInput.disabled = state.isSending;
  elements.sendButton.disabled = state.isSending;
  elements.providerSelect.disabled = state.isSending;
  elements.resetSessionButton.disabled = state.isSending;
}

function autoResizeTextarea() {
  elements.chatInput.style.height = "auto";
  elements.chatInput.style.height = `${elements.chatInput.scrollHeight}px`;
}

function showStatus(message) {
  elements.statusBanner.textContent = message;
  elements.statusBanner.classList.remove("hidden");
}

function hideStatus() {
  elements.statusBanner.textContent = "";
  elements.statusBanner.classList.add("hidden");
}

function persistHistory() {
  localStorage.setItem(STORAGE_KEYS.history, JSON.stringify(state.messages));
}

function loadHistory() {
  const rawHistory = localStorage.getItem(STORAGE_KEYS.history);
  if (!rawHistory) {
    return [createWelcomeMessage()];
  }

  try {
    const parsedHistory = JSON.parse(rawHistory);
    if (!Array.isArray(parsedHistory) || !parsedHistory.length) {
      return [createWelcomeMessage()];
    }

    const normalizedMessages = parsedHistory
      .filter(
        (message) =>
          message &&
          typeof message.id === "string" &&
          typeof message.role === "string" &&
          typeof message.content === "string",
      )
      .map((message) => ({
        ...message,
        pending: false,
        isError: Boolean(message.isError),
        sourcesUsed: Array.isArray(message.sourcesUsed) ? message.sourcesUsed : [],
        exactChunks: Array.isArray(message.exactChunks) ? message.exactChunks : [],
        provider: typeof message.provider === "string" ? message.provider : null,
      }));

    return normalizedMessages.length ? normalizedMessages : [createWelcomeMessage()];
  } catch {
    return [createWelcomeMessage()];
  }
}

function loadProvider() {
  const provider = localStorage.getItem(STORAGE_KEYS.provider);
  return ["openai", "claude", "llama"].includes(provider) ? provider : "openai";
}

function loadOrCreateSessionId() {
  const savedSessionId = localStorage.getItem(STORAGE_KEYS.sessionId);
  if (savedSessionId) {
    return savedSessionId;
  }

  const nextSessionId = createSessionId();
  localStorage.setItem(STORAGE_KEYS.sessionId, nextSessionId);
  return nextSessionId;
}

function createSessionId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `tcc-qa-${crypto.randomUUID()}`;
  }

  return `tcc-qa-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function getAuthorLabel(message) {
  if (message.isError) {
    return "Erro";
  }

  if (message.role === "user") {
    return "Você";
  }

  if (message.role === "system") {
    return "Sistema";
  }

  return `IA · ${message.provider || state.provider}`;
}

function formatTimestamp(isoDate) {
  const date = new Date(isoDate);
  if (Number.isNaN(date.getTime())) {
    return "";
  }

  return new Intl.DateTimeFormat("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

async function parseJsonSafe(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function extractErrorMessage(payload, statusCode) {
  if (payload && typeof payload.detail === "string") {
    return payload.detail;
  }

  if (statusCode === 503) {
    return "O provedor selecionado não está configurado no servidor.";
  }

  if (statusCode === 400) {
    return "A API rejeitou a requisição. Revise o modelo selecionado e tente novamente.";
  }

  return "Não foi possível processar a resposta da API. Verifique se o backend está online.";
}
