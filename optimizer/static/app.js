const input = document.querySelector('#promptInput');
const composer = document.querySelector('#composer');
const sendButton = document.querySelector('#sendButton');
const messages = document.querySelector('#messages');
const chatScroll = document.querySelector('#chatScroll');
const welcome = document.querySelector('#welcome');
const template = document.querySelector('#messageTemplate');
const providerDialog = document.querySelector('#providerDialog');
const providerForm = document.querySelector('#providerForm');
const providerError = document.querySelector('#providerError');

let sessionProvider = null;
let submitting = false;
let lastOutput = '';

function timeLabel() {
  return new Intl.DateTimeFormat([], { hour: 'numeric', minute: '2-digit' }).format(new Date());
}

function scrollToEnd() {
  chatScroll.scrollTo({ top: chatScroll.scrollHeight, behavior: 'smooth' });
}

function autoResize() {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 145)}px`;
}

function setConnection({ label, detail, source = 'OLLAMA', offline = false }) {
  document.querySelector('#connectionText').textContent = label;
  document.querySelector('#modelState').textContent = detail;
  document.querySelector('#sourceChip').textContent = source;
  document.querySelector('#statusDot').style.background = offline ? 'var(--danger)' : 'var(--lime)';
}

function addMessage(content, author, options = {}) {
  const node = template.content.firstElementChild.cloneNode(true);
  node.classList.toggle('user', author === 'You');
  node.dataset.traceId = options.traceId || '';
  node.querySelector('.message-meta b').textContent = author;
  node.querySelector('time').textContent = timeLabel();
  node.querySelector('pre').textContent = content;
  const actions = node.querySelector('.message-actions');
  actions.hidden = !options.traceId;
  if (options.traceId) {
    const slider = node.querySelector('.score-slider');
    const display = node.querySelector('.score-display');
    const submitBtn = node.querySelector('.score-submit');
    if (slider && display && submitBtn) {
        slider.addEventListener('input', () => { display.textContent = slider.value; });
        submitBtn.addEventListener('click', () => submitScore(node, slider.value, submitBtn));
    }
    node.querySelector('[data-copy-response]').addEventListener('click', () => copyText(content, node.querySelector('[data-copy-response]')));
  }
  messages.append(node);
  scrollToEnd();
  return node;
}

function addTyping() {
  const node = document.createElement('article');
  node.className = 'message';
  node.innerHTML = '<div class="avatar" aria-hidden="true"></div><div class="typing" aria-label="Generating"><i></i><i></i><i></i></div>';
  messages.append(node);
  scrollToEnd();
  return node;
}

function setSubmitting(value) {
  submitting = value;
  sendButton.disabled = value;
  sendButton.querySelector('span').textContent = value ? 'Generating…' : 'Generate prompt';
}

async function copyText(value, button) {
  try {
    await navigator.clipboard.writeText(value);
    const original = button.textContent;
    button.textContent = 'Copied';
    window.setTimeout(() => { button.textContent = original; }, 1300);
  } catch {
    button.textContent = 'Copy failed';
  }
}

async function submitScore(message, score, button) {
  const traceId = message.dataset.traceId;
  if (!traceId) return;
  try {
    const response = await fetch('/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ trace_id: traceId, signal: `score:${score}` }) });
    if (!response.ok) throw new Error();
    button.textContent = 'Rated';
    button.disabled = true;
  } catch {
    button.textContent = 'Try again';
  }
}

async function ask(rawText) {
  const text = rawText.trim();
  if (!text || submitting) return;
  welcome.hidden = true;
  addMessage(text, 'You');
  input.value = '';
  autoResize();
  setSubmitting(true);
  const typing = addTyping();
  const started = performance.now();
  const body = { text };
  if (sessionProvider) body.provider = sessionProvider;
  try {
    const response = await fetch('/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'The selected model is unavailable.');
    typing.remove();
    lastOutput = data.output;
    addMessage(data.output, 'Prompt Optimizer', { traceId: data.trace_id });
    document.querySelector('#confidence').textContent = `${Math.round(data.confidence * 100)}%`;
    document.querySelector('#latency').textContent = `${Math.round(performance.now() - started)}ms`;
    document.querySelector('#trace').textContent = `trace_id  ${data.trace_id}\ncategory  ${data.category}\nbackend   ${data.backend}\nsource    ${data.source}`;
    setConnection({ label: data.source === 'api_key' ? 'API key connected' : 'Ollama connected', detail: `${data.backend} · ${data.source === 'api_key' ? 'session API key' : 'local Ollama'}`, source: data.source === 'api_key' ? 'API KEY' : 'OLLAMA' });
  } catch (error) {
    typing.remove();
    addMessage(`I couldn’t generate a prompt: ${error.message}`, 'Prompt Optimizer');
    setConnection({ label: sessionProvider ? 'API provider unavailable' : 'Ollama unavailable', detail: 'Check your selected model connection and try again.', source: sessionProvider ? 'API KEY' : 'OLLAMA', offline: true });
  } finally {
    setSubmitting(false);
    input.focus();
  }
}

async function refreshHealth() {
  try {
    const response = await fetch('/health');
    const data = await response.json();
    setConnection({ label: sessionProvider ? 'API key selected' : 'Ollama default', detail: sessionProvider ? `${sessionProvider.model} · session API key` : `${data.backend} · local Ollama`, source: sessionProvider ? 'API KEY' : 'OLLAMA' });
  } catch {
    setConnection({ label: 'Server unavailable', detail: 'Start Prompt Optimizer, then refresh this page.', offline: true });
  }
}

function resetConversation() {
  messages.replaceChildren();
  welcome.hidden = false;
  lastOutput = '';
  conversationId = null;
  document.querySelector('#confidence').textContent = '—';
  document.querySelector('#latency').textContent = '—';
  document.querySelector('#trace').textContent = 'No response yet.';
  input.focus();
}

composer.addEventListener('submit', (e) => {
  e.preventDefault();
  ask(input.value);
});

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    ask(input.value);
  }
});

input.addEventListener('input', autoResize);

document.querySelector('#newChat').addEventListener('click', resetConversation);

document.querySelectorAll('.suggestion').forEach(btn => {
  btn.addEventListener('click', () => ask(btn.dataset.prompt));
});

document.querySelector('#openProviderFromSide')?.addEventListener('click', () => providerDialog.showModal());
document.querySelector('#closeProvider')?.addEventListener('click', () => providerDialog.close());
document.querySelector('#clearProvider')?.addEventListener('click', () => {
  sessionProvider = null;
  providerDialog.close();
  refreshHealth();
});

if (providerForm) {
  providerForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const formData = new FormData(providerForm);
    const url = formData.get('base_url');
    const model = formData.get('model');
    const key = formData.get('api_key');
    if (!url || !model) {
      providerError.textContent = 'Base URL and Model are required.';
      providerError.hidden = false;
      return;
    }
    providerError.hidden = true;
    sessionProvider = { url, model, key };
    providerDialog.close();
    refreshHealth();
  });
}

document.querySelector('#ollamaChoice')?.addEventListener('click', () => {
  sessionProvider = null;
  providerDialog.close();
  refreshHealth();
});

refreshHealth();

