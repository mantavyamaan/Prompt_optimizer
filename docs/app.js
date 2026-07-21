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
let conversationId = null;

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
  if (options.traceId) {
    actions.hidden = false;
    const slider = node.querySelector('.score-slider');
    const display = node.querySelector('.score-display');
    const submitBtn = node.querySelector('.score-submit');
    if (slider && display && submitBtn) {
        slider.addEventListener('input', () => { display.textContent = slider.value; });
        submitBtn.addEventListener('click', () => submitScore(node, slider.value, submitBtn));
    }
    node.querySelector('[data-copy-response]').addEventListener('click', () => copyText(content, node.querySelector('[data-copy-response]')));
  } else {
    actions.remove();
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
  // Simulate network request
  await new Promise(resolve => setTimeout(resolve, 500));
  button.textContent = 'Rated';
  button.disabled = true;
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
  try {
    await new Promise(resolve => setTimeout(resolve, 1500));
    const data = {
        output: "This is a static demo preview. To use the full Prompt Optimizer, view traces, connect LLMs, and automatically evolve your prompts using Genetic Algorithms, please install the project from GitHub:\n\nhttps://github.com/mantavyamaan/Prompt_optimizer",
        trace_id: "demo-trace-" + Math.floor(Math.random() * 10000),
        confidence: 0.99,
        category: "prompt_design",
        backend: "demo-model",
        source: "github-pages"
    };
    typing.remove();
    lastOutput = data.output;
    addMessage(data.output, 'Prompt Optimizer', { traceId: data.trace_id });
    document.querySelector('#confidence').textContent = `${Math.round(data.confidence * 100)}%`;
    document.querySelector('#latency').textContent = `${Math.round(performance.now() - started)}ms`;
    document.querySelector('#trace').textContent = `trace_id  ${data.trace_id}\ncategory  ${data.category}\nbackend   ${data.backend}\nsource    ${data.source}`;
    setConnection({ label: 'Demo Mode', detail: 'Read-only preview', source: 'DEMO' });
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
  setConnection({ label: 'Demo Mode Active', detail: 'Read-only preview on GitHub Pages', source: 'DEMO' });
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

sendButton.addEventListener('click', () => {
  ask(input.value);
});
input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(input.value); } });
input.addEventListener('input', autoResize);
document.querySelector('#newChat').addEventListener('click', resetConversation);
document.querySelectorAll('.suggestion').forEach(btn => { btn.addEventListener('click', () => ask(btn.dataset.prompt)); });
refreshHealth();
