// Shared DOM helpers, not a UI framework. All remote text is escaped before rendering.
export const $ = (selector, root = document) => root.querySelector(selector);
export const escape = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[ch]));
export const problemLink = (id, mode = 'code') => `#/question/${encodeURIComponent(id)}?mode=${encodeURIComponent(mode)}`;
export const pct = value => `${Math.round((value || 0) * 100)}%`;
export const date = value => value ? new Date(value * 1000).toLocaleDateString(undefined, {month:'short', day:'numeric'}) : '—';
export const status = q => q.stats?.solves ? 'solved' : q.stats?.attempts || q.has_draft ? 'progress' : 'new';

const paths = {
  library:'M4 5h6v14H4z M14 5h6v14h-6z M7 8v3 M17 8v3',
  layers:'m12 3 9 5-9 5-9-5 9-5Z M3 12l9 5 9-5 M3 16l9 5 9-5',
  clock:'M12 8v5l3 2 M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0',
  route:'M5 6h11a3 3 0 0 1 0 6H8a3 3 0 0 0 0 6h11 M16 15l3 3-3 3 M5 3v6',
  terminal:'m5 7 5 5-5 5 M13 17h6',
  search:'M16 16l5 5 M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0',
  arrow:'M4 12h15 m-6-6 6 6-6 6',
  back:'M20 12H5 m6-6-6 6 6 6',
  chevron:'m9 5 7 7-7 7',
  check:'m5 12 4 4L19 6',
  circle:'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0',
  target:'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0 M17 12a5 5 0 1 1-10 0 5 5 0 0 1 10 0 M12 11v2',
  play:'m8 5 11 7-11 7V5Z',
  pause:'M8 5v14 M16 5v14',
  next:'m5 5 10 7-10 7V5Z M19 5v14',
  previous:'m19 5-10 7 10 7V5Z M5 5v14',
  book:'M12 5c-3-2-7-2-10-1v15c3-1 7-1 10 1 3-2 7-2 10-1V4c-3-1-7-1-10 1Zm0 0v15',
  code:'m8 6-6 6 6 6 M16 6l6 6-6 6 M14 3l-4 18',
  bulb:'M9 18h6 M10 21h4 M8 14c-5-5-1-11 4-11s9 6 4 11l-1 2H9l-1-2Z',
  refresh:'M20 7v5h-5 M4 17v-5h5 M5 7a8 8 0 0 1 14 0 M19 17a8 8 0 0 1-14 0',
  menu:'M4 6h16 M4 12h16 M4 18h16',
  close:'m6 6 12 12 M6 18 18 6',
  alert:'m12 3 10 18H2L12 3Z M12 9v5 M12 17v1',
  spark:'m12 3 2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4L12 3Z',
  lock:'M6 10h12v11H6z M8 10V6a4 4 0 0 1 8 0v4 M12 14v3',
  box:'m12 3 9 5v9l-9 5-9-5V8l9-5Z M3 8l9 5 9-5 M12 13v9 M8 5l9 5',
  external:'M14 3h7v7 M21 3l-11 11 M10 3H3v18h18v-7',
};
export const icon = (name, cls = '') => `<svg class="icon ${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${paths[name] || paths.circle}"/></svg>`;

export async function api(path, body, signal) {
  let response;
  try {
    response = await fetch(`/api${path}`, {method:body === undefined ? 'GET' : 'POST',
      headers:body === undefined ? {} : {'Content-Type':'application/json'},
      body:body === undefined ? undefined : JSON.stringify(body), signal});
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('Cannot reach the local server. Check that LLM by Hand is running.');
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status}).`);
  return data;
}

let toastTimer;
export function toast(text, error = false) {
  const node = $('#toast');
  node.textContent = text;
  node.className = `toast visible ${error ? 'error' : ''}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove('visible'), 4500);
}

export function dialog({title, body, symbol = 'spark', confirm = 'Continue', cancel = 'Cancel', danger = false}) {
  const node = $('#app-dialog');
  if (node.open) return Promise.resolve(false);
  $('#dialog-title').textContent = title;
  $('#dialog-body').innerHTML = body;
  $('#dialog-icon').innerHTML = icon(symbol);
  $('#dialog-confirm').textContent = confirm;
  $('#dialog-confirm').className = `button ${danger ? 'danger' : 'primary'}`;
  $('#dialog-cancel').textContent = cancel;
  $('#dialog-cancel').hidden = !cancel;
  node.returnValue = '';
  return new Promise(resolve => {
    node.addEventListener('close', () => resolve(node.returnValue === 'confirm'), {once:true});
    node.showModal();
  });
}

// The bank uses a small Markdown subset. HTML is never accepted; links are http(s)-only.
function inline(text) {
  const tokens = /(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^\s)]+\))/g;
  let output = '', cursor = 0;
  for (const match of text.matchAll(tokens)) {
    output += escape(text.slice(cursor, match.index));
    const token = match[0];
    if (token.startsWith('`')) output += `<code>${escape(token.slice(1, -1))}</code>`;
    else if (token.startsWith('**')) output += `<strong>${escape(token.slice(2, -2))}</strong>`;
    else {
      const [, label, url] = token.match(/^\[([^\]]+)\]\((.*)\)$/);
      output += /^https?:\/\//i.test(url) ? `<a href="${escape(url)}" target="_blank" rel="noopener noreferrer">${escape(label)}</a>` : escape(label);
    }
    cursor = match.index + token.length;
  }
  return output + escape(text.slice(cursor));
}

export function markdown(text = '') {
  const lines = String(text).replace(/\r/g, '').split('\n');
  let html = '', paragraph = [], list = '', code = null, itemOpen=false;
  const flush = () => {
    if (paragraph.length) html += `<p>${inline(paragraph.join(' '))}</p>`;
    paragraph = [];
    if (list) { html += `${itemOpen ? '</li>' : ''}</${list}>`; list = ''; itemOpen=false; }
  };
  for (const line of lines) {
    if (/^\s*```/.test(line)) {
      if (code !== null) { html += `<pre><code>${escape(code.join('\n'))}</code></pre>`; code = null; }
      else { flush(); code = []; }
      continue;
    }
    if (code !== null) { code.push(line); continue; }
    const heading = line.match(/^(#{1,6})\s+(.+)/);
    const item = line.match(/^\s*(?:[-*]|\d+\.)\s+(.+)/);
    if (!line.trim()) { flush(); continue; }
    if (heading) { flush(); const level = Math.min(heading[1].length + 1, 6); html += `<h${level}>${inline(heading[2])}</h${level}>`; }
    else if (item) {
      const tag = /^\s*\d+\./.test(line) ? 'ol' : 'ul';
      if (list !== tag) { flush(); html += `<${tag}>`; list = tag; }
      else if (itemOpen) html += '</li>';
      html += `<li>${inline(item[1])}`; itemOpen=true;
    } else if (list && /^\s+\S/.test(line)) {
      html += ` ${inline(line.trim())}`;
    } else if (/^>\s?/.test(line)) { flush(); html += `<blockquote>${inline(line.replace(/^>\s?/, ''))}</blockquote>`; }
    else { if (list) flush(); paragraph.push(line.trim()); }
  }
  flush();
  if (code !== null) html += `<pre><code>${escape(code.join('\n'))}</code></pre>`;
  return html;
}

export const empty = (title, description, symbol = 'search') => `<div class="empty-state">${icon(symbol)}<h3>${escape(title)}</h3><p>${escape(description)}</p></div>`;
export const difficulty = q => `<span class="difficulty difficulty-${q.difficulty <= 2 ? 'easy' : q.difficulty <= 3 ? 'medium' : 'hard'}" title="Difficulty ${q.difficulty} of 5"><span class="difficulty-bars" aria-hidden="true">${[1,2,3].map((v, i) => `<i style="height:${7+i*4}px" class="${v <= Math.ceil(q.difficulty * 3 / 5) ? 'filled' : ''}"></i>`).join('')}</span>${q.difficulty <= 2 ? 'Foundational' : q.difficulty <= 3 ? 'Intermediate' : 'Advanced'}</span>`;

// Storage can be unavailable in private browsing. The server remains the source of truth.
export const storage = {
  get(key) { try { return localStorage.getItem(`qk:${key}`); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(`qk:${key}`, value); return true; } catch { return false; } },
  remove(key) { try { localStorage.removeItem(`qk:${key}`); } catch { /* unavailable */ } },
};
