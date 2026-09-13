const $ = id => document.getElementById(id);
let controller;
let runFeff;
let downloadUrls = [];

function status(text, state = '') {
  $('status').textContent = text;
  $('status').dataset.state = state;
}

function clearResults() {
  downloadUrls.forEach(url => URL.revokeObjectURL(url));
  downloadUrls = [];
  $('downloads').replaceChildren();
  $('plot').replaceChildren(Object.assign(document.createElement('p'), {
    textContent: 'χ(k) will appear here after the calculation.',
  }));
}

function download(name, bytes, type = 'application/octet-stream') {
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  downloadUrls.push(url);
  const link = Object.assign(document.createElement('a'), { href: url, download: name, textContent: `Download ${name}` });
  $('downloads').append(link);
}

function plotChi(bytes) {
  const points = new TextDecoder().decode(bytes).split('\n')
    .filter(line => line.trim() && !line.trimStart().startsWith('#'))
    .map(line => line.trim().split(/\s+/).slice(0, 2).map(Number))
    .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (points.length < 2) return;
  const xs = points.map(([x]) => x), ys = points.map(([, y]) => y);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const x = value => 65 + (value - xmin) / (xmax - xmin || 1) * 570;
  const y = value => 30 + (ymax - value) / (ymax - ymin || 1) * 230;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 660 320');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', `EXAFS chi spectrum with ${points.length} points`);
  function add(name, attributes, text) {
    const element = document.createElementNS(svg.namespaceURI, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    if (text !== undefined) element.textContent = text;
    svg.append(element);
  }
  for (let tick = 0; tick <= 4; tick++) {
    const value = ymin + (ymax - ymin) * tick / 4;
    add('line', { x1: 65, x2: 635, y1: y(value), y2: y(value), stroke: '#dfe6f0' });
    add('text', { x: 56, y: y(value) + 5, 'text-anchor': 'end', fill: '#53657c', 'font-size': 14 }, value.toPrecision(2));
  }
  for (let tick = 0; tick <= 4; tick++) {
    const value = xmin + (xmax - xmin) * tick / 4;
    add('text', { x: x(value), y: 282, 'text-anchor': 'middle', fill: '#53657c', 'font-size': 14 }, value.toFixed(1));
  }
  add('text', { x: 20, y: 20, fill: '#18273c', 'font-size': 16 }, 'χ(k)');
  add('text', { x: 350, y: 310, 'text-anchor': 'middle', fill: '#18273c', 'font-size': 16 }, 'k (Å⁻¹)');
  add('polyline', { points: points.map(([a, b]) => `${x(a)},${y(b)}`).join(' '),
    fill: 'none', stroke: '#2458d6', 'stroke-width': 2, 'stroke-linejoin': 'round' });
  $('plot').replaceChildren(svg);
}

async function loadSample() {
  const response = await fetch('/sample.inp');
  if (!response.ok) throw new Error(`Cannot load example: HTTP ${response.status}`);
  $('input').value = await response.text();
  status('Ready');
}

async function calculate() {
  if (controller) throw new Error('A calculation is already running.');
  if (!runFeff) throw new Error('The WebAssembly adapter has not loaded.');
  if (!$('input').value.trim()) throw new Error('Enter a FEFF input first.');
  controller = new AbortController();
  for (const id of ['run', 'sample', 'file', 'input']) $(id).disabled = true;
  $('cancel').disabled = false;
  clearResults();
  $('log').textContent = '';
  $('summary').textContent = 'Calculating…';
  status('Running');
  const start = performance.now();
  try {
    const result = await runFeff({
      input: $('input').value,
      signal: controller.signal,
      onLog: ({ stream, text }) => {
        if (stream === 'stderr') {
          $('log').textContent += text;
          $('log').scrollTop = $('log').scrollHeight;
        }
      },
    });
    if (result.exitCode !== 0) throw new Error(result.report.error.message);
    const seconds = ((performance.now() - start) / 1000).toFixed(1);
    $('summary').textContent = `${result.report.data.atoms} atoms · ${Object.keys(result.files).length} output files · ${seconds} seconds`;
    if (result.files['chi.dat']) plotChi(result.files['chi.dat']);
    else $('plot').textContent = 'This input completed without an EXAFS χ(k) spectrum.';
    for (const name of ['chi.dat', 'xmu.dat']) {
      if (result.files[name]) download(name, result.files[name], 'text/plain');
    }
    download('report.json', JSON.stringify(result.report, null, 2), 'application/json');
    status('Complete', 'complete');
    return { status: 'complete', atoms: result.report.data.atoms, files: Object.keys(result.files) };
  } catch (error) {
    const cancelled = error.name === 'AbortError';
    status(cancelled ? 'Cancelled' : 'Failed', cancelled ? '' : 'error');
    $('summary').textContent = cancelled ? 'Calculation cancelled. You can edit the input and run again.' : error.message;
    throw error;
  } finally {
    controller = undefined;
    for (const id of ['run', 'sample', 'file', 'input']) $(id).disabled = false;
    $('cancel').disabled = true;
  }
}

$('run').onclick = () => calculate().catch(error => {
  if (error.name !== 'AbortError') { status('Failed', 'error'); $('summary').textContent = error.message; }
});
$('cancel').onclick = () => controller?.abort();
$('sample').onclick = () => loadSample().catch(error => { status('Failed', 'error'); $('summary').textContent = error.message; });
$('file').onchange = async event => {
  const file = event.target.files[0];
  if (!file) return;
  try { $('input').value = await file.text(); status('Ready'); }
  catch (error) { status('Failed', 'error'); $('summary').textContent = error.message; }
};

try {
  ({ runFeff } = await import('../dist/index.mjs'));
  await loadSample();
  $('run').disabled = false;
} catch (error) {
  status('Setup needed', 'error');
  $('summary').textContent = `${error.message}. Run npm run build --prefix wasm, then reload.`;
}

// Use the same action for browsers exposing the optional WebMCP registry.
if (document.modelContext?.registerTool && runFeff) {
  const lifecycle = new AbortController();
  addEventListener('pagehide', () => lifecycle.abort(), { once: true });
  try {
    await document.modelContext.registerTool({
      name: 'run_feff_calculation',
      description: 'Run the FEFF input currently shown in the editor and display its spectrum and downloads.',
      inputSchema: { type: 'object', properties: {}, additionalProperties: false },
      annotations: { readOnlyHint: false, untrustedContentHint: true },
      execute: input => {
        if (!input || typeof input !== 'object' || Array.isArray(input) || Object.keys(input).length) {
          throw new TypeError('Expected an empty object.');
        }
        return calculate();
      },
    }, { signal: lifecycle.signal });
  } catch (error) { console.warn('Optional browser tool could not register:', error); }
}
