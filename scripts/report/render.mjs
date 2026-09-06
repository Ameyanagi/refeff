// Pure presentation helpers. Numerical comparisons always use all source rows.
export function matrixStatus(report) {
  const cases = report.cases ?? [];
  const primary = cases.length > 0 && cases.every(item => item.comparison.passed);
  const workflows = report.workflows ?? [];
  const expected = report.expectedWorkflows ?? [];
  const provenance = report.workflowProvenance;
  const matching = provenance && provenance.dirty === false && report.provenance?.dirty === false && ["rustCommit", "feffCommit", "rustBinarySha256", "feffDriverSha256"].every(key =>
    typeof provenance[key] === "string" && provenance[key] === report.provenance?.[key]);
  const ids = new Set(workflows.map(item => item.id));
  const complete = expected.length > 0 && ids.size === workflows.length
    && expected.length === workflows.length && expected.every(id => ids.has(id)) && matching;
  if (!primary) return { status: "review", label: "Review primary spectra" };
  if (!complete) return { status: "pending", label: "Primary spectra pass; matrix not evaluated" };
  if (workflows.some(item => (item.status ?? (item.passed ? "pass" : "fail")) !== "pass"))
    return { status: "review", label: "Primary spectra pass; matrix needs review" };
  return { status: "pass", label: "Release matrix passes" };
}
export function extent(values) {
  let min = Infinity, max = -Infinity;
  for (const value of values) {
    if (Number.isFinite(value)) {
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
  }
  return [min, max];
}
// Preserve extrema in each bucket so narrow peaks survive display decimation.
export function displayIndices(values, limit = 1200) {
  if (values.length <= limit) return values.map((_, index) => index);
  const stride = Math.ceil(values.length / Math.max(1, (limit - 2) / 2));
  const indices = [0];
  for (let begin = 1; begin < values.length - 1; begin += stride) {
    let min = begin, max = begin;
    for (let i = begin + 1; i < Math.min(begin + stride, values.length - 1); i++) {
      if (values[i] < values[min]) min = i;
      if (values[i] > values[max]) max = i;
    }
    indices.push(...new Set([min, max].sort((a,b) => a-b)));
  }
  indices.push(values.length - 1);
  return indices;
}
export function formatSpeed(value) {
  if (!(value > 0) || !Number.isFinite(value)) return "not measured";
  return value >= 1 ? value.toFixed(2) + "× faster" : (1/value).toFixed(2) + "× slower";
}
const SPECTRUM_RELATIVE_TOLERANCE = 5e-5;
export function renderHtml(report) {
  const serialized = JSON.stringify(report).replaceAll("<", "\\u003c");
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FEFF → Rust parity lab</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #08101d;
      --panel: rgba(16, 27, 46, 0.88);
      --panel-2: #101c30;
      --line: #263853;
      --text: #e7eefb;
      --muted: #94a7c3;
      --blue: #4d7cff;
      --coral: #ec6a5c;
      --green: #51c39a;
      --amber: #e5b85c;
      --shadow: 0 20px 70px rgba(0, 0, 0, 0.28);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 15% 0%, rgba(77,124,255,.18), transparent 30rem),
        radial-gradient(circle at 90% 10%, rgba(81,195,154,.12), transparent 28rem),
        var(--bg);
      color: var(--text);
      min-height: 100vh;
    }
    main { width: min(1440px, calc(100% - 40px)); margin: 0 auto; padding: 52px 0 80px; }
    header { display: grid; grid-template-columns: 1.4fr .6fr; gap: 32px; align-items: end; margin-bottom: 34px; }
    .eyebrow { color: var(--green); font-size: 12px; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
    h1 { margin: 10px 0 12px; font-size: clamp(42px, 7vw, 82px); line-height: .96; letter-spacing: -.055em; }
    .lede { color: var(--muted); max-width: 760px; font-size: 17px; line-height: 1.65; }
    .stamp { justify-self: end; color: var(--muted); font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; text-align: right; }
    .status {
      display: inline-flex; align-items: center; gap: 8px; padding: 7px 11px; border-radius: 999px;
      background: rgba(81,195,154,.12); color: #79dbb8; border: 1px solid rgba(81,195,154,.3);
      font-size: 12px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase;
    }
    .status::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; box-shadow: 0 0 12px currentColor; }
    .status.review { background: rgba(229,184,92,.12); color: var(--amber); border-color: rgba(229,184,92,.3); }
    .status.pending { background: rgba(148,167,195,.1); color: var(--muted); border-color: rgba(148,167,195,.24); }
    .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 26px 0 44px; }
    .kpi, .panel {
      background: linear-gradient(145deg, rgba(19,33,56,.94), rgba(12,22,39,.94));
      border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow);
    }
    .kpi { padding: 20px; min-height: 128px; }
    .kpi .label { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
    .kpi .value { display: block; margin-top: 12px; font-size: 34px; font-weight: 760; letter-spacing: -.04em; }
    .kpi .note { color: var(--muted); font-size: 12px; margin-top: 5px; }
    section { margin-top: 50px; }
    .section-head { display: flex; justify-content: space-between; gap: 22px; align-items: end; margin-bottom: 18px; }
    h2 { margin: 0; font-size: 27px; letter-spacing: -.025em; }
    .section-head p { margin: 0; max-width: 650px; color: var(--muted); line-height: 1.55; font-size: 14px; text-align: right; }
    .case-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    .gallery-toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 16px; }
    .gallery-filter {
      appearance: none; border: 1px solid var(--line); border-radius: 999px; padding: 8px 12px;
      background: rgba(12,22,39,.9); color: var(--muted); cursor: pointer; font: inherit;
      font-size: 12px; font-weight: 700;
    }
    .gallery-filter.active { color: var(--text); border-color: rgba(77,124,255,.7); background: rgba(77,124,255,.16); }
    .gallery-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    .gallery-card { padding: 16px; content-visibility: auto; contain-intrinsic-size: 520px; }
    .gallery-card .chart { min-height: 205px; }
    .gallery-card .residual { min-height: 105px; margin-top: 8px; }
    .gallery-card .legend { min-height: 28px; }
    .gallery-metrics {
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px;
      color: var(--muted); font: 10px ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .gallery-metrics strong { display: block; margin-top: 3px; color: var(--text); font-size: 12px; }
    .gallery-note { color: var(--muted); font-size: 12px; line-height: 1.55; margin: -4px 0 16px; }
    .panel { padding: 22px; overflow: hidden; }
    .panel-head { display: flex; justify-content: space-between; gap: 16px; align-items: start; margin-bottom: 14px; }
    h3 { margin: 0; font-size: 19px; }
    .subtitle { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .chart { width: 100%; min-height: 310px; border-radius: 12px; background: rgba(5,12,23,.48); border: 1px solid rgba(73,98,132,.28); }
    .residual { min-height: 170px; margin-top: 12px; }
    svg { display: block; width: 100%; height: 100%; overflow: visible; }
    .axis { stroke: #435875; stroke-width: 1; }
    .grid { stroke: #253852; stroke-width: 1; stroke-dasharray: 3 6; }
    .tick { fill: #89a0be; font-size: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .legend { display: flex; flex-wrap: wrap; gap: 13px; margin: 12px 0 0; color: var(--muted); font-size: 11px; }
    .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 18px; height: 3px; border-radius: 3px; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 12px; }
    th { color: var(--muted); font-weight: 650; text-align: left; padding: 9px 8px; border-bottom: 1px solid var(--line); }
    td { padding: 9px 8px; border-bottom: 1px solid rgba(38,56,83,.62); font-variant-numeric: tabular-nums; }
    td.number, th.number { text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .pass { color: var(--green); font-weight: 750; }
    .fail { color: var(--coral); font-weight: 750; }
    .review { color: var(--amber); font-weight: 750; }
    .pending { color: var(--muted); font-weight: 750; }
    .performance { display: grid; grid-template-columns: 1.15fr .85fr; gap: 18px; }
    .workflow-summary { overflow-x: auto; }
    .workflow-summary table { min-width: 720px; margin-top: 0; }
    .bars { display: grid; gap: 22px; margin-top: 20px; }
    .bar-row { display: grid; grid-template-columns: 125px 1fr 84px; gap: 12px; align-items: center; }
    .bar-label { color: var(--muted); font-size: 12px; }
    .bar-track { position: relative; height: 30px; background: #07101d; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    .bar-fill { height: 100%; min-width: 3px; border-radius: 7px; }
    .bar-value { text-align: right; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .sample-dots { display: flex; align-items: center; gap: 5px; margin: 8px 0 0 137px; }
    .sample-dots i { display: block; width: 7px; height: 7px; border-radius: 50%; background: #6f83a0; }
    dl { display: grid; grid-template-columns: max-content 1fr; gap: 10px 16px; margin: 0; font-size: 12px; }
    dt { color: var(--muted); }
    dd { margin: 0; overflow-wrap: anywhere; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    details { margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; line-height: 1.6; }
    summary { color: var(--text); cursor: pointer; font-weight: 700; }
    a { color: #8da9ff; }
    footer { color: var(--muted); margin-top: 54px; font-size: 11px; display: flex; justify-content: space-between; gap: 20px; }
    @media (max-width: 960px) {
      header, .performance { grid-template-columns: 1fr; }
      .stamp { justify-self: start; text-align: left; }
      .kpis { grid-template-columns: 1fr 1fr; }
      .case-grid { grid-template-columns: 1fr; }
      .gallery-grid { grid-template-columns: 1fr 1fr; }
      .section-head { align-items: start; flex-direction: column; }
      .section-head p { text-align: left; }
    }
    @media (max-width: 560px) {
      main { width: min(100% - 22px, 1440px); padding-top: 28px; }
      .kpis { grid-template-columns: 1fr; }
      .gallery-grid { grid-template-columns: 1fr; }
      .bar-row { grid-template-columns: 95px 1fr 64px; }
      .sample-dots { margin-left: 107px; }
    }
    .table-scroll { overflow-x:auto; max-width:100%; }
    .plot-tools { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:8px; font-size:12px; }
    .plot-tools button { padding:6px 10px; border:1px solid var(--line); border-radius:6px; background:var(--panel-2); color:inherit; }
    .plot-tools output { width:100%; overflow-wrap:anywhere; }
    button, summary, a { touch-action:manipulation; }
    button { cursor:pointer; }
    button:focus-visible, a:focus-visible, summary:focus-visible, input:focus-visible { outline:3px solid #ffcd66; outline-offset:3px; }
    .status.fail, .status.review { color:#ffb090; background:#442322; border-color:#cc755d; }
    .status.pending { color:#ffde8c; background:#40371c; border-color:#a38a46; }
    .chart { overflow-x:auto; }
    .chart svg { min-width:360px; }
    .tick { font-size:12px; }
    @media print {
      :root { color-scheme:light; --bg:white; --panel:white; --panel-2:white; --muted:#444; --line:#bbb; }
      body, .panel, main { background:white !important; color:black !important; }
      .plot-tools, .gallery-filters { display:none; }
      .case-grid, .gallery-grid { display:block; }
      .panel { break-inside:avoid; margin:12px 0; box-shadow:none; }
      .table-scroll { overflow:visible; }
      .gallery-card { content-visibility:visible; }
      svg { color:black; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <div class="eyebrow">Numerical parity · release performance</div>
        <h1>FEFF → Rust<br>parity lab</h1>
        <p class="lede">Fresh, side-by-side spectrum outputs from the pure-Rust release build and the local sequential FEFF reference, measured on the same machine and visualized without external chart libraries.</p>
      </div>
      <div class="stamp">
        <span class="status" id="overall-status">Loading</span>
        <p id="generated-at"></p>
        <p>Raw measurements: <a href="report.json">report.json</a></p>
      </div>
    </header>

    <div class="kpis" id="kpis"></div>

    <section>
      <div class="section-head">
        <div><div class="eyebrow">01 · output parity</div><h2>Spectra on top of each other</h2></div>
        <p>Each panel shows one physical observable: signed χ(k) for EXAFS and non-negative μ(E) absorption for XANES. Solid lines are FEFF; dashed lines are Rust. Lower panels show Rust − FEFF residuals.</p>
      </div>
      <div class="case-grid" id="cases"></div>
    </section>

    <section id="gallery-section" hidden>
      <div class="section-head">
        <div><div class="eyebrow">02 · frozen artifact gallery</div><h2>Other registered output plots</h2></div>
        <p>Workflow-aware observables from the last available FEFF/Rust artifact pairs. These plots are frozen evidence; modified source now postdates the artifacts, so they are not presented as a current-checkout rerun.</p>
      </div>
      <p class="gallery-note" id="gallery-note" aria-live="polite"></p>
      <div class="gallery-toolbar" id="gallery-filters" aria-label="Filter plot gallery"></div>
      <div class="gallery-grid" id="gallery"></div>
    </section>

    <section>
      <div class="section-head">
        <div><div class="eyebrow">03 · performance</div><h2>Release build vs sequential FEFF</h2></div>
        <p>Median wall-clock time after one discarded warm-up. Both complete workflows use one thread and fresh output directories; dots show individual timed samples.</p>
      </div>
      <div class="performance">
        <div class="panel">
          <div class="panel-head"><div><h3>Full workflow runtime</h3><div class="subtitle">Lower is better · median seconds</div></div></div>
          <div class="bars" id="performance-bars"></div>
        </div>
        <div class="panel">
          <div class="panel-head"><div><h3>Input-stage throughput</h3><div class="subtitle">44 FEFF examples × 5 iterations</div></div></div>
          <div id="input-stage"></div>
          <details open>
            <summary>Machine and binary provenance</summary>
            <dl id="provenance" style="margin-top:14px"></dl>
          </details>
        </div>
      </div>
    </section>

    <section id="workflow-section" hidden>
      <div class="section-head">
        <div><div class="eyebrow">04 · full method matrix</div><h2>Every supported stock workflow</h2></div>
        <p>Frozen Rust release results against provenance-tracked FEFF10 references. Long workflows are timed independently so their cost remains visible.</p>
      </div>
      <div class="panel workflow-summary">
        <table>
          <thead><tr><th>Workflow</th><th>Status</th><th class="number">Elapsed</th><th>Evidence</th></tr></thead>
          <tbody id="workflow-rows"></tbody>
        </table>
      </div>
    </section>

    <section>
      <div class="panel">
        <div class="section-head">
          <div><div class="eyebrow">05 · methodology</div><h2>How to read this report</h2></div>
          <p>This is a local benchmark snapshot, not a claim about all hardware or compiler configurations.</p>
        </div>
        <dl id="method"></dl>
      </div>
    </section>

    <footer><span>Generated by scripts/feff-visual-report.mjs</span><span id="commit-footer"></span></footer>
  </main>
  <script>
    const report = ${serialized};
    const formatSeconds = value => value < .01 ? (value * 1000).toFixed(2) + " ms" : value.toFixed(3) + " s";
    const formatMetric = value => value === 0 ? "0" : value.toExponential(3);
    const formatSpeed = ${formatSpeed.toString()};
    const extent = ${extent.toString()};
    const displayIndices = ${displayIndices.toString()};
    const escapeHtml = value => String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
    const workflowResults = report.workflows ?? [];
    const workflowStatus = item => item.status ?? (item.passed ? "pass" : "fail");
    const workflowCounts = workflowResults.reduce((counts, item) => {
      const status = workflowStatus(item);
      counts[status] = (counts[status] ?? 0) + 1;
      return counts;
    }, {});
    const overall = ${JSON.stringify(matrixStatus(report))};
    document.getElementById("overall-status").textContent = overall.label;
    document.getElementById("overall-status").className = "status " + overall.status;
    document.getElementById("generated-at").textContent = new Date(report.generatedAt).toLocaleString();
    document.getElementById("commit-footer").textContent = "Rust " + report.provenance.rustCommit.slice(0, 10) + " · FEFF " + report.provenance.feffCommit.slice(0, 10);

    const xanes = report.cases.find(item => item.id === "XANES/BN");
    const exafs = report.cases.find(item => item.id === "EXAFS/Cu");
    const kpis = [
      ["Parity", workflowResults.length
        ? (workflowCounts.pass ?? 0) + "/" + workflowResults.length
        : report.cases.filter(item => item.comparison.passed).length + "/" + report.cases.length,
       workflowResults.length
         ? (workflowCounts.review ?? 0) + " review · " + (workflowCounts.pending ?? 0) + " pending"
         : "primary spectrum outputs"],
      ["Max relative L2", formatMetric(extent(report.cases.map(item => item.comparison.maxRelativeL2))[1]), "registered limit " + formatMetric(${SPECTRUM_RELATIVE_TOLERANCE})],
      ["RDINP speed", formatSpeed(report.inputStage.speedup), formatSeconds(report.inputStage.rust.averageSeconds) + " Rust / run"],
      ["BN XANES", formatSpeed(xanes?.benchmark.speedup), "full pipeline median"],
    ];
    document.getElementById("kpis").innerHTML = kpis.map(([label, value, note]) =>
      '<div class="kpi"><div class="label">' + label + '</div><span class="value">' + value + '</span><div class="note">' + note + '</div></div>'
    ).join("");

    const failures = document.createElement("nav"); failures.setAttribute("aria-label", "Review findings");
    failures.innerHTML = report.cases.filter(item => !item.comparison.passed).map(item => '<a href="#case-' + escapeHtml(item.id.replaceAll(/[^a-z0-9_-]/gi,"-")) + '">Review ' + escapeHtml(item.title) + '</a>').join(' · ');
    if (workflowResults.some(item => workflowStatus(item) !== "pass")) failures.innerHTML += ' <a href="#workflow-section">Review workflow evidence</a>';
    document.getElementById("cases").before(failures);
    const caseRoot = document.getElementById("cases");
    for (const item of report.cases) {
      const panel = document.createElement("article");
      panel.className = "panel";
      panel.id = "case-" + item.id.replaceAll(/[^a-z0-9_-]/gi,"-");
      const rows = item.comparison.columns.map(column =>
        '<tr><td>' + column.name + '</td><td class="number">' + formatMetric(column.relativeL2) +
        '</td><td class="number">' + formatMetric(column.maxAbsolute) +
        '</td><td class="' + (column.passed ? "pass" : "fail") + '">' + (column.passed ? "PASS" : "FAIL") + '</td></tr>'
      ).join("") + (item.comparison.physicalChecks ?? []).map(check =>
        '<tr><td>' + check.name + '</td><td class="number">—</td><td class="number">' +
        formatMetric(check.maximumViolation) + '</td><td class="' + (check.passed ? "pass" : "fail") + '">' +
        (check.passed ? "PASS" : "FAIL") + '</td></tr>'
      ).join("");
      panel.innerHTML =
        '<div class="panel-head"><div><h3>' + escapeHtml(item.title) + '</h3><div class="subtitle">' + item.subtitle + ' · ' + item.comparison.rows + ' rows</div></div>' +
        '<span class="status ' + (item.comparison.passed ? 'pass' : 'review') + '">' + (item.comparison.passed ? "Pass" : "Review") + '</span></div>' +
        '<div class="chart"></div><div class="legend"></div><div class="chart residual"></div>' +
        '<table><thead><tr><th>Column</th><th class="number">Relative L2</th><th class="number">Max |Δ|</th><th>Status</th></tr></thead><tbody>' + rows + '</tbody></table>' +
        '<details><summary>Run files and timing spread</summary><p>FEFF: <code>' + item.files.feff + '</code><br>Rust: <code>' + item.files.rust +
        '</code></p><p>FEFF median ' + formatSeconds(item.benchmark.feff.medianSeconds) + ' (σ ' + item.benchmark.feff.standardDeviationSeconds.toFixed(3) +
        's); Rust median ' + formatSeconds(item.benchmark.rust.medianSeconds) + ' (σ ' + item.benchmark.rust.standardDeviationSeconds.toFixed(3) + 's).</p></details>';
      caseRoot.appendChild(panel);
      drawOverlay(panel.querySelector(".chart"), item);
      drawResidual(panel.querySelector(".residual"), item);
      panel.querySelector(".legend").innerHTML = item.comparison.series.map(series =>
        '<span><i class="swatch" style="background:' + series.color + '"></i>' + series.label + ' · FEFF solid / Rust dashed</span>'
      ).join("");
    }

    const galleryCases = report.galleryCases ?? [];
    if (galleryCases.length) {
      document.getElementById("gallery-section").hidden = false;
      const galleryRoot = document.getElementById("gallery");
      const galleryFamilies = [...new Set(galleryCases.map(item => item.family))].sort();
      const galleryWorkflows = new Set(galleryCases.map(item => item.workflow)).size;
      document.getElementById("gallery-note").textContent =
        galleryCases.length + " additional registered output plots across " + galleryWorkflows +
        " workflows. Solid lines are FEFF; dashed lines are Rust. REVIEW status remains authoritative even when curves appear close.";
      const filterItems = [
        ["all", "All · " + galleryCases.length],
        ["review", "Review · " + galleryCases.filter(item => item.status === "review").length],
        ...galleryFamilies.map(family => [
          family,
          family + " · " + galleryCases.filter(item => item.family === family).length,
        ]),
      ];
      document.getElementById("gallery-filters").innerHTML = filterItems.map(([value, label], index) =>
        '<button class="gallery-filter' + (index === 0 ? " active" : "") +
        '" aria-pressed="' + (index === 0) + '" type="button" data-filter="' + escapeHtml(value) + '">' + escapeHtml(label) + '</button>'
      ).join("");

      const pendingCharts = new WeakMap();
      const renderCard = panel => {
        const item = pendingCharts.get(panel);
        if (!item || panel.hidden) return;
        drawOverlay(panel.querySelector(".chart"), item);
        drawResidual(panel.querySelector(".residual"), item);
        pendingCharts.delete(panel);
      };
      const observer = typeof IntersectionObserver === "undefined" ? null : new IntersectionObserver(entries => {
        for (const entry of entries) if (entry.isIntersecting) { renderCard(entry.target); observer.unobserve(entry.target); }
      }, {rootMargin: "300px"});
      window.addEventListener("beforeprint", () => { for (const panel of galleryRoot.children) renderCard(panel); });
      for (const item of galleryCases) {
        const panel = document.createElement("article");
        panel.className = "panel gallery-card";
        panel.dataset.family = item.family;
        panel.dataset.status = item.status;
        panel.innerHTML =
          '<div class="panel-head"><div><h3>' + escapeHtml(item.title) + '</h3><div class="subtitle">' +
          escapeHtml(item.output + " · " + item.subtitle + " · " + item.comparison.rows + " rows") +
          '</div></div><span class="status ' + escapeHtml(item.status) + '">' +
          escapeHtml(item.status) + '</span></div><div class="chart"></div><div class="legend"></div>' +
          '<div class="chart residual"></div><div class="gallery-metrics">' +
          '<div>plot rel L2<strong>' + formatMetric(item.plottedMaxRelativeL2) + '</strong></div>' +
          '<div>plot max |Δ|<strong>' + formatMetric(item.plottedMaxAbsolute) + '</strong></div>' +
          '<div>rows<strong>' + item.comparison.rows + '</strong></div></div>' +
          '<details><summary>Evidence and files</summary><p>' + escapeHtml(item.evidence) +
          '</p><p>FEFF: <code>' + escapeHtml(item.files.feff) + '</code><br>Rust: <code>' +
          escapeHtml(item.files.rust) + '</code></p></details>';
        galleryRoot.appendChild(panel);
        pendingCharts.set(panel, item);
        if (observer) observer.observe(panel); else renderCard(panel);
        panel.querySelector(".legend").innerHTML = item.comparison.series.map(series =>
          '<span><i class="swatch" style="background:' + escapeHtml(series.color) + '"></i>' +
          escapeHtml(series.label) + ' · FEFF solid / Rust dashed</span>'
        ).join("");
      }

      for (const button of document.querySelectorAll(".gallery-filter")) {
        button.addEventListener("click", () => {
          const filter = button.dataset.filter;
          for (const candidate of document.querySelectorAll(".gallery-filter")) {
            candidate.classList.toggle("active", candidate === button);
            candidate.setAttribute("aria-pressed", String(candidate === button));
          }
          let visible = 0;
          for (const card of document.querySelectorAll(".gallery-card")) {
            card.hidden = filter !== "all"
              && (filter === "review" ? card.dataset.status !== "review" : card.dataset.family !== filter);
            if (!card.hidden) { visible++; if (observer) observer.observe(card); else renderCard(card); }
          }
          document.getElementById("gallery-note").textContent = visible + " plots shown. Full-resolution data is available from each plot.";
        });
      }
    }

    const performanceRoot = document.getElementById("performance-bars");
    const maxRuntime = extent(report.cases.flatMap(item => [item.benchmark.rust.medianSeconds, item.benchmark.feff.medianSeconds]))[1];
    for (const item of report.cases) {
      const group = document.createElement("div");
      group.innerHTML = '<div style="font-weight:750;margin-bottom:10px">' + item.id + '<span style="color:var(--muted);font-weight:500;margin-left:8px">' + formatSpeed(item.benchmark.speedup) + '</span></div>';
      for (const [engine, color] of [["FEFF", "var(--coral)"], ["Rust", "var(--blue)"]]) {
        const key = engine.toLowerCase();
        const stats = item.benchmark[key];
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = '<div class="bar-label">' + engine + '</div><div class="bar-track"><div class="bar-fill" style="width:' +
          (stats.medianSeconds / maxRuntime * 100).toFixed(2) + '%;background:' + color + '"></div></div><div class="bar-value">' +
          formatSeconds(stats.medianSeconds) + '</div>';
        group.appendChild(row);
        const dots = document.createElement("div");
        dots.className = "sample-dots";
        dots.innerHTML = stats.samples.map((sample, index) =>
          '<i title="sample ' + (index + 1) + ': ' + sample.toFixed(4) + ' s" style="opacity:' + (.45 + .55 * sample / stats.maximumSeconds) + '"></i>'
        ).join("");
        group.appendChild(dots);
      }
      performanceRoot.appendChild(group);
    }

    const input = report.inputStage;
    document.getElementById("input-stage").innerHTML =
      '<div style="display:flex;align-items:end;gap:10px;margin:22px 0 8px"><strong style="font-size:44px;letter-spacing:-.05em">' +
      formatSpeed(input.speedup) + '</strong></div>' +
      '<table><tbody><tr><td>Rust</td><td class="number">' + formatSeconds(input.rust.averageSeconds) +
      '</td></tr><tr><td>FEFF</td><td class="number">' + formatSeconds(input.feff.averageSeconds) +
      '</td></tr><tr><td>Successful runs</td><td class="number">' + input.rust.successful + ' / ' + input.rust.runs + '</td></tr></tbody></table>';

    if (workflowResults.length) {
      document.getElementById("workflow-section").hidden = false;
      document.getElementById("workflow-rows").innerHTML = workflowResults.map(item => {
        const status = workflowStatus(item);
        return '<tr><td>' + escapeHtml(item.id) + '</td><td class="' + escapeHtml(status) + '">' +
        escapeHtml(status.toUpperCase()) + '</td><td class="number">' +
        (Number.isFinite(item.elapsedSeconds) ? formatSeconds(item.elapsedSeconds) : "—") +
        '</td><td>' + escapeHtml(item.detail ?? "") + '</td></tr>';
      }).join("");
    }

    const provenanceRows = [
      ["CPU", report.provenance.cpu],
      ["Logical cores", report.provenance.logicalCores],
      ["Memory", report.provenance.memoryGiB.toFixed(0) + " GiB"],
      ["Platform", report.provenance.platform],
      ["Rust", report.provenance.rustVersion],
      ["FEFF", report.provenance.feffVersion],
      ["Rust commit", report.provenance.rustCommit],
      ["FEFF commit", report.provenance.feffCommit],
    ];
    document.getElementById("provenance").innerHTML = provenanceRows.map(([key, value]) => '<dt>' + key + '</dt><dd>' + value + '</dd>').join("");
    document.getElementById("method").innerHTML = Object.entries(report.method).map(([key, value]) =>
      '<dt>' + key.replaceAll(/([A-Z])/g, " $1").replace(/^./, letter => letter.toUpperCase()) + '</dt><dd>' + value + '</dd>'
    ).join("");

    function drawOverlay(container, item) {
      drawChart(container, item, false);
    }
    function drawResidual(container, item) {
      drawChart(container, item, true);
    }
    function drawChart(container, item, residual) {
      const width = Math.max(360, Math.min(720, container.clientWidth || 720)), height = residual ? 210 : 330;
      const padding = { left: 58, right: 20, top: 34, bottom: 48 };
      const xValues = item.comparison.x.feff;
      const values = residual
        ? item.comparison.series.flatMap(series => series.residual)
        : item.comparison.series.flatMap(series => [...series.feff, ...series.rust]);
      const [xMin, xMax] = extent(xValues);
      let [yMin, yMax] = extent(values);
      if (![xMin, xMax, yMin, yMax].every(Number.isFinite)) { container.textContent = "No finite plot data"; return; }
      if (residual) {
        const extent = Math.max(Math.abs(yMin), Math.abs(yMax), 1e-16);
        yMin = -extent; yMax = extent;
      }
      const yPad = (yMax - yMin || 1) * .08;
      const nonNegative = !residual
        && item.comparison.series.every(series => series.nonNegative)
        && (item.comparison.physicalChecks ?? []).every(check => check.passed);
      yMin = nonNegative ? 0 : yMin - yPad;
      yMax += yPad;
      const sx = value => padding.left + (value - xMin) / (xMax - xMin || 1) * (width - padding.left - padding.right);
      const sy = value => padding.top + (yMax - value) / (yMax - yMin || 1) * (height - padding.top - padding.bottom);
      const pathFor = (xs, ys) => displayIndices(ys).map((index, point) => (point ? "L" : "M") + sx(xs[index]).toFixed(2) + "," + sy(ys[index]).toFixed(2)).join(" ");
      const xTicks = Array.from({length: width < 500 ? 4 : 6}, (_, index) => xMin + (xMax - xMin) * index / (width < 500 ? 3 : 5));
      const yTicks = Array.from({length: 5}, (_, index) => yMin + (yMax - yMin) * index / 4);
      const grid = [
        ...xTicks.map(value => '<line class="grid" x1="' + sx(value) + '" x2="' + sx(value) + '" y1="' + padding.top + '" y2="' + (height-padding.bottom) + '"/><text class="tick" x="' + sx(value) + '" y="' + (height-24) + '" text-anchor="middle">' + compact(value) + '</text>'),
        ...yTicks.map(value => '<line class="grid" x1="' + padding.left + '" x2="' + (width-padding.right) + '" y1="' + sy(value) + '" y2="' + sy(value) + '"/><text class="tick" x="' + (padding.left-8) + '" y="' + (sy(value)+3) + '" text-anchor="end">' + compact(value) + '</text>'),
      ].join("");
      const lines = item.comparison.series.map(series => residual
        ? '<path d="' + pathFor(xValues, series.residual) + '" fill="none" stroke="' + series.color + '" stroke-width="1.5"/>'
        : '<path d="' + pathFor(xValues, series.feff) + '" fill="none" stroke="' + series.color + '" stroke-width="2"/><path d="' +
          pathFor(item.comparison.x.rust, series.rust) + '" fill="none" stroke="' + series.color + '" stroke-width="1.5" stroke-dasharray="7 5" opacity=".9"/>'
      ).join("");
      container.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + width + ' ' + height + '" role="img" aria-label="' + escapeHtml(item.title) + (residual ? " residual" : " overlay") + '">' +
        '<style>.grid{stroke:#71829b;stroke-opacity:.25}.axis{stroke:#71829b}.tick{fill:#879ab7;font:12px sans-serif}</style>' + grid + '<line class="axis" x1="' + padding.left + '" x2="' + (width-padding.right) + '" y1="' + (height-padding.bottom) + '" y2="' + (height-padding.bottom) + '"/>' +
        (residual && yMin <= 0 && yMax >= 0 ? '<line x1="' + padding.left + '" x2="' + (width-padding.right) + '" y1="' + sy(0) + '" y2="' + sy(0) + '" stroke="#7e91aa" stroke-width="1"/>' : "") +
        lines + '<text class="tick" x="' + (width/2) + '" y="' + (height-6) + '" text-anchor="middle">' + escapeHtml(axisLabel(item)) + '</text><text x="12" y="16" font-size="12" fill="#879ab7">' + escapeHtml(residual ? 'Δ (Rust − FEFF)' : item.comparison.series.map(series => series.label).join(', ')) + '</text></svg>';
      addPlotTools(container, item, residual);
    }
    // Tables remain selectable and keyboard-scrollable on narrow screens.
    for (const table of document.querySelectorAll("table")) {
      const wrapper = document.createElement("div"); wrapper.className = "table-scroll";
      wrapper.tabIndex = 0; wrapper.setAttribute("role", "region"); wrapper.setAttribute("aria-label", "Report data table");
      table.replaceWith(wrapper); wrapper.appendChild(table);
    }
    function axisLabel(item) {
      const label = item.columns[item.xColumn];
      if (/energy|omega/i.test(label)) return label + " (eV)";
      if (/^k$|wave number/i.test(label)) return label + " (Å⁻¹)";
      return label; // Gallery-specific codecs already carry their own units.
    }
    function download(name, type, data) {
      const url = URL.createObjectURL(new Blob([data], {type}));
      const link = document.createElement("a"); link.href = url; link.download = name; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    function addPlotTools(container, item, residual) {
      const tools = document.createElement("div"); tools.className = "plot-tools";
      const name = item.id.replaceAll(/[^a-z0-9_-]/gi, "-") + (residual ? "-residual" : "-overlay");
      const button = (label, action) => { const node = document.createElement("button"); node.type = "button"; node.textContent = label; node.onclick = action; tools.appendChild(node); };
      button("SVG", () => download(name + ".svg", "image/svg+xml", container.querySelector("svg").outerHTML));
      button("PNG", () => {
        const svg = container.querySelector("svg");
        const url = URL.createObjectURL(new Blob([svg.outerHTML], {type:"image/svg+xml"}));
        const image = new Image(); image.onload = () => {
          const canvas = document.createElement("canvas"); canvas.width = svg.viewBox.baseVal.width * 2; canvas.height = svg.viewBox.baseVal.height * 2;
          const ctx = canvas.getContext("2d"); ctx.fillStyle = "#08101d"; ctx.fillRect(0,0,canvas.width,canvas.height); ctx.drawImage(image,0,0,canvas.width,canvas.height);
          canvas.toBlob(blob => { if (blob) download(name + ".png", "image/png", blob); }); URL.revokeObjectURL(url);
        }; image.onerror = () => URL.revokeObjectURL(url); image.src = url;
      });
      button("CSV · all points", () => {
        const csv = value => '"' + String(value).replaceAll('"','""') + '"';
        const columns = [axisLabel(item) + " FEFF", axisLabel(item) + " Rust", ...item.comparison.series.flatMap(series => [series.label + " FEFF", series.label + " Rust", series.label + " residual"])];
        const rows = item.comparison.x.feff.map((x,i) => [x,item.comparison.x.rust[i],...item.comparison.series.flatMap(series => [series.feff[i],series.rust[i],series.residual[i]])]);
        download(name + ".csv", "text/csv", [columns,...rows].map(row => row.map(csv).join(",")).join("\\n"));
      });
      const zoomLabel = document.createElement("label"); zoomLabel.textContent = "Zoom ";
      const zoom = document.createElement("input"); zoom.type = "range"; zoom.min = "1"; zoom.max = "5"; zoom.step = ".5"; zoom.value = "1";
      zoom.oninput = () => { container.querySelector("svg").style.width = (Number(zoom.value) * 100) + "%"; };
      zoomLabel.appendChild(zoom); tools.appendChild(zoomLabel);
      const pointLabel = document.createElement("label"); pointLabel.textContent = "Point ";
      const point = document.createElement("input"); point.type = "number"; point.min = "1"; point.max = item.comparison.x.feff.length; point.value = "1"; point.style.width = "6em";
      pointLabel.appendChild(point); tools.appendChild(pointLabel);
      const readout = document.createElement("output"); readout.setAttribute("aria-live", "polite");
      const inspect = () => { const value = Number(point.value); const i = Math.max(0, Math.min(item.comparison.x.feff.length-1, Number.isFinite(value) ? Math.trunc(value)-1 : 0)); readout.textContent = axisLabel(item) + " = " + item.comparison.x.feff[i] + "; " + item.comparison.series.map(series => series.label + ": FEFF " + series.feff[i] + ", Rust " + series.rust[i] + ", Δ " + series.residual[i]).join("; "); };
      point.oninput = inspect; inspect(); tools.appendChild(readout); container.appendChild(tools);
    }
    function compact(value) {
      const absolute = Math.abs(value);
      if ((absolute > 0 && absolute < .001) || absolute >= 10000) return value.toExponential(1);
      return Number(value.toFixed(absolute < 10 ? 3 : 1)).toString();
    }
  </script>
</body>
</html>`;
}
