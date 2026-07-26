#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";

const SPECTRUM_RELATIVE_TOLERANCE = 5e-5;
const SPECTRUM_ABSOLUTE_TOLERANCE = 5e-8;

const CASES = [
  {
    id: "EXAFS/Cu",
    title: "Cu K-edge EXAFS",
    subtitle: "Path expansion and χ(k)",
    output: "chi.dat",
    columns: ["k", "chi", "magnitude", "phase"],
    xColumn: 0,
    plots: [
      { column: 1, label: "χ(k)", color: "#4d7cff" },
      { column: 2, label: "|χ(k)|", color: "#ec6a5c" },
    ],
  },
  {
    id: "XANES/BN",
    title: "BN B K-edge XANES",
    subtitle: "SCF + 87-atom FMS spectrum",
    output: "xmu.dat",
    columns: ["photon energy", "relative energy", "wave number", "mu", "mu0", "chi"],
    xColumn: 0,
    plots: [
      { column: 3, label: "μ(E)", color: "#4d7cff" },
      { column: 5, label: "χ(E)", color: "#ec6a5c" },
    ],
  },
];

const args = parseArguments(process.argv.slice(2));
const root = path.resolve(args.root);
const outputRoot = path.resolve(root, args.output);
const sessionId = new Date().toISOString().replaceAll(/[:.]/g, "-");
const runRoot = path.join(outputRoot, "runs", sessionId);
const rustBinary = path.resolve(root, args.rustBinary);
const feffDriver = path.resolve(root, args.feffDriver);
const xtaskBinary = path.resolve(root, args.xtaskBinary);

await mkdir(runRoot, { recursive: true });

const provenance = collectProvenance();
const inputStage = runInputStageBenchmark();
const cases = [];

for (const definition of CASES) {
  console.log(`\n${definition.id}: warm-up`);
  await benchmarkRun(definition, "rust", "warmup");
  await benchmarkRun(definition, "feff", "warmup");

  const measured = { rust: [], feff: [] };
  for (let index = 0; index < args.iterations; index += 1) {
    const engines = index % 2 === 0 ? ["rust", "feff"] : ["feff", "rust"];
    for (const engine of engines) {
      console.log(`${definition.id}: ${engine} sample ${index + 1}/${args.iterations}`);
      measured[engine].push(await benchmarkRun(definition, engine, `sample-${index + 1}`));
    }
  }

  const rustOutput = measured.rust.at(-1).outputPath;
  const feffOutput = measured.feff.at(-1).outputPath;
  const rustRows = parseNumericTable(await readFile(rustOutput, "utf8"));
  const feffRows = parseNumericTable(await readFile(feffOutput, "utf8"));
  const comparison = compareRows(definition, feffRows, rustRows);

  cases.push({
    ...definition,
    comparison,
    benchmark: {
      rust: summarizeRuns(measured.rust),
      feff: summarizeRuns(measured.feff),
      speedup: median(measured.feff.map((run) => run.wallSeconds))
        / median(measured.rust.map((run) => run.wallSeconds)),
      warmupsPerEngine: 1,
      measuredIterations: args.iterations,
      threadPolicy: "Both engines forced to one thread",
    },
    files: {
      feff: path.relative(root, feffOutput),
      rust: path.relative(root, rustOutput),
    },
  });
}

const report = {
  generatedAt: new Date().toISOString(),
  provenance,
  method: {
    releaseBuild: true,
    timing: "Wall clock measured around each complete process; one discarded warm-up per engine.",
    ordering: "Measured Rust and FEFF runs alternate order to reduce thermal and ordering bias.",
    isolation: "Every run receives a fresh output directory.",
    parity:
      "Direct numeric comparison of fresh Rust and FEFF outputs. Relative L2 is computed per column; the registered spectrum tolerance is 5e-5 relative with 5e-8 absolute.",
    caveat:
      "FEFF uses the local sequential reference driver. Its historical build flags are not recorded in the fixture manifest, so timings describe these exact local binaries rather than every possible FEFF build.",
  },
  inputStage,
  cases,
};

await writeFile(path.join(outputRoot, "report.json"), `${JSON.stringify(report, null, 2)}\n`);
await writeFile(path.join(outputRoot, "index.html"), renderHtml(report));

console.log(`\nVisual report: ${path.join(outputRoot, "index.html")}`);
console.log(`Raw report:    ${path.join(outputRoot, "report.json")}`);

function parseArguments(values) {
  const parsed = {
    root: process.cwd(),
    output: "target/feff-comparison-report",
    iterations: 5,
    inputIterations: 5,
    rustBinary: "target/release/refeff",
    feffDriver: "feff10/bin/feff",
    xtaskBinary: "target/release/xtask",
  };
  for (let index = 0; index < values.length; index += 1) {
    const flag = values[index];
    const value = values[index + 1];
    if (flag === "--root") parsed.root = value;
    else if (flag === "--output") parsed.output = value;
    else if (flag === "--iterations") parsed.iterations = Number.parseInt(value, 10);
    else if (flag === "--rust-binary") parsed.rustBinary = value;
    else if (flag === "--feff-driver") parsed.feffDriver = value;
    else if (flag === "--xtask-binary") parsed.xtaskBinary = value;
    else if (flag === "--help") {
      console.log(`Usage: node scripts/feff-visual-report.mjs [options]

Options:
  --iterations N       Timed full-workflow samples per engine (default: 5)
  --output PATH        Report directory (default: target/feff-comparison-report)
  --rust-binary PATH   Rust release binary
  --feff-driver PATH   Sequential FEFF reference driver
  --xtask-binary PATH  Release xtask binary`);
      process.exit(0);
    } else {
      throw new Error(`unknown or incomplete argument: ${flag}`);
    }
    index += 1;
  }
  if (!Number.isInteger(parsed.iterations) || parsed.iterations < 1) {
    throw new Error("--iterations must be a positive integer");
  }
  return parsed;
}

function collectProvenance() {
  const rustCommit = commandText("git", ["rev-parse", "HEAD"], root);
  const feffCommit = commandText("git", ["rev-parse", "HEAD"], path.join(root, "feff10"));
  const rustVersion = commandText(rustBinary, ["--version"], root);
  const feffVersion = "FEFF 10.0.0 sequential module driver";
  return {
    rustCommit,
    feffCommit,
    rustVersion,
    feffVersion,
    rustCompiler: commandText("rustc", ["--version"], root),
    localFortranCompiler: commandText("gfortran", ["--version"], root).split("\n")[0],
    platform: `${os.type()} ${os.release()} ${os.arch()}`,
    cpu: os.cpus()[0]?.model ?? "unknown",
    logicalCores: os.cpus().length,
    memoryGiB: os.totalmem() / 1024 ** 3,
  };
}

function runInputStageBenchmark() {
  const result = spawnSync(
    xtaskBinary,
    ["bench-e2e", "--iterations", String(args.inputIterations), "--reference"],
    {
      cwd: root,
      encoding: "utf8",
      env: { ...process.env, REFEFF_THREADS: "1" },
      maxBuffer: 64 * 1024 * 1024,
    },
  );
  if (result.status !== 0) {
    throw new Error(`input-stage benchmark failed:\n${result.stdout}\n${result.stderr}`);
  }
  const rust = parseInputBenchmarkLine(result.stdout, "rust rdinp");
  const feff = parseInputBenchmarkLine(result.stdout, "feff10 rdinp");
  return {
    rust,
    feff,
    speedup: feff.averageSeconds / rust.averageSeconds,
    raw: result.stdout.trim(),
  };
}

function parseInputBenchmarkLine(output, prefix) {
  const line = output
    .split(/\r?\n/)
    .find((candidate) => candidate.startsWith(`${prefix}:`));
  if (!line) throw new Error(`missing ${prefix} benchmark summary`);
  const number = (label) => {
    const match = line.match(new RegExp(`${label}=([0-9.]+)`));
    if (!match) throw new Error(`missing ${label} in ${line}`);
    return Number(match[1]);
  };
  return {
    inputs: number("inputs"),
    iterations: number("iterations"),
    runs: number("runs"),
    successful: number("ok"),
    failed: number("failed"),
    totalSeconds: number("time"),
    averageSeconds: number("avg/run"),
  };
}

async function benchmarkRun(definition, engine, label) {
  const safeCase = definition.id.replaceAll("/", "-").toLowerCase();
  const runDirectory = path.join(runRoot, safeCase, engine, label);
  await mkdir(runDirectory, { recursive: true });

  let command;
  let commandArgs;
  let cwd;
  if (engine === "rust") {
    command = rustBinary;
    commandArgs = [
      "--threads",
      "1",
      "--quiet",
      "run",
      "-i",
      path.join(root, "reference-work", "golden", definition.id, "feff.inp"),
      "-o",
      runDirectory,
    ];
    cwd = root;
  } else {
    const sourceDirectory = path.join(root, "feff10", "examples", definition.id);
    await cp(sourceDirectory, runDirectory, { recursive: true, force: true });
    command = feffDriver;
    commandArgs = [];
    cwd = runDirectory;
  }

  const started = process.hrtime.bigint();
  const result = spawnSync("/usr/bin/time", ["-lp", command, ...commandArgs], {
    cwd,
    encoding: "utf8",
    env: { ...process.env, REFEFF_THREADS: "1" },
    maxBuffer: 64 * 1024 * 1024,
  });
  const wallSeconds = Number(process.hrtime.bigint() - started) / 1e9;
  await writeFile(path.join(runDirectory, "benchmark.stdout.log"), result.stdout ?? "");
  await writeFile(path.join(runDirectory, "benchmark.stderr.log"), result.stderr ?? "");

  const outputPath = path.join(runDirectory, definition.output);
  if (result.status !== 0) {
    throw new Error(
      `${definition.id} ${engine} ${label} failed with status ${result.status}; see ${runDirectory}`,
    );
  }
  await readFile(outputPath);

  return {
    label,
    wallSeconds,
    userSeconds: parseTimeMetric(result.stderr, "user"),
    systemSeconds: parseTimeMetric(result.stderr, "sys"),
    maximumResidentBytes: parseIntegerMetric(result.stderr, "maximum resident set size"),
    outputPath,
  };
}

function parseTimeMetric(text, name) {
  const match = text.match(new RegExp(`(?:^|\\n)(?:\\s*)([0-9.]+)\\s+${name}(?:\\s|$)`));
  return match ? Number(match[1]) : null;
}

function parseIntegerMetric(text, name) {
  const match = text.match(new RegExp(`(?:^|\\n)\\s*([0-9]+)\\s+${name}`));
  return match ? Number(match[1]) : null;
}

function parseNumericTable(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"))
    .map((line) =>
      line
        .split(/\s+/)
        .map((token) => Number(token.replace(/[dD]/g, "E"))),
    )
    .filter((row) => row.length > 0 && row.every(Number.isFinite));
}

function compareRows(definition, feffRows, rustRows) {
  if (feffRows.length !== rustRows.length) {
    throw new Error(
      `${definition.id} row count differs: FEFF ${feffRows.length}, Rust ${rustRows.length}`,
    );
  }
  const columnCount = definition.columns.length;
  const columns = [];
  for (let column = 0; column < columnCount; column += 1) {
    const feff = feffRows.map((row) => row[column]);
    const rust = rustRows.map((row) => row[column]);
    if (feff.some((value) => value === undefined) || rust.some((value) => value === undefined)) {
      throw new Error(`${definition.id} has a short numeric row in column ${column}`);
    }
    const differences = rust.map((value, index) => value - feff[index]);
    const diffSquared = sum(differences.map((value) => value * value));
    const feffSquared = sum(feff.map((value) => value * value));
    const rustSquared = sum(rust.map((value) => value * value));
    const scale = Math.max(Math.sqrt(feffSquared), Math.sqrt(rustSquared));
    const relativeL2 = scale > 0 ? Math.sqrt(diffSquared) / scale : 0;
    const absoluteL2 = Math.sqrt(diffSquared);
    const tolerance = Math.max(
      SPECTRUM_ABSOLUTE_TOLERANCE * Math.sqrt(feff.length),
      SPECTRUM_RELATIVE_TOLERANCE * scale,
    );
    columns.push({
      name: definition.columns[column],
      relativeL2,
      absoluteL2,
      maxAbsolute: Math.max(...differences.map(Math.abs)),
      rms: Math.sqrt(diffSquared / differences.length),
      passed: absoluteL2 <= tolerance,
    });
  }
  const x = {
    feff: feffRows.map((row) => row[definition.xColumn]),
    rust: rustRows.map((row) => row[definition.xColumn]),
  };
  const series = definition.plots.map((plot) => ({
    ...plot,
    feff: feffRows.map((row) => row[plot.column]),
    rust: rustRows.map((row) => row[plot.column]),
    residual: rustRows.map((row, index) => row[plot.column] - feffRows[index][plot.column]),
  }));
  return {
    rows: feffRows.length,
    passed: columns.every((column) => column.passed),
    maxRelativeL2: Math.max(...columns.map((column) => column.relativeL2)),
    maxAbsolute: Math.max(...columns.map((column) => column.maxAbsolute)),
    columns,
    x,
    series,
  };
}

function summarizeRuns(runs) {
  const wall = runs.map((run) => run.wallSeconds);
  const rss = runs.map((run) => run.maximumResidentBytes).filter(Number.isFinite);
  return {
    samples: wall,
    medianSeconds: median(wall),
    meanSeconds: mean(wall),
    minimumSeconds: Math.min(...wall),
    maximumSeconds: Math.max(...wall),
    standardDeviationSeconds: standardDeviation(wall),
    p95Seconds: percentile(wall, 0.95),
    medianMaximumResidentMiB: rss.length ? median(rss) / 1024 ** 2 : null,
  };
}

function commandText(command, commandArgs, cwd) {
  const result = spawnSync(command, commandArgs, { cwd, encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error(`${command} ${commandArgs.join(" ")} failed: ${result.stderr}`);
  }
  return result.stdout.trim();
}

function sum(values) {
  return values.reduce((total, value) => total + value, 0);
}

function mean(values) {
  return sum(values) / values.length;
}

function median(values) {
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2
    ? ordered[middle]
    : (ordered[middle - 1] + ordered[middle]) / 2;
}

function percentile(values, ratio) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.min(ordered.length - 1, Math.ceil(ratio * ordered.length) - 1)];
}

function standardDeviation(values) {
  const average = mean(values);
  return Math.sqrt(mean(values.map((value) => (value - average) ** 2)));
}

function renderHtml(report) {
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
    .performance { display: grid; grid-template-columns: 1.15fr .85fr; gap: 18px; }
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
      .section-head { align-items: start; flex-direction: column; }
      .section-head p { text-align: left; }
    }
    @media (max-width: 560px) {
      main { width: min(100% - 22px, 1440px); padding-top: 28px; }
      .kpis { grid-template-columns: 1fr; }
      .bar-row { grid-template-columns: 95px 1fr 64px; }
      .sample-dots { margin-left: 107px; }
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
        <p>Solid lines are FEFF; dashed lines are Rust. The lower panels show Rust − FEFF residuals. Metrics use each registered FEFF-format column, not header text.</p>
      </div>
      <div class="case-grid" id="cases"></div>
    </section>

    <section>
      <div class="section-head">
        <div><div class="eyebrow">02 · performance</div><h2>Release build vs sequential FEFF</h2></div>
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

    <section>
      <div class="panel">
        <div class="section-head">
          <div><div class="eyebrow">03 · methodology</div><h2>How to read this report</h2></div>
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
    const formatSpeed = value => value >= 1 ? value.toFixed(2) + "× faster" : (1 / value).toFixed(2) + "× slower";
    const allPassed = report.cases.every(item => item.comparison.passed);
    document.getElementById("overall-status").textContent = allPassed ? "Primary spectra pass" : "Review differences";
    document.getElementById("generated-at").textContent = new Date(report.generatedAt).toLocaleString();
    document.getElementById("commit-footer").textContent = "Rust " + report.provenance.rustCommit.slice(0, 10) + " · FEFF " + report.provenance.feffCommit.slice(0, 10);

    const xanes = report.cases.find(item => item.id === "XANES/BN");
    const exafs = report.cases.find(item => item.id === "EXAFS/Cu");
    const kpis = [
      ["Parity", report.cases.filter(item => item.comparison.passed).length + "/" + report.cases.length, "primary spectrum outputs"],
      ["Max relative L2", formatMetric(Math.max(...report.cases.map(item => item.comparison.maxRelativeL2))), "registered limit " + formatMetric(${SPECTRUM_RELATIVE_TOLERANCE})],
      ["RDINP speed", formatSpeed(report.inputStage.speedup), formatSeconds(report.inputStage.rust.averageSeconds) + " Rust / run"],
      ["BN XANES", formatSpeed(xanes.benchmark.speedup), "full pipeline median"],
    ];
    document.getElementById("kpis").innerHTML = kpis.map(([label, value, note]) =>
      '<div class="kpi"><div class="label">' + label + '</div><span class="value">' + value + '</span><div class="note">' + note + '</div></div>'
    ).join("");

    const caseRoot = document.getElementById("cases");
    for (const item of report.cases) {
      const panel = document.createElement("article");
      panel.className = "panel";
      const rows = item.comparison.columns.map(column =>
        '<tr><td>' + column.name + '</td><td class="number">' + formatMetric(column.relativeL2) +
        '</td><td class="number">' + formatMetric(column.maxAbsolute) +
        '</td><td class="' + (column.passed ? "pass" : "fail") + '">' + (column.passed ? "PASS" : "FAIL") + '</td></tr>'
      ).join("");
      panel.innerHTML =
        '<div class="panel-head"><div><h3>' + item.title + '</h3><div class="subtitle">' + item.subtitle + ' · ' + item.comparison.rows + ' rows</div></div>' +
        '<span class="status">' + (item.comparison.passed ? "Pass" : "Review") + '</span></div>' +
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

    const performanceRoot = document.getElementById("performance-bars");
    const maxRuntime = Math.max(...report.cases.flatMap(item => [item.benchmark.rust.medianSeconds, item.benchmark.feff.medianSeconds]));
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
      input.speedup.toFixed(2) + '×</strong><span style="color:var(--green);padding-bottom:7px">Rust faster</span></div>' +
      '<table><tbody><tr><td>Rust</td><td class="number">' + formatSeconds(input.rust.averageSeconds) +
      '</td></tr><tr><td>FEFF</td><td class="number">' + formatSeconds(input.feff.averageSeconds) +
      '</td></tr><tr><td>Successful runs</td><td class="number">' + input.rust.successful + ' / ' + input.rust.runs + '</td></tr></tbody></table>';

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
      const width = 720, height = residual ? 170 : 310;
      const padding = { left: 58, right: 20, top: 20, bottom: 38 };
      const xValues = item.comparison.x.feff;
      const values = residual
        ? item.comparison.series.flatMap(series => series.residual)
        : item.comparison.series.flatMap(series => [...series.feff, ...series.rust]);
      const xMin = Math.min(...xValues), xMax = Math.max(...xValues);
      let yMin = Math.min(...values), yMax = Math.max(...values);
      if (residual) {
        const extent = Math.max(Math.abs(yMin), Math.abs(yMax), 1e-16);
        yMin = -extent; yMax = extent;
      }
      const yPad = (yMax - yMin || 1) * .08;
      yMin -= yPad; yMax += yPad;
      const sx = value => padding.left + (value - xMin) / (xMax - xMin || 1) * (width - padding.left - padding.right);
      const sy = value => padding.top + (yMax - value) / (yMax - yMin || 1) * (height - padding.top - padding.bottom);
      const pathFor = (xs, ys) => ys.map((value, index) => (index ? "L" : "M") + sx(xs[index]).toFixed(2) + "," + sy(value).toFixed(2)).join(" ");
      const xTicks = Array.from({length: 6}, (_, index) => xMin + (xMax - xMin) * index / 5);
      const yTicks = Array.from({length: 5}, (_, index) => yMin + (yMax - yMin) * index / 4);
      const grid = [
        ...xTicks.map(value => '<line class="grid" x1="' + sx(value) + '" x2="' + sx(value) + '" y1="' + padding.top + '" y2="' + (height-padding.bottom) + '"/><text class="tick" x="' + sx(value) + '" y="' + (height-14) + '" text-anchor="middle">' + compact(value) + '</text>'),
        ...yTicks.map(value => '<line class="grid" x1="' + padding.left + '" x2="' + (width-padding.right) + '" y1="' + sy(value) + '" y2="' + sy(value) + '"/><text class="tick" x="' + (padding.left-8) + '" y="' + (sy(value)+3) + '" text-anchor="end">' + compact(value) + '</text>'),
      ].join("");
      const lines = item.comparison.series.map(series => residual
        ? '<path d="' + pathFor(xValues, series.residual) + '" fill="none" stroke="' + series.color + '" stroke-width="1.5"/>'
        : '<path d="' + pathFor(xValues, series.feff) + '" fill="none" stroke="' + series.color + '" stroke-width="2"/><path d="' +
          pathFor(item.comparison.x.rust, series.rust) + '" fill="none" stroke="' + series.color + '" stroke-width="1.5" stroke-dasharray="7 5" opacity=".9"/>'
      ).join("");
      container.innerHTML = '<svg viewBox="0 0 ' + width + ' ' + height + '" role="img" aria-label="' + item.title + (residual ? " residual" : " overlay") + '">' +
        grid + '<line class="axis" x1="' + padding.left + '" x2="' + (width-padding.right) + '" y1="' + (height-padding.bottom) + '" y2="' + (height-padding.bottom) + '"/>' +
        (residual && yMin <= 0 && yMax >= 0 ? '<line x1="' + padding.left + '" x2="' + (width-padding.right) + '" y1="' + sy(0) + '" y2="' + sy(0) + '" stroke="#7e91aa" stroke-width="1"/>' : "") +
        lines + '<text class="tick" x="' + (width/2) + '" y="' + (height-2) + '" text-anchor="middle">' + item.columns[item.xColumn] + '</text></svg>';
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
