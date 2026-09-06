#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { renderHtml, extent } from "./report/render.mjs";

const SPECTRUM_RELATIVE_TOLERANCE = 5e-5;
const SPECTRUM_ABSOLUTE_TOLERANCE = 5e-8;

const CASES = [
  {
    id: "EXAFS/Cu",
    title: "Cu K-edge EXAFS",
    subtitle: "Path expansion and χ(k)",
    output: "chi.dat",
    columns: ["k", "chi (signed fine structure)", "magnitude", "phase"],
    xColumn: 0,
    plots: [{ column: 1, label: "χ(k) · signed fine structure", color: "#4d7cff" }],
  },
  {
    id: "XANES/BN",
    title: "BN B K-edge XANES",
    subtitle: "SCF + 87-atom FMS spectrum",
    output: "xmu.dat",
    columns: [
      "photon energy",
      "relative energy",
      "wave number",
      "mu (absorption)",
      "mu0 (atomic background)",
      "chi (signed fine structure)",
    ],
    xColumn: 0,
    plots: [
      {
        column: 3,
        label: "μ(E) · absorption",
        color: "#4d7cff",
        nonNegative: true,
      },
    ],
  },
];

const args = parseArguments(process.argv.slice(2));
const root = path.resolve(args.root);
const outputRoot = path.resolve(root, args.output);
const expectedWorkflows = JSON.parse(await readFile(path.join(root, "compatibility/feff10.json"), "utf8")).stock_workflows;
let workflowProvenance = null;

if (args.renderExisting) {
  const reportPath = path.join(outputRoot, "report.json");
  const report = applyVisualizationCorrections(JSON.parse(await readFile(reportPath, "utf8")));
  const workflows = await loadWorkflowSummary();
  if (workflows.length) {
    report.workflows = workflows;
    report.galleryCases = await loadArtifactGallery(workflows);
  }
  report.expectedWorkflows = expectedWorkflows;
  report.workflowProvenance = workflowProvenance;
  report.visualizationUpdatedAt = new Date().toISOString();
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  await writeFile(path.join(outputRoot, "index.html"), renderHtml(report));
  console.log(`Visual report: ${path.join(outputRoot, "index.html")}`);
  console.log(`Raw report:    ${reportPath}`);
  process.exit(0);
}

const sessionId = new Date().toISOString().replaceAll(/[:.]/g, "-");
const runRoot = path.join(outputRoot, "runs", sessionId);
const rustBinary = path.resolve(root, args.rustBinary);
const feffDriver = path.resolve(root, args.feffDriver);
const xtaskBinary = path.resolve(root, args.xtaskBinary);

await mkdir(runRoot, { recursive: true });

const provenance = collectProvenance();
const inputStage = runInputStageBenchmark();
const workflows = await loadWorkflowSummary();
const galleryCases = await loadArtifactGallery(workflows);
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
  expectedWorkflows,
  workflowProvenance,
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
  workflows,
  galleryCases,
  cases,
};

applyVisualizationCorrections(report);
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
    workflowSummary: "target/final-parity/workflow-summary.json",
    renderExisting: false,
  };
  for (let index = 0; index < values.length; index += 1) {
    const flag = values[index];
    if (flag === "--render-existing") {
      parsed.renderExisting = true;
      continue;
    }
    const value = values[index + 1];
    if (flag === "--root") parsed.root = value;
    else if (flag === "--output") parsed.output = value;
    else if (flag === "--iterations") parsed.iterations = Number.parseInt(value, 10);
    else if (flag === "--rust-binary") parsed.rustBinary = value;
    else if (flag === "--feff-driver") parsed.feffDriver = value;
    else if (flag === "--xtask-binary") parsed.xtaskBinary = value;
    else if (flag === "--workflow-summary") parsed.workflowSummary = value;
    else if (flag === "--help") {
      console.log(`Usage: node scripts/feff-visual-report.mjs [options]

Options:
  --render-existing    Re-render an existing report without rerunning benchmarks
  --iterations N       Timed full-workflow samples per engine (default: 5)
  --output PATH        Report directory (default: target/feff-comparison-report)
  --rust-binary PATH   Rust release binary
  --feff-driver PATH   Sequential FEFF reference driver
  --xtask-binary PATH  Release xtask binary
  --workflow-summary PATH
                       Frozen all-workflow summary JSON`);
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

async function loadWorkflowSummary() {
  const summaryPath = path.resolve(root, args.workflowSummary);
  try {
    const summary = JSON.parse(await readFile(summaryPath, "utf8"));
    if (!Array.isArray(summary.workflows)) {
      throw new Error(`${summaryPath} must contain a workflows array`);
    }
    workflowProvenance = summary.provenance ?? null;
    return summary.workflows;
  } catch (error) {
    if (error?.code === "ENOENT") return [];
    throw error;
  }
}

async function loadArtifactGallery(workflows) {
  const gallery = [];
  for (const workflow of workflows) {
    for (const output of requiredArtifactTargets(workflow.id)) {
      if (
        (workflow.id === "EXAFS/Cu" && output === "chi.dat")
        || (workflow.id === "XANES/BN" && output === "xmu.dat")
      ) {
        continue;
      }
      const definition = artifactPlotDefinition(workflow.id, output);
      if (!definition) continue;
      const rustPath = path.join(root, "target", "xtask-parity", workflow.id, output);
      const goldenCandidates = workflow.id === "DANES/GeCl_4"
        ? [
            path.join(root, "target", "xtask-parity-reference", workflow.id, output),
            path.join(root, "reference-work", "golden", workflow.id, output),
          ]
        : [path.join(root, "reference-work", "golden", workflow.id, output)];
      let rustRows;
      try {
        rustRows = parseStrictArtifactTable(
          await readFile(rustPath, "utf8"),
          definition.columns.length,
          `${workflow.id}/${output} Rust`,
        );
      } catch (error) {
        if (error?.code === "ENOENT") continue;
        throw error;
      }
      let goldenPath;
      let feffRows;
      for (const candidate of goldenCandidates) {
        try {
          feffRows = parseStrictArtifactTable(
            await readFile(candidate, "utf8"),
            definition.columns.length,
            `${workflow.id}/${output} FEFF`,
          );
          goldenPath = candidate;
          break;
        } catch (error) {
          if (error?.code !== "ENOENT") throw error;
        }
      }
      if (!goldenPath || !feffRows) continue;
      const comparison = compareRows(definition, feffRows, rustRows);
      const plottedColumns = definition.plots.map((plot) => comparison.columns[plot.column]);
      gallery.push({
        ...definition,
        workflow: workflow.id,
        family: workflow.id.split("/")[0],
        output,
        status: workflow.status ?? (workflow.passed ? "pass" : "fail"),
        evidence: "Frozen/latest-built artifact pair; not proof from the exact modified checkout.",
        comparison,
        plottedMaxRelativeL2: extent(plottedColumns.map((column) => column.relativeL2))[1],
        plottedMaxAbsolute: extent(plottedColumns.map((column) => column.maxAbsolute))[1],
        files: {
          feff: path.relative(root, goldenPath),
          rust: path.relative(root, rustPath),
        },
      });
    }
  }
  return gallery;
}

function requiredArtifactTargets(example) {
  const segments = example.toUpperCase().split("/");
  const workflow = segments[0];
  if (workflow === "EXAFS") return ["chi.dat"];
  if (["EELS", "ELNES", "EXELFS"].includes(workflow)) return ["eels.dat"];
  if (workflow === "COMPTON") return ["compton.dat"];
  if (workflow === "CRPA") return ["crpa.dat"];
  if (["DANES", "FPRIME"].includes(workflow)) return ["danes.dat", "xmu.dat"];
  if (workflow === "DEBYE" && segments.includes("DM") && segments.includes("EXAFS")) {
    return ["dmdw.out", "chi.dat"];
  }
  if (workflow === "DEBYE" && segments.includes("DM") && segments.includes("XANES")) {
    return ["dmdw.out", "xmu.dat"];
  }
  if (workflow === "DEBYE") return ["xmu.dat"];
  if (workflow === "KSPACE" && segments[1] === "GRAPHITE") return ["eels.dat"];
  if (workflow === "RIXS") return ["rixsET.dat"];
  return ["xmu.dat"];
}

function artifactPlotDefinition(workflowId, output) {
  const family = workflowId.split("/")[0];
  const base = {
    id: `${workflowId} · ${output}`,
    title: workflowId,
    output,
    xColumn: 0,
  };
  if (output === "chi.dat") {
    return {
      ...base,
      subtitle: "signed EXAFS fine structure",
      columns: ["k", "chi", "magnitude", "phase"],
      plots: [{ column: 1, label: "χ(k) · signed", color: "#4d7cff" }],
    };
  }
  if (output === "eels.dat") {
    return {
      ...base,
      subtitle: "EELS total, background, and fine structure",
      columns: [
        "energy loss",
        "total",
        "atomic background",
        "fine structure",
        "xx",
        "xy",
        "xz",
        "yx",
        "yy",
        "yz",
        "zx",
        "zy",
        "zz",
      ],
      plots: [
        { column: 1, label: "total", color: "#4d7cff", nonNegative: true },
        {
          column: 2,
          label: "atomic background",
          color: "#ec6a5c",
          nonNegative: true,
        },
        { column: 3, label: "fine structure · signed", color: "#51c39a" },
      ],
    };
  }
  if (output === "danes.dat") {
    return {
      ...base,
      subtitle: "signed anomalous scattering components",
      columns: [
        "relative energy",
        "Matsubara",
        "Sommerfeld",
        "anomalous",
        "tail",
        "total",
        "difference",
      ],
      plots: [
        { column: 5, label: "total · signed", color: "#4d7cff" },
        { column: 3, label: "anomalous · signed", color: "#ec6a5c" },
      ],
    };
  }
  if (output !== "xmu.dat") return null;
  if (family === "FPRIME") {
    return {
      ...base,
      subtitle: "signed anomalous scattering factors",
      columns: ["photon energy", "relative energy", "f′", "f′0", "f″", "f″0"],
      plots: [
        { column: 2, label: "f′ · signed", color: "#4d7cff" },
        { column: 4, label: "f″ · signed", color: "#ec6a5c" },
      ],
    };
  }
  if (family === "DANES") {
    return {
      ...base,
      subtitle: "signed DANES spectrum",
      columns: [
        "photon energy",
        "relative energy",
        "wave number",
        "signed spectrum",
        "atomic term",
        "fine structure",
      ],
      plots: [{ column: 3, label: "signed spectrum", color: "#4d7cff" }],
    };
  }
  if (family === "NRIXS") {
    return {
      ...base,
      subtitle: "non-resonant inelastic scattering",
      columns: [
        "photon energy",
        "relative energy",
        "wave number",
        "S(q,ω)",
        "S0(q,ω)",
        "chiq × S0",
      ],
      plots: [
        { column: 3, label: "S(q,ω)", color: "#4d7cff", nonNegative: true },
        { column: 4, label: "S0(q,ω)", color: "#ec6a5c", nonNegative: true },
      ],
    };
  }
  if (family === "XMCD" || family === "XNCD") {
    return {
      ...base,
      subtitle: "signed dichroic spectrum",
      columns: [
        "photon energy",
        "relative energy",
        "wave number",
        "dichroic signal",
        "atomic term",
        "fine structure",
      ],
      plots: [{ column: 3, label: "dichroic signal · signed", color: "#4d7cff" }],
    };
  }
  const emission = family === "XES";
  return {
    ...base,
    subtitle: emission ? "emission intensity" : "absorption spectrum",
    columns: [
      "photon energy",
      "relative energy",
      "wave number",
      emission ? "emission intensity" : "mu",
      "atomic background",
      "fine structure",
    ],
    plots: [
      {
        column: 3,
        label: emission ? "emission intensity" : "μ(E) · absorption",
        color: "#4d7cff",
        nonNegative: true,
      },
    ],
  };
}

function applyVisualizationCorrections(report) {
  const primaryColumns = new Map([
    ["EXAFS/Cu", { column: 1, label: "χ(k) · signed fine structure", color: "#4d7cff" }],
    [
      "XANES/BN",
      {
        column: 3,
        label: "μ(E) · absorption",
        color: "#4d7cff",
        nonNegative: true,
      },
    ],
  ]);
  const columnLabels = new Map([
    [
      "EXAFS/Cu",
      ["k", "chi (signed fine structure)", "magnitude", "phase"],
    ],
    [
      "XANES/BN",
      [
        "photon energy",
        "relative energy",
        "wave number",
        "mu (absorption)",
        "mu0 (atomic background)",
        "chi (signed fine structure)",
      ],
    ],
  ]);

  for (const item of report.cases) {
    const primary = primaryColumns.get(item.id);
    const labels = columnLabels.get(item.id);
    if (!primary || !labels) continue;
    item.columns = labels;
    item.plots = [primary];
    const primarySeries = item.comparison.series.find(
      (series) => series.column === primary.column,
    );
    if (primarySeries) {
      primarySeries.label = primary.label;
      primarySeries.color = primary.color;
      primarySeries.nonNegative = primary.nonNegative ?? false;
      item.comparison.series = [primarySeries];
    }
    for (const [index, column] of item.comparison.columns.entries()) {
      column.name = labels[index] ?? column.name;
    }
  }
  report.method.visualization =
    "The benchmark panels show signed chi(k) for EXAFS and non-negative mu(E) absorption for XANES. The frozen gallery selects observables by workflow semantics, including signed dichroism/scattering and non-negative physical spectra.";
  report.method.artifactFreshness =
    "Gallery pairs are frozen/latest-built evidence. The modified source checkout postdates them, so the gallery is not a substitute for the pending current-source release sweep.";
  return report;
}

function collectProvenance() {
  const rustCommit = commandText("git", ["rev-parse", "HEAD"], root);
  const feffCommit = commandText("git", ["rev-parse", "HEAD"], path.join(root, "feff10"));
  const rustVersion = commandText(rustBinary, ["--version"], root);
  const feffVersion = "FEFF 10.0.0 sequential module driver";
  return {
    rustCommit,
    feffCommit,
    dirty: commandText("git", ["status", "--porcelain", "--untracked-files=no"], root).length > 0,
    rustVersion,
    feffVersion,
    rustBinarySha256: createHash("sha256").update(readFileSync(rustBinary)).digest("hex"),
    feffDriverSha256: createHash("sha256").update(readFileSync(feffDriver)).digest("hex"),
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

function parseStrictArtifactTable(text, expectedColumns, label) {
  const rows = [];
  for (const [index, rawLine] of text.split(/\r?\n/).entries()) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const tokens = line.split(/\s+/);
    if (tokens.length !== expectedColumns) {
      throw new Error(
        `${label} line ${index + 1} has ${tokens.length} columns; expected ${expectedColumns}`,
      );
    }
    const row = tokens.map((token) => Number(token.replace(/[dD]/g, "E")));
    if (row.some((value) => !Number.isFinite(value))) {
      throw new Error(`${label} line ${index + 1} contains a non-finite numeric value`);
    }
    rows.push(row);
  }
  if (!rows.length) throw new Error(`${label} contains no numeric rows`);
  return rows;
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
      maxAbsolute: extent(differences.map(Math.abs))[1],
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
  const physicalChecks = series
    .filter((item) => item.nonNegative)
    .map((item) => {
      const feffMinimum = extent(item.feff)[0];
      const rustMinimum = extent(item.rust)[0];
      const minimum = Math.min(feffMinimum, rustMinimum);
      return {
        name: `${item.label} is non-negative`,
        feffMinimum,
        rustMinimum,
        maximumViolation: Math.max(0, -minimum),
        passed: minimum >= -SPECTRUM_ABSOLUTE_TOLERANCE,
      };
    });
  return {
    rows: feffRows.length,
    passed:
      columns.every((column) => column.passed)
      && physicalChecks.every((check) => check.passed),
    maxRelativeL2: extent(columns.map((column) => column.relativeL2))[1],
    maxAbsolute: extent(columns.map((column) => column.maxAbsolute))[1],
    columns,
    physicalChecks,
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
    minimumSeconds: extent(wall)[0],
    maximumSeconds: extent(wall)[1],
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
