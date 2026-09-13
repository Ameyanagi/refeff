# WebAssembly

ReFEFF builds for `wasm32-wasip1`. The CLI, including the full scheduler, and
the Rust `refeff::Runner` API run serially on this target. The browser adapter
runs the same WASI binary in a dedicated Web Worker with a virtual filesystem
provided by [browser_wasi_shim](https://github.com/bjorn3/browser_wasi_shim).
All input and generated files stay in the browser's memory.

## Build and run with WASI

From the repository root, using the pinned Rust toolchain and Node.js 24:

```sh
rustup target add wasm32-wasip1
cargo build --release -p refeff-cli --bin refeff --bin feff --target wasm32-wasip1 --locked
node wasm/run-wasi.mjs target/wasm32-wasip1/release/refeff.wasm \
  --json run --input crates/refeff/tests/data/znse.inp --output target/wasm-znse
```

`run-wasi.mjs` uses [Node's WASI implementation](https://nodejs.org/api/wasi.html),
exposes the working directory as `.`, and maps a private scratch directory to
`/tmp`. Pass paths relative to the working directory. The runner propagates
the program's exit code and removes its scratch directory after execution.
The equivalent FEFF-compatible `feff.wasm` reads `feff.inp` in that directory.

Other WASI Preview 1 hosts can use these binaries directly. They must supply
filesystem, clock, and randomness imports, expose the input/output directory,
and set `TMPDIR` to a writable guest directory. For example, with
[Wasmtime](https://docs.wasmtime.dev/cli-options.html):

```sh
mkdir -p target/wasm-tmp
wasmtime run --dir . --env TMPDIR=target/wasm-tmp \
  target/wasm32-wasip1/release/refeff.wasm \
  run --input crates/refeff/tests/data/znse.inp --output target/wasm-znse
```

Rust applications can cross-compile against `refeff` using the same target.
`Runner::run_files` and `Runner::run_in_memory` retain their existing APIs;
the latter still needs the WASI scratch filesystem internally. The `exafs`,
`sfconv`, `full`, and `serde` Cargo features remain available.

## Browser integration

For a complete local example with an input editor, spectrum plot, cancellation,
and downloads, run these commands from the repository root:

```sh
npm ci --prefix wasm
npm run build --prefix wasm
npm start --prefix wasm
```

Open <http://127.0.0.1:4173> and click **Run calculation**. The page starts
with the bundled ZnSe input; you can edit it or open another self-contained
`feff.inp`. The server only serves the example and local WASM assets.
Calculations run in the browser. Stop the server with Ctrl+C, or set `PORT`
to use a different port. The example source is in [`examples/`](examples/).

To integrate the adapter into your own application:

```sh
npm ci --prefix wasm
npm run build --prefix wasm
```

The build compiles the Rust CLI and writes `wasm/dist/index.mjs`,
`worker.mjs`, `refeff.wasm`, and third-party license files. Copy the entire
`dist` directory to your application's static assets, for example
`public/refeff/`, and serve it over HTTP(S). This is a local integration
package; it has not been published to npm.

```js
import { runFeff } from '/refeff/index.mjs';

const controller = new AbortController();
const result = await runFeff({
  input: await inputFile.text(), // string or Uint8Array containing feff.inp
  files: {},                    // optional relative path -> string/Uint8Array
  signal: controller.signal,
  onLog: ({ stream, text }) => console.log(stream, text),
});

if (result.exitCode !== 0) throw new Error(result.report.error.message);
const chiText = new TextDecoder().decode(result.files['chi.dat']);
const xmuBytes = result.files['xmu.dat'];
console.log(result.report.data.stages, chiText);
// Use xmuBytes with Blob/download APIs or your application's spectrum parser.
```

Provide auxiliary files such as included card files, CIFs, or dynamical
matrices through `files`, preserving relative names used in the input:

```js
const result = await runFeff({
  input: 'INCLUDE inputs/model.inp\n',
  files: { 'inputs/model.inp': modelText },
});
```

The `input` property always supplies the root `feff.inp`, overriding a file
of that name in `files`. Absolute paths, empty components, `.`/`..`, NULs,
and backslashes in supplied filenames are rejected. Each call has a fresh
filesystem; files are not shared between calculations.

The promise returns `{ exitCode, report, files, stdout, stderr }`. `report`
is the CLI's versioned JSON response; `files` maps output-relative names to
`Uint8Array` payloads, including binary FEFF handoffs. Calculation errors
return a nonzero `exitCode` with a structured error report and any partial
outputs. Loading failures, invalid API inputs, and WebAssembly traps reject
the promise. An optional `wasmUrl` overrides the binary's location.

Call `controller.abort()` from your UI while the promise is pending to stop
the worker immediately and discard its outputs. This rejects with an
`AbortError` (or the custom abort reason). `onLog` receives stdout/stderr
chunks, including stage progress on stderr. Each calculation runs in its own
worker, so the page stays responsive.

## Scope and limits

- WebAssembly uses one calculation thread. `--threads`, `REFEFF_THREADS`,
  and `Runner::with_threads` are clamped to one; CLI reports reflect that.
  Native builds retain their existing Rayon/faer parallelism.
- Browser support uses WASI Preview 1 imports and the bundled shim. The
  file-backed facade does not run directly on `wasm32-unknown-unknown` or
  through `wasm-bindgen`. No SharedArrayBuffer or cross-origin isolation
  headers are required by this adapter.
- Browser inputs, intermediate artifacts, and returned output all consume
  memory. Large clusters may exceed browser or 32-bit WebAssembly memory
  limits. The adapter returns every output artifact.
- Numerical validation covers the complete bundled ZnSe EXAFS workflow,
  including FMS, against the serial native build. The complete CLI compiles,
  but this is not a new parity certification for every FEFF workflow.

## Verification

```sh
cargo build --release -p refeff-cli --bin refeff --locked
npm test --prefix wasm
cd wasm
npx playwright install chromium
npm run test:browser
```

The runtime tests execute the same ZnSe input natively, under Node WASI, and
with the browser filesystem. Every numeric column of `chi.dat` and
`xmu.dat` must agree within `5e-8 + 5e-5 * abs(reference)`. Chromium tests
execute the built worker, verify output and page responsiveness, and exercise
cancellation, input errors, and failed binary loading.

To run the Rust facade contracts on WASI, from the repository root:

```sh
export CARGO_TARGET_WASM32_WASIP1_RUNNER="node $PWD/wasm/run-wasi.mjs"
cargo test --release -p refeff --target wasm32-wasip1 \
  --lib --test execution --locked -- --test-threads 1
```

CI runs these checks and cross-checks the reduced Cargo feature combinations.
