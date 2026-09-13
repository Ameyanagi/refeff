import { WASI, File, Directory, OpenFile, ConsoleStdout, PreopenDirectory } from '@bjorn3/browser_wasi_shim';

const encoder = new TextEncoder();

function insertFile(directory, path, bytes) {
  if (typeof path !== 'string' || /[\\\0]/.test(path)) {
    throw new TypeError(`Invalid workspace path: ${path}`);
  }
  const parts = path.split('/');
  if (parts.some(part => !part || part === '.' || part === '..')) {
    throw new TypeError(`Invalid workspace path: ${path}`);
  }
  let parent = directory;
  for (const part of parts.slice(0, -1)) {
    if (!parent.contents.has(part)) parent.contents.set(part, new Directory([]));
    parent = parent.contents.get(part);
    if (!(parent instanceof Directory)) throw new TypeError(`File/directory conflict: ${path}`);
  }
  const name = parts.at(-1);
  if (parent.contents.get(name) instanceof Directory) {
    throw new TypeError(`File/directory conflict: ${path}`);
  }
  if (typeof bytes === 'string') bytes = encoder.encode(bytes);
  if (!(bytes instanceof Uint8Array)) throw new TypeError(`Expected text or Uint8Array for ${path}`);
  parent.contents.set(name, new File(bytes.slice()));
}

function collectFiles(directory, prefix = '', files = Object.create(null)) {
  for (const [name, entry] of directory.contents) {
    const path = prefix + name;
    if (entry instanceof Directory) collectFiles(entry, `${path}/`, files);
    else if (entry instanceof File) files[path] = entry.data.slice();
  }
  return files;
}

// Shared by the browser worker and runtime contract tests. Each invocation owns
// its WASI instance and filesystem, including the temporary scratch directory.
export async function execute(module, { input, files = {} }, onLog = () => {}) {
  const workspace = new Directory([]);
  for (const [path, bytes] of Object.entries(files)) insertFile(workspace, path, bytes);
  insertFile(workspace, 'feff.inp', input);
  const output = new Directory([]);
  const root = new Map([
    ['work', workspace], ['output', output], ['tmp', new Directory([])],
  ]);
  let stdout = '';
  let stderr = '';
  const outDecoder = new TextDecoder();
  const errDecoder = new TextDecoder();
  const wasi = new WASI([
    'refeff', '--json', 'run', '--input', '/work/feff.inp', '--output', '/output',
  ], ['TMPDIR=/tmp'], [
    new OpenFile(new File([])),
    new ConsoleStdout(bytes => {
      const text = outDecoder.decode(bytes, { stream: true });
      stdout += text;
      onLog({ stream: 'stdout', text });
    }),
    new ConsoleStdout(bytes => {
      const text = errDecoder.decode(bytes, { stream: true });
      stderr += text;
      onLog({ stream: 'stderr', text });
    }),
    new PreopenDirectory('/', root),
  ], { debug: false });
  const instance = await WebAssembly.instantiate(module, {
    wasi_snapshot_preview1: wasi.wasiImport,
  });
  const exitCode = wasi.start(instance);
  stdout += outDecoder.decode();
  stderr += errDecoder.decode();
  return { exitCode, report: JSON.parse(stdout), files: collectFiles(output), stdout, stderr };
}
