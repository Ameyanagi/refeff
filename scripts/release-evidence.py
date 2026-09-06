#!/usr/bin/env python3
"""Verify locally produced parity evidence for the exact revision being published.

Usage: release-evidence.py evidence.json
The JSON must contain schema_version=1, commit, toolchain, input_manifest_sha256,
and workflows [{id,status}], covering compatibility/feff10.json stock_workflows.
Evidence is supplied from a reviewed CI artifact or a committed release record.
"""
import hashlib, json, pathlib, subprocess, sys
root = pathlib.Path(__file__).resolve().parents[1]
evidence_path = pathlib.Path(sys.argv[1]).resolve()
data = json.loads(evidence_path.read_text())
manifest = root / 'compatibility/feff10.json'
expected = json.loads(manifest.read_text())['stock_workflows']
commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
toolchain = subprocess.check_output(['rustc','--version'],cwd=root,text=True).strip()
input_hashes = data.get('input_sha256', {})
provenance = data.get('provenance', {})
def input_matches(name, expected_hash):
 path = (evidence_path.parent / 'inputs' / name).resolve()
 try:
  path.relative_to((evidence_path.parent / 'inputs').resolve())
  return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
 except (ValueError, OSError): return False
checks = {
 'schema': data.get('schema_version') == 1,
 'exact revision': data.get('commit') == commit,
 'toolchain': data.get('toolchain') == toolchain,
 'input inventory': data.get('input_manifest_sha256') == hashlib.sha256(manifest.read_bytes()).hexdigest(),
 'complete workflows': sorted(row['id'] for row in data.get('workflows',[])) == sorted(expected),
 'all workflows pass': bool(data.get('workflows')) and all(row.get('status') == 'pass' for row in data['workflows']),
 'input payload hashes': bool(input_hashes) and all(input_matches(name, value) for name,value in input_hashes.items()),
 'workflow inputs': all(workflow+'/feff.inp' in input_hashes for workflow in expected),
 'reference revision': data.get('reference_commit') == json.loads(manifest.read_text())['upstream']['revision'],
 'matching test provenance': provenance.get('rustCommit') == commit and provenance.get('dirty') is False and provenance.get('rustCompiler') == toolchain and provenance.get('feffCommit') == data.get('reference_commit'),
 'tested binary identity': all(isinstance(provenance.get(key),str) and len(provenance[key]) == 64 and all(c in '0123456789abcdef' for c in provenance[key]) for key in ['rustBinarySha256','feffDriverSha256']),
 'clean source': data.get('dirty') is False,
}
failed = [name for name, passed in checks.items() if not passed]
if failed: raise SystemExit('Release evidence rejected: '+', '.join(failed))
print('Release evidence verified for '+commit)
