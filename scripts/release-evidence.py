#!/usr/bin/env python3
"""Verify locally produced parity evidence for the exact revision being published.

Usage: release-evidence.py evidence.json
The JSON must contain schema_version=2, commit, toolchain, input_manifest_sha256,
workflows [{id,status}], covering compatibility/feff10.json stock_workflows,
and the complete strict release-readiness report recorded on local hardware.
Evidence is supplied from a reviewed CI artifact or a committed release record.
"""
import hashlib, json, pathlib, runpy, subprocess, sys
root = pathlib.Path(__file__).resolve().parents[1]
evidence_path = pathlib.Path(sys.argv[1]).resolve()
data = json.loads(evidence_path.read_text())
manifest = root / 'compatibility/feff10.json'
expected = json.loads(manifest.read_text())['stock_workflows']
commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
toolchain = subprocess.check_output(['rustc','--version'],cwd=root,text=True).strip()
input_hashes = data.get('input_sha256', {})
provenance = data.get('provenance', {})
readiness = data.get('release_readiness', {})
audit = readiness.get('report', {})
def strict_readiness_passes(report):
 return (
  report.get('passed') is True
  and report.get('filters') == {'modules': [], 'rows': [], 'open_only': False, 'detail': False}
  and report.get('open_compatibility_items') == []
  and all(report.get(gate, {}).get('passed') is True and report[gate].get('error') is None
          for gate in ['production_scope', 'port_status', 'compatibility_matrix'])
 )
def input_matches(name, expected_hash):
 path = (evidence_path.parent / 'inputs' / name).resolve()
 try:
  path.relative_to((evidence_path.parent / 'inputs').resolve())
  return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
 except (ValueError, OSError): return False
def references_match():
 hashes = data.get('reference_manifest_sha256', {})
 if sorted(hashes) != sorted(expected): return False
 repairs = runpy.run_path(str(root / 'scripts/repair-native-references.py'))['REPAIRS']
 for workflow, checksum in hashes.items():
  directory = evidence_path.parent / 'references' / workflow
  try:
   path = directory / 'manifest.json'
   if hashlib.sha256(path.read_bytes()).hexdigest() != checksum: return False
   reference = json.loads(path.read_text())
   if reference.get('feff10_rev') != data.get('reference_commit'): return False
   if workflow in repairs:
    path = directory / '.native-reference-repair.json'
    files = {item['path']: item['sha256'] for item in reference['files']}
    if files.get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest(): return False
    repair = json.loads(path.read_text())
    if repair.get('native_commit') != reference['feff10_rev'] or repair.get('example') != workflow: return False
    if repair.get('generator_sha256') != hashlib.sha256((root/'scripts/repair-native-references.py').read_bytes()).hexdigest(): return False
    if repair.get('original_source_sha256') != repairs[workflow]['sha256']: return False
    outputs = repair.get('output_sha256', {})
    if not all(name in outputs for name in repairs[workflow]['outputs']): return False
    if any(files.get(name) != digest for name,digest in outputs.items()): return False
  except (OSError, ValueError, KeyError, TypeError): return False
 return True
checks = {
 'schema': data.get('schema_version') == 2,
 'exact revision': data.get('commit') == commit,
 'toolchain': data.get('toolchain') == toolchain,
 'input inventory': data.get('input_manifest_sha256') == hashlib.sha256(manifest.read_bytes()).hexdigest(),
 'complete workflows': sorted(row['id'] for row in data.get('workflows',[])) == sorted(expected),
 'completed run': data.get('complete') is True and data.get('binary_unchanged') is True,
 'all workflows pass': bool(data.get('workflows')) and all(row.get('status') == 'pass' for row in data['workflows']),
 'input payload hashes': bool(input_hashes) and all(input_matches(name, value) for name,value in input_hashes.items()),
 'workflow inputs': all(workflow+'/feff.inp' in input_hashes for workflow in expected),
 'native reference manifests and repairs': references_match(),
 'reference revision': data.get('reference_commit') == json.loads(manifest.read_text())['upstream']['revision'],
 'matching test provenance': provenance.get('rustCommit') == commit and provenance.get('dirty') is False and provenance.get('rustCompiler') == toolchain and provenance.get('feffCommit') == data.get('reference_commit'),
 'tested binary identity': all(isinstance(provenance.get(key),str) and len(provenance[key]) == 64 and all(c in '0123456789abcdef' for c in provenance[key]) for key in ['rustBinarySha256','feffDriverSha256']),
 'clean source': data.get('dirty') is False,
 'exact-revision readiness audit': readiness.get('commit') == commit and readiness.get('toolchain') == toolchain,
 'strict local readiness': strict_readiness_passes(audit),
}
failed = [name for name, passed in checks.items() if not passed]
if failed: raise SystemExit('Release evidence rejected: '+', '.join(failed))
print('Release evidence verified for '+commit)
