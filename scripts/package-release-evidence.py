#!/usr/bin/env python3
"""Package a completed local workflow summary plus actual inputs for the release gate."""
import argparse, hashlib, json, pathlib, subprocess, zipfile
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--summary',type=pathlib.Path,required=True)
p.add_argument('--inputs',type=pathlib.Path,required=True,help='FEFF examples/fixture root')
p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args();root=pathlib.Path(__file__).resolve().parents[1]
summary=json.loads(a.summary.read_text());manifest=root/'compatibility/feff10.json';inventory=json.loads(manifest.read_text())
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip():p.error('commit the tested source before recording release evidence')
provenance=summary.get('provenance',{})
if provenance.get('rustCommit') != commit:p.error('summary provenance must identify the current commit')
if provenance.get('dirty') is not False:p.error('summary must come from a clean source tree')
if provenance.get('feffCommit') != inventory['upstream']['revision']:p.error('summary reference revision does not match the pinned FEFF revision')
toolchain=subprocess.check_output(['rustc','--version'],cwd=root,text=True).strip()
if provenance.get('rustCompiler') != toolchain:p.error('summary compiler must match the current pinned compiler')
for field in ['rustBinarySha256','feffDriverSha256']:
 value=provenance.get(field,'')
 if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):p.error('summary lacks binary provenance: '+field)
if sorted(row['id'] for row in summary.get('workflows',[])) != sorted(inventory['stock_workflows']):p.error('summary must cover the complete stock workflow inventory exactly once')
if any(row.get('status') != 'pass' for row in summary['workflows']):p.error('every workflow must pass')
files={}
for workflow in inventory['stock_workflows']:
 directory=a.inputs/workflow
 if not (directory/'feff.inp').is_file():p.error('missing input for '+workflow)
 for source in directory.rglob('*'):
  if source.is_symlink():p.error('evidence inputs must not contain symlinks: '+str(source))
  if source.is_file():files[source.relative_to(a.inputs).as_posix()]=source
hashes={name:hashlib.sha256(source.read_bytes()).hexdigest() for name,source in files.items()}
if summary.get('input_sha256') != hashes:p.error('inputs changed since testing, or summary lacks the test-time input_sha256 inventory')
evidence={'schema_version':1,'commit':commit,'dirty':False,'reference_commit':inventory['upstream']['revision'],'toolchain':toolchain,'input_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'input_sha256':hashes,'workflows':summary['workflows'],'provenance':summary['provenance']}
a.output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED) as archive:
 archive.writestr('evidence.json',json.dumps(evidence,indent=2)+'\n')
 for name,source in files.items():archive.write(source,'inputs/'+name)
print(a.output)
