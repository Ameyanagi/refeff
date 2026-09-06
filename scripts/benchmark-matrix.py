#!/usr/bin/env python3
"""Measure cold/reuse/recompute/memory and threads; all runs use temporary cwds.

Self-contained inputs only for memory mode (use the checked-in ZnSe example).
Build the benchmark example with the desired feature set first; its executable
hash identifies the actual linked build. The script never reuses user outputs.
"""
import argparse, hashlib, json, os, pathlib, platform, re, statistics, subprocess, tempfile, time
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',required=True,type=pathlib.Path)
p.add_argument('--input',required=True,type=pathlib.Path)
p.add_argument('--output',required=True,type=pathlib.Path)
p.add_argument('--samples',type=int,default=5)
p.add_argument('--threads',type=int,nargs='+',default=[1,2,4])
a=p.parse_args()
if a.samples<1 or any(n<1 for n in a.threads): p.error('samples and threads must be positive')
binary=a.binary.resolve(); source=a.input.resolve()
report={'schema_version':1,'platform':platform.platform(),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'measurements':[]}
with tempfile.TemporaryDirectory(prefix='refeff-bench-') as temp:
 root=pathlib.Path(temp)
 for threads in a.threads:
  for mode in ['cold','reuse','recompute','memory','typed','provenance']:
   out=root/f'{threads}-{mode}'
   env=dict(os.environ);env['REFEFF_CACHE']='provenance' if mode=='provenance' else 'audit'
   command=[str(binary),str(source),str(out),'reuse' if mode=='provenance' else mode,str(threads)]
   # Populate the directory, then discard an additional validation warmup.
   if mode in ['reuse','recompute','provenance']:
    for _ in range(2): subprocess.run(command,cwd=root,env=env,check=True,capture_output=True)
   samples=[]
   for sample in range(a.samples):
    measured=command.copy()
    if mode=='cold': measured[2]=str(out/str(sample))
    timing=['/usr/bin/time','-l' if platform.system()=='Darwin' else '-v']
    start=time.perf_counter();result=subprocess.run(timing+measured,cwd=root,env=env,check=True,capture_output=True,text=True)
    row=json.loads(result.stdout);row['process_seconds']=time.perf_counter()-start
    match=re.search(r'(\d+)\s+maximum resident set size',result.stderr) if platform.system()=='Darwin' else re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',result.stderr)
    row['peak_rss_bytes']=int(match[1])*(1 if platform.system()=='Darwin' else 1024) if match else None
    samples.append(row)
   report['measurements'].append({'mode':mode,'threads':threads,'samples':samples,'median_seconds':statistics.median(row['seconds'] for row in samples),'stdev_seconds':statistics.pstdev(row['seconds'] for row in samples)})
   print(f'{mode}, {threads} threads: {report["measurements"][-1]["median_seconds"]:.3f}s',flush=True)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
