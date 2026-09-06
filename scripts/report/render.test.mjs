import {test} from 'node:test';
import assert from 'node:assert/strict';
import {matrixStatus, formatSpeed, extent, displayIndices, renderHtml} from './render.mjs';
const base = () => ({cases:[{comparison:{passed:true}}], provenance:{rustCommit:'a',feffCommit:'b',dirty:false,rustBinarySha256:'a'.repeat(64),feffDriverSha256:'b'.repeat(64)}});
test('missing, partial and mismatched evidence never certify the matrix', () => {
  const report = base();
  assert.equal(matrixStatus(report).status,'pending');
  report.workflows=[{id:'one',status:'pass'}]; report.expectedWorkflows=['one','two']; report.workflowProvenance=report.provenance;
  assert.equal(matrixStatus(report).status,'pending');
  report.expectedWorkflows=['one']; report.workflowProvenance={rustCommit:'old',feffCommit:'b'};
  assert.equal(matrixStatus(report).status,'pending');
  report.workflowProvenance=report.provenance;
  assert.equal(matrixStatus(report).status,'pass');
  report.workflows[0].status='review'; assert.equal(matrixStatus(report).status,'review');
  report.workflows[0].status='fail'; assert.equal(matrixStatus(report).status,'review');
  report.workflows[0].status='pending'; assert.equal(matrixStatus(report).status,'review');
  report.cases=[]; assert.notEqual(matrixStatus(report).status,'pass');
});
test('timing labels correctly describe slow and invalid samples',()=>{
  assert.equal(formatSpeed(.5),'2.00× slower');assert.equal(formatSpeed(2),'2.00× faster');
  assert.equal(formatSpeed(0),'not measured');
});
test('large arrays have bounded plots and preserve isolated extrema',()=>{
  const points=Array.from({length:1_000_000},(_,i)=>Math.sin(i)); points[513]=900; points[561]=-901;
  assert.deepEqual(extent(points),[-901,900]);
  const selected=displayIndices(points); assert.ok(selected.length<=1202);
  assert.ok(selected.includes(513));assert.ok(selected.includes(561)); assert.equal(selected.at(-1),points.length-1);
});
test('portable HTML embeds syntactically valid JavaScript and escapes source data',()=>{
  const html=renderHtml({...base(), title:'</script><script>alert(1)</script>'});
  const script=html.match(/<script>([\s\S]*)<\/script>/)[1];
  assert.doesNotThrow(()=>new Function(script));
  assert.ok(!script.includes('</script><script>'));assert.match(html,/aria-pressed/);assert.match(html,/@media print/);
});

test('rendered controls run: review styling, lazy cards, filters and full CSV rows',async()=>{
  const {runInNewContext}=await import('node:vm');
  const all=[], ids=new Map(), blobs=[];
  class Element {
    constructor(tag='div'){this.tagName=tag;this.children=[];this.style={};this.dataset={};this.attributes={};this.className='';this.hidden=false;this.clientWidth=400;all.push(this);this.classList={toggle:(name,on)=>{const values=new Set(this.className.split(' '));on?values.add(name):values.delete(name);this.className=[...values].join(' ');}};}
    set innerHTML(html){this.html=html;this.children=[];
      for(const match of html.matchAll(/<button\s+([^>]+)>([^<]*)<\/button>/g)){const button=new Element('button');button.textContent=match[2];button.className=match[1].match(/class="([^"]*)"/)?.[1]??'';button.dataset.filter=match[1].match(/data-filter="([^"]*)"/)?.[1];button.setAttribute('aria-pressed',match[1].match(/aria-pressed="([^"]*)"/)?.[1]);this.children.push(button);}
    }
    get innerHTML(){return this.html??'';}
    appendChild(child){this.children.push(child);return child;}
    querySelector(selector){let child=this.children.find(node=>selector.startsWith('.')?node.className.split(' ').includes(selector.slice(1)):node.tagName===selector);if(!child){child=new Element(selector==='svg'?'svg':'div');if(selector.startsWith('.'))child.className=selector.slice(1);this.children.push(child);}return child;}
    setAttribute(key,value){this.attributes[key]=value;}
    addEventListener(event,callback){this['on'+event]=callback;}
    before(){} replaceWith(){} click(){}
  }
  const document={createElement:tag=>new Element(tag),getElementById:id=>{if(!ids.has(id))ids.set(id,new Element());return ids.get(id);},querySelectorAll:selector=>all.filter(node=>selector.startsWith('.')?node.className.split(' ').includes(selector.slice(1)):node.tagName===selector)};
  const stats={averageSeconds:1,medianSeconds:1,standardDeviationSeconds:0,samples:[1],maximumSeconds:1,successful:1,runs:1};
  const item={id:'EXAFS/Cu',title:'Cu',subtitle:'fixture',output:'chi.dat',columns:['k','chi'],xColumn:0,files:{rust:'r',feff:'f'},benchmark:{rust:stats,feff:stats,speedup:1},comparison:{passed:false,rows:2,maxRelativeL2:.1,columns:[{name:'chi',relativeL2:.1,maxAbsolute:.1,passed:false}],x:{feff:[1,2],rust:[1,2]},series:[{label:'χ(k)',color:'#00f',feff:[1,2],rust:[1.1,2],residual:[.1,0]}]}};
  const report={...base(),generatedAt:'2026-09-06',cases:[item],inputStage:{speedup:.5,rust:stats,feff:stats},method:{test:'fixture'},provenance:{...base().provenance,memoryGiB:1},galleryCases:[{...item,id:'g',family:'EXAFS',workflow:'one',status:'review',evidence:'fixture',plottedMaxRelativeL2:.1,plottedMaxAbsolute:.1}]};
  const observers=[];
  class IntersectionObserver {constructor(callback){this.callback=callback;observers.push(this);}observe(target){this.target=target;}unobserve(){}}
  const script=renderHtml(report).match(/<script>([\s\S]*)<\/script>/)[1];
  runInNewContext(script,{document,window:{addEventListener(){}},IntersectionObserver,Blob,URL:{createObjectURL:blob=>{blobs.push(blob);return 'blob:fixture';},revokeObjectURL(){}},setTimeout(){},Image:class{} });
  assert.match(ids.get('cases').children[0].innerHTML,/status review/);
  assert.match(ids.get('input-stage').innerHTML,/2.00× slower/);
  assert.equal(observers.length,1);
  const gallery=ids.get('gallery').children[0];assert.equal(gallery.querySelector('.chart').innerHTML,'');
  observers[0].callback([{target:gallery,isIntersecting:true}]);assert.match(gallery.querySelector('.chart').innerHTML,/<svg/);
  const buttons=document.querySelectorAll('.gallery-filter');buttons[1].onclick();assert.equal(buttons[1].attributes['aria-pressed'],'true');assert.equal(buttons[0].attributes['aria-pressed'],'false');
  all.find(node=>node.textContent==='CSV · all points').onclick();
  const csv=await blobs.at(-1).text();assert.equal(csv.split('\n').length,3);assert.match(csv,/1\.1/);
});
