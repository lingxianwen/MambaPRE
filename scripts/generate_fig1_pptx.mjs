import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const root = path.resolve(import.meta.dirname, '../..');
const runtime = 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const skill = 'C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
process.env.RUNTIME_NODE_MODULES = path.join(runtime, 'node/node_modules');
const { Presentation, PresentationFile } = await import(pathToFileURL(path.join(process.env.RUNTIME_NODE_MODULES, '@oai/artifact-tool/dist/artifact_tool.mjs')));
const build = path.join(root, 'tmp/fig1_pptx_final');
await fs.mkdir(build, { recursive: true });
const pres = Presentation.create({ slideSize: { width: 1800, height: 600 } });
const slide = pres.slides.add();
slide.background.fill = '#FFFFFF';
const C = { ink:'#273444', edge:'#74808A', blue:'#357DB1', bf:'#EDF6FD', teal:'#238478', tf:'#EDF8F4', purple:'#7965A8', pf:'#F4F0FA', orange:'#B76A26', of:'#FFF6E8' };
const X=x=>30+x*100, Y=y=>30+(5.45-y)*100;
let count=0;
function box(name,x,y,w,h,fill='#FFFFFF',stroke=C.edge,radius=7,lw=1.8,dashed=false) {
  return slide.shapes.add({name,geometry:'rect',position:{left:X(x),top:Y(y+h),width:w*100,height:h*100},fill,line:{fill:stroke,width:lw,style:dashed?'dashed':'solid'},borderRadius:radius});
}
function txt(name,text,x,y,w,h,size=25,bold=false,align='center',color=C.ink,font='Arial') {
  const s=slide.shapes.add({name,geometry:'textbox',position:{left:X(x),top:Y(y+h/2),width:w*100,height:h*100},fill:'none',line:{fill:'none',width:0}});
  s.text=text;
  s.text.style={typeface:font,fontSize:size,bold,color,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:{left:0,right:0,top:0,bottom:0}};
  return s;
}
function line(name,points,color=C.ink,width=2.3,dashed=false,arrow=false,fill='none') {
  const pts=points.map(([x,y])=>[X(x),Y(y)]);
  if(arrow){
    const [ax,ay]=pts.at(-2),[bx,by]=pts.at(-1),a=Math.atan2(by-ay,bx-ax),s=10;
    pts.push([bx-s*Math.cos(a-.45),by-s*Math.sin(a-.45)],[bx,by],[bx-s*Math.cos(a+.45),by-s*Math.sin(a+.45)]);
  }
  const minx=Math.min(...pts.map(p=>p[0])),miny=Math.min(...pts.map(p=>p[1]));
  const w=Math.max(1,Math.max(...pts.map(p=>p[0]))-minx),h=Math.max(1,Math.max(...pts.map(p=>p[1]))-miny);
  return slide.shapes.add({name,geometry:'custom',position:{left:minx,top:miny,width:w,height:h},fill,line:{fill:color,width,style:dashed?'dashed':'solid'},customPaths:[{width:w,height:h,commands:pts.map((p,i)=>({[i?'lineTo':'moveTo']:{x:p[0]-minx,y:p[1]-miny}}))}]});
}
function dot(name,x,y,r,color=C.ink,stroke='none'){
 return slide.shapes.add({name,geometry:'ellipse',position:{left:X(x-r),top:Y(y+r),width:200*r,height:200*r},fill:color,line:{fill:stroke,width:stroke==='none'?0:1.4}});
}
function variable(name,base,sup,sub,x,y,size=26){
 txt(name+' base',base,x,y,.30,.34,size,false,'left',C.ink,'Cambria Math');
 if(sup)txt(name+' superscript',sup,x+.24,y+.14,.35,.24,size*.62,false,'left',C.ink,'Cambria Math');
 if(sub)txt(name+' subscript',sub,x+.24,y-.11,.35,.24,size*.62,false,'left',C.ink,'Cambria Math');
}

// The figure is reconstructed from fig1_overview.tex in native PowerPoint objects.
box('Stage A: Input encoding',0,1.18,3.15,4.22,C.bf);
box('Stage B: Repeated dual-stream block',3.38,1.18,9.45,4.22,'#F5FAF8');
box('Stage C: Field prediction',13.06,1.18,4.34,4.22,C.pf);
txt('Stage A title','(a) Input encoding',.12,5.08,2.91,.42,26,true);
txt('Stage B title','(b) Dual-stream block × N',4.02,5.08,8.15,.42,26,true);
txt('Stage C title','(c) Field prediction',13.23,5.08,4,.42,26,true);

for(const [j,y] of [3.82,4.05,4.28].entries()){
 box(`Server icon tray ${j+1}`,.31,y,.64,.17,'#FFFFFF',C.blue,3,1.4);
 dot(`Server status ${j+1}`,.4,y+.085,.025,C.teal);
 line(`Server slot ${j+1}`,[[.52,y+.085],[.86,y+.085]],C.blue,1.1);
}
txt('Protocol message label','Protocol\nmessage',1.08,4.29,1.93,.62,27,true,'left');
txt('Input sequence base','x',1.08,3.86,.3,.32,25,false,'left',C.ink,'Cambria Math');
txt('Input sequence subscript','1:L',1.25,3.76,.55,.23,17,false,'left',C.ink,'Cambria Math');
['03','00','02','1C','⋯'].forEach((v,i)=>{let x=.35+.5*i;box('Input byte '+i,x,3.30,.45,.38);txt('Input byte value '+i,v,x,3.49,.45,.30,22,false,'center',C.ink,'Consolas');});
line('Bytes to embedding',[[1.575,3.23],[1.575,2.92]],C.ink,2.3,false,true);
box('Embedding module',.28,1.56,2.59,1.34);
for(let i=0;i<3;i++)for(let j=0;j<3;j++)box(`Embedding cell ${i},${j}`,.48+.16*i,2.29+.16*j,.125,.125,'#B8D2E6',C.blue,0,.8);
txt('Byte embedding label','Byte\nembedding',1.03,2.55,1.8,.53,23,true,'left');
txt('Position embedding label','+ position',1.03,2.13,1.8,.30,22,false,'left');
txt('Embedding tensor','H⁰ ∈ ℝ',.60,1.82,1.35,.40,25,false,'center',C.ink,'Cambria Math');
txt('Embedding tensor dimensions','L × d',1.64,1.96,.78,.26,17,false,'left',C.ink,'Cambria Math');

box('Bi-Mamba module',4.33,3.05,3.22,1.38,C.bf);
const nodes=[[4.55,3.78],[4.78,4.13],[5.04,3.82],[5.05,4.17]];
[[0,1],[1,2],[0,3]].forEach(([a,b],i)=>line('State graph edge '+i,[nodes[a],nodes[b]],C.blue,1.5));
nodes.forEach(([x,y],i)=>dot('State graph node '+i,x,y,.065,'#FFFFFF',C.blue));
txt('Bi-Mamba title','Bi-Mamba',5.27,4.17,2.18,.30,27,true,'left');
txt('Bi-Mamba residual label','Global + residual',5.27,3.88,2.20,.32,25,false,'left');
line('Forward scan',[[4.68,3.44],[5.73,3.44]],C.blue,2.2,false,true);
line('Backward scan',[[5.73,3.20],[4.68,3.20]],C.blue,2.2,false,true);
txt('Scan direction labels','forward\nbackward',5.91,3.32,1.45,.53,21,false,'left');

box('Local convolution module',4.33,1.55,3.22,1.21,C.tf);
txt('Local Conv1D title','Local Conv1D',5.27,2.56,2.2,.32,26,true,'left');
txt('Local residual label','Local + residual',5.27,2.25,2.2,.30,24,false,'left');
for(let i=0;i<4;i++)for(let j=0;j<3;j++)box(`Local convolution cell ${i},${j}`,4.48+.17*i,2.02+.17*j,.14,.14,'#D7EAE5',C.teal,0,1);
box('Sliding convolution window',4.46,2.17,.70,.18,'none',C.orange,2,2.4);
txt('DW and PW operations','DW (k = 7) + PW conv.',4.45,1.77,2.99,.30,22);

line('Embedding to input fork',[[2.87,2.20],[3.93,2.20]],C.ink,2.3,false,true);
txt('Block input tensor','H⁰',3.06,2.46,.65,.33,25,false,'center',C.ink,'Cambria Math');
dot('Input split',3.93,2.20,.038);
line('Fork to global stream',[[3.93,2.20],[3.93,3.82],[4.33,3.82]],C.ink,2.3,false,true);
line('Fork to local stream',[[3.93,2.20],[4.33,2.20]],C.ink,2.3,false,true);

box('Structure-aware fusion module',8.7,1.55,3.16,2.88,C.of);
line('Fusion filter icon',[[8.89,4.20],[9.48,4.20],[9.27,3.96],[9.27,3.72],[9.10,3.63],[9.10,3.96],[8.89,4.20]],C.orange,1.8,false,false,'#F2E2D2');
line('Filter icon crossbar',[[8.96,4.10],[9.39,4.10]],C.orange,1.5);
txt('Structure-aware fusion title','Structure-aware\nfusion',9.63,4.02,2.18,.66,25,true,'left');
txt('Gate label','Channel-wise byte gate',8.84,3.42,2.89,.29,22);
variable('Gate equation variable','G','ℓ','',9.76,3.10,25);
txt('Gate sigmoid equation','= σ(·)',10.24,3.10,1.15,.30,25,false,'left',C.ink,'Cambria Math');
line('Fusion divider',[[8.91,2.88],[11.65,2.88]],'#D9B996',1.1);
variable('Mixture gate term','G','ℓ','',9.56,2.59,29);
txt('First product operator','⊙',10.02,2.59,.40,.35,29,false,'center',C.ink,'Cambria Math');
variable('Mixture local tensor','H','ℓ','l',10.48,2.59,29);
txt('Second term prefix','+ (1 −',9.00,2.21,1.08,.35,28,false,'left',C.ink,'Cambria Math');
variable('Complement gate term','G','ℓ','',10.10,2.21,28);
txt('Second product operator',') ⊙',10.57,2.21,.66,.35,28,false,'left',C.ink,'Cambria Math');
variable('Mixture global tensor','H','ℓ','g',11.17,2.21,28);
txt('Mixture description','Learned local/global mixture\nat every byte position',8.83,1.79,2.9,.42,20);
line('Global tensor to fusion',[[7.55,3.82],[8.7,3.82]],C.ink,2.3,false,true);
variable('Global stream tensor','H','ℓ','g',7.93,4.12,25);
line('Local tensor to fusion',[[7.55,2.20],[8.7,2.20]],C.ink,2.3,false,true);
variable('Local stream tensor','H','ℓ','l',7.93,2.50,25);
line('Fusion output to head split',[[11.86,2.85],[13.30,2.85]],C.ink,2.3,false,true);
txt('Final hidden tensor','Hᴺ',12.12,3.10,.80,.36,25,false,'center',C.ink,'Cambria Math');

box('Boundary head module',13.53,3.83,3.52,.62);
box('Checklist icon outline',13.76,3.90,.43,.47,'#FFFFFF',C.purple,3,1.4);
for(const [j,y] of [4.02,4.14,4.26].entries()){
 line('Checklist tick '+j,[[13.80,y],[13.84,y-.04],[13.90,y+.045]],C.teal,1.4);
 line('Checklist line '+j,[[13.94,y],[14.14,y]],C.purple,1.1);
}
txt('Boundary head title','Boundary head',14.35,4.14,2.11,.40,26,true,'left');
txt('Boundary prediction symbol','ŷᵇ',16.58,4.14,.42,.38,27,false,'center',C.ink,'Cambria Math');
box('Auxiliary heads module',13.53,2.54,3.52,.63);
txt('Auxiliary heads label','Role head ŷʳ / Type head ŷᵗ',13.64,2.855,3.31,.42,23);
dot('Head split',13.30,2.85,.038);
line('Hidden tensor to boundary head',[[13.30,2.85],[13.30,4.14],[13.53,4.14]],C.ink,2.3,false,true);
line('Hidden tensor to auxiliary heads',[[13.30,2.85],[13.53,2.85]],C.ink,2.3,false,true);
line('Boundary output to field spans',[[15.29,3.83],[15.29,3.61]],C.ink,2.3,false,true);
box('Output field 1',13.68,3.27,.66,.34,'#D3E4F0');
box('Output field 2',14.39,3.27,.66,.34,'#CEE4DF');
box('Output remaining fields',15.10,3.27,1.81,.34,'#E1DBEB');
txt('Field 1 label','f₁',13.68,3.44,.66,.30,25,false,'center',C.ink,'Cambria Math');
txt('Field 2 label','f₂',14.39,3.44,.66,.30,25,false,'center',C.ink,'Cambria Math');
txt('Remaining fields label','f₃  ⋯  fₖ',15.10,3.44,1.81,.30,25,false,'center',C.ink,'Cambria Math');
txt('Dimensions and metadata note','d = 128, N = 4; per-byte outputs\nNo protocol ID at inference',13.33,1.87,3.92,.60,23);

box('Training-only supervision lane',3.38,.18,14.02,.78,C.of,C.orange,7,2.0,true);
txt('Training-only label','Training only',3.60,.59,1.88,.36,26,true,'left',C.orange);
txt('Gold roles','Gold role yʳ\nstructure / payload',5.55,.60,2.22,.54,22,false,'left');
line('Gold roles to target',[[7.78,.59],[8.70,.59]],C.orange,2.3,true,true);
txt('Gate training target','Role-derived structure target\nGate regularizer ℒg',8.85,.60,2.88,.54,21);
line('Role supervision to fusion',[[10.28,.96],[10.28,1.55]],C.orange,2.3,true,true);
txt('Gold labels and task losses','Gold labels yᵇ, yʳ, yᵗ\nℒb + λrℒr + λtℒt',13.40,.60,3.68,.54,23);
line('Supervision bus',[[17.25,.96],[17.25,4.14]],C.orange,2.3,true);
line('Boundary supervision',[[17.25,4.14],[17.05,4.14]],C.orange,2.3,true,true);
line('Semantic supervision',[[17.25,2.855],[17.05,2.855]],C.orange,2.3,true,true);
line('Feature flow legend line',[[.14,.78],[.55,.78]],C.ink,2.3,false,true);
txt('Feature flow legend','feature flow',.66,.78,2.25,.32,22,false,'left');
line('Supervision legend line',[[.14,.40],[.55,.40]],C.orange,2.3,true,true);
txt('Supervision legend','supervision',.66,.40,2.25,.32,22,false,'left',C.orange);

slide.speakerNotes.textFrame.setText('Source: ICASSP2026_Paper_Templates/fig1_overview.tex, current Mamba-PRE Figure 1. Byte values and field spans are schematic. Solid paths carry features; dashed orange paths are training-only supervision.');
await (await PresentationFile.exportPptx(pres)).save(path.join(build,'candidate.pptx'));
await fs.writeFile(path.join(build,'model.json'),JSON.stringify(pres.toProto()));
const preview=await pres.export({slide,format:'png',scale:1});
await fs.writeFile(path.join(build,'preview.png'),new Uint8Array(await preview.arrayBuffer()));
console.log('Draft exported: '+path.join(build,'candidate.pptx'));
const { finalizePresentation }=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const output=path.join(root,'ICASSP2026_Paper_Templates/Fig1_editable_final.pptx');
await finalizePresentation({workspaceDir:root,candidatePath:path.join(build,'candidate.pptx'),finalPath:output,pythonExecutable:path.join(runtime,'python/python.exe'),integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','17145000,5715000'],explicitTotalSlideCount:1,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'design',families:['Arial','Cambria Math','Consolas']},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log('Final: '+output);
