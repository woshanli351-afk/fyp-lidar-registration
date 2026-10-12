const COLORS={r1:'#42c9f5',r3:'#ff9a3d',green:'#3ee6a8',yellow:'#ffd166',red:'#ff6b6b',purple:'#a889ff',cyan:'#35b9e8',muted:'#7189a0',grid:'#173049'};
const STAGES=[
  {p:0,title:'Independent local mapping',note:'Robot1 and Robot3 begin in separate local frames.',snap:'LOCAL'},
  {p:.20,title:'Scan Context candidate retrieval',note:'Ring Key narrows the search; exact Scan Context returns a Rank-1 candidate and coarse yaw.',snap:'LOCAL'},
  {p:.34,title:'GICP geometric verification',note:'Coarse-to-fine GICP checks geometric consistency against the frozen 0.6091 threshold.',snap:'LOCAL'},
  {p:.47,title:'First inter-robot bridge',note:'The first verified candidate supplies a metric candidate-to-query transform.',snap:'K1'},
  {p:.58,title:'Progressive batch fusion',note:'Accepted constraints accumulate through stable K10 batch snapshots.',snap:'K10'},
  {p:.90,title:'Final clean map',note:'All 75 frozen clean loop factors produce the final K75 batch result.',snap:'K75'}
];
let D=null,S={p:0,playing:false,speed:1,mapOverride:null,view:'system'},last=0;
const $=q=>document.querySelector(q),$$=q=>[...document.querySelectorAll(q)];

fetch('data/demo.json').then(r=>{if(!r.ok)throw new Error(`data/demo.json ${r.status}`);return r.json()}).then(data=>{D=data;const params=new URLSearchParams(location.search),preview=Number(params.get('p'));if(Number.isFinite(preview))S.p=Math.max(0,Math.min(1,preview));if(params.get('view')==='analysis')S.view='analysis';bind();render();requestAnimationFrame(tick)}).catch(error=>{$('#sceneTitle').textContent='Demo data failed to load';console.error(error)});

function bind(){
  document.body.classList.toggle('system-view',S.view==='system');
  $('#systemView').classList.toggle('active',S.view==='system');
  $('#analysisView').classList.toggle('active',S.view==='analysis');
  $('#systemView').onclick=()=>setView('system');
  $('#analysisView').onclick=()=>setView('analysis');
  $('#play').onclick=togglePlay;
  $('#reset').onclick=resetReplay;
  $('#fullscreen').onclick=()=>document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();
  $('#speed').onchange=e=>S.speed=+e.target.value;
  $('#timeline').oninput=e=>{S.p=+e.target.value/1000;S.mapOverride=null;render()};
  $$('.stagebar button').forEach((button,index)=>button.onclick=()=>jumpToStage(index));
  $$('.snapshot-dots button').forEach(button=>button.onclick=()=>{S.mapOverride=button.dataset.map;renderMap()});
  $('#accuracyDirection').onchange=renderAccuracy;
  $('#efficiencyDirection').onchange=renderEfficiency;
  document.addEventListener('keydown',event=>{
    if(event.target.matches('select,input'))return;
    if(event.code==='Space'){event.preventDefault();togglePlay()}
    else if(event.key.toLowerCase()==='r')resetReplay();
    else if(/^[1-6]$/.test(event.key))jumpToStage(Number(event.key)-1);
  });
}

function setView(view){
  S.view=view;
  document.body.classList.toggle('system-view',view==='system');
  $('#systemView').classList.toggle('active',view==='system');
  $('#analysisView').classList.toggle('active',view==='analysis');
  render();
}

function togglePlay(){S.playing=!S.playing;$('#play').textContent=S.playing?'❚❚ Pause':'▶ Play'}
function resetReplay(){S.p=0;S.playing=false;S.mapOverride=null;$('#play').textContent='▶ Play';render()}
function jumpToStage(index){S.p=STAGES[index].p;S.playing=false;S.mapOverride=null;$('#play').textContent='▶ Play';render()}

function tick(time){
  if(D&&S.playing){S.p=Math.min(1,S.p+(time-last)/60000*S.speed);if(S.p===1){S.playing=false;$('#play').textContent='▶ Play'}render()}
  last=time;requestAnimationFrame(tick);
}

function stageIndex(){let out=0;STAGES.forEach((stage,index)=>{if(S.p>=stage.p)out=index});return out}
function loopProgress(){if(S.p<.47)return 0;if(S.p>=.90)return 75;return Math.max(1,Math.min(75,Math.floor(1+(S.p-.47)/.43*74)))}
function activeEvent(){if(S.p<.20)return null;const count=loopProgress();return D.events[Math.max(0,Math.min(count,D.events.length)-1)]}
function snapshot(){const loops=loopProgress();if(loops===0)return'local';if(loops<10)return'K1';if(loops<75)return'K10';return'K75'}

function render(){
  const idx=stageIndex(),stage=STAGES[idx],loops=loopProgress();
  const seconds=Math.round(S.p*60);$('#timeline').value=Math.round(S.p*1000);$('#timeLabel').textContent=`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')} / 01:00`;
  $('#sceneTitle').textContent=stage.title;$('#sceneNote').textContent=stage.note;$('#snapshotName').textContent=snapshot().toUpperCase();$('#loopCount').textContent=`${loops} / 75`;
  $('#graphState').textContent=loops?'1 connected graph':'2 components';
  $('#systemPhase').textContent=stage.title.toUpperCase();$('#systemLoops').textContent=`${loops} / 75`;
  const event=activeEvent();$('#systemEvent').textContent=event?`${event.query_robot} #${event.query_keyframe} ↔ ${event.candidate_robot} #${event.candidate_keyframe}`:'Waiting for inter-robot candidate';
  $('#systemQuality').textContent=event&&idx>=2?`${event.gicp_quality.toFixed(4)} / ${D.metadata.gicp_threshold.toFixed(4)} threshold`:'—';
  $$('.stagebar button').forEach((button,index)=>button.classList.toggle('active',index===idx));
  renderTrajectory();renderEvent();renderMap();renderAccuracy();renderEfficiency();
}

function bounds(groups){
  const points=groups.flatMap(group=>Object.values(group).flat());let xs=points.map(p=>p[1]),ys=points.map(p=>p[2]);
  let minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys),pad=Math.max(maxx-minx,maxy-miny)*.08;
  return[minx-pad,miny-pad,maxx+pad,maxy+pad];
}
function xy(point,b,w,h){const scale=Math.min((w-64)/(b[2]-b[0]),(h-64)/(b[3]-b[1])),ox=(w-(b[2]-b[0])*scale)/2,oy=(h-(b[3]-b[1])*scale)/2;return[ox+(point[1]-b[0])*scale,h-oy-(point[2]-b[1])*scale]}
function drawGrid(ctx,w,h){ctx.clearRect(0,0,w,h);ctx.fillStyle='#07111e';ctx.fillRect(0,0,w,h);ctx.strokeStyle=COLORS.grid;ctx.lineWidth=1;for(let x=40;x<w;x+=80){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke()}for(let y=40;y<h;y+=80){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}}
function lerpTrajectory(a,b,t){const out={};for(const robot of ['robot1','robot3']){const aa=a[robot],bb=b[robot],n=Math.min(aa.length,bb.length);out[robot]=aa.slice(0,n).map((p,i)=>[p[0],p[1]*(1-t)+bb[i][1]*t,p[2]*(1-t)+bb[i][2]*t])}return out}
function currentTrajectory(){
  if(S.p<.47)return D.trajectories.local;
  if(S.p<.58)return lerpTrajectory(D.trajectories.K1,D.trajectories.K10,(S.p-.47)/.11);
  if(S.p<.90)return lerpTrajectory(D.trajectories.K10,D.trajectories.K75,(S.p-.58)/.32);
  return D.trajectories.K75;
}
function renderTrajectory(){
  const canvas=$('#trajectory'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height,tr=currentTrajectory(),snap=snapshot(),map=snap==='local'?null:D.map_points[snap];
  const boundGroups=[tr];
  if(map)boundGroups.push({robot1:map.robot1.map(p=>[0,p[0],p[1]]),robot3:map.robot3.map(p=>[0,p[0],p[1]])});
  const b=bounds(boundGroups);drawGrid(ctx,w,h);
  if(map){
    for(const [robot,color] of [['robot1',COLORS.r1],['robot3',COLORS.r3]]){
      ctx.fillStyle=color;ctx.globalAlpha=.20;
      for(let i=0;i<map[robot].length;i+=2){const p=map[robot][i],u=xy([0,p[0],p[1]],b,w,h);ctx.fillRect(u[0],u[1],1.5,1.5)}
    }
    ctx.globalAlpha=1;
  }
  const loops=loopProgress();
  for(let i=0;i<loops;i++){
    const event=D.events[i],q=findPoint(tr,event.query_robot,event.query_keyframe),c=findPoint(tr,event.candidate_robot,event.candidate_keyframe);if(!q||!c)continue;
    const a=xy(q,b,w,h),z=xy(c,b,w,h);ctx.strokeStyle=i===loops-1?COLORS.green:'rgba(62,230,168,.22)';ctx.lineWidth=i===loops-1?2.4:1;ctx.setLineDash(i===loops-1?[7,5]:[]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...z);ctx.stroke();
  }
  ctx.setLineDash([]);
  for(const robot of ['robot1','robot3']){
    const pts=tr[robot],color=robot==='robot1'?COLORS.r1:COLORS.r3;ctx.strokeStyle=color;ctx.lineWidth=2.7;ctx.beginPath();pts.forEach((p,i)=>{const u=xy(p,b,w,h);i?ctx.lineTo(...u):ctx.moveTo(...u)});ctx.stroke();ctx.fillStyle=color;pts.filter((_,i)=>i%12===0).forEach(p=>{const u=xy(p,b,w,h);ctx.beginPath();ctx.arc(u[0],u[1],2.2,0,Math.PI*2);ctx.fill()});
    const travel=Math.min(1,S.p/.2),marker=pts[Math.min(pts.length-1,Math.floor(travel*(pts.length-1)))],m=xy(marker,b,w,h);ctx.shadowColor=color;ctx.shadowBlur=16;ctx.beginPath();ctx.arc(m[0],m[1],8,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;ctx.fillStyle='#07111e';ctx.font='bold 9px system-ui';ctx.textAlign='center';ctx.fillText(robot==='robot1'?'R1':'R3',m[0],m[1]+3);
  }
  ctx.textAlign='left';
  const event=loops?activeEvent():null;if(event){const q=findPoint(tr,event.query_robot,event.query_keyframe),c=findPoint(tr,event.candidate_robot,event.candidate_keyframe);if(q&&c){for(const p of [xy(q,b,w,h),xy(c,b,w,h)]){ctx.fillStyle=COLORS.green;ctx.beginPath();ctx.arc(p[0],p[1],6,0,Math.PI*2);ctx.fill()}}}
  ctx.fillStyle='#9bb4ca';ctx.font='14px system-ui';ctx.fillText(`${snap.toUpperCase()} · ${loops} verified inter-robot constraints`,18,27);ctx.fillStyle='#607d96';ctx.font='11px ui-monospace,monospace';ctx.fillText(loops?'WORLD / COMMON FRAME':'ROBOT1 LOCAL + ROBOT3 LOCAL FRAMES',18,h-18);
}
function findPoint(tr,robot,keyframe){const list=tr[robot];if(!list)return null;let best=list[0],distance=Infinity;for(const point of list){const d=Math.abs(point[0]-keyframe);if(d<distance){best=point;distance=d}}return best}

function renderEvent(){
  const event=activeEvent(),idx=stageIndex();
  if(!event){$('#eventGrid').className='event-grid muted-event';$('#eventGrid').innerHTML='<span>Replay has not reached the first candidate.</span>';$('#eventState').className='state waiting';$('#eventState').textContent='WAITING';$('#qualityBar').style.width='0';$('#qualityValue').textContent='—';$('#eventMessage').textContent='Robot maps remain independent.';drawCloud(null,'waiting');return}
  const mode=idx===1?'candidate':idx===2?'verifying':'accepted';
  $('#eventState').className='state '+(mode==='accepted'?'verified':'candidate');$('#eventState').textContent=mode==='candidate'?'RANK-1':mode==='verifying'?'VERIFYING':'VERIFIED';
  $('#eventGrid').className='event-grid';$('#eventGrid').innerHTML=`<div><small>Query</small><b>${event.query_robot} #${event.query_keyframe}</b></div><div><small>Candidate</small><b>${event.candidate_robot} #${event.candidate_keyframe}</b></div><div><small>SC score</small><b>${event.sc_score.toFixed(4)} · Rank 1</b></div><div><small>SC yaw init</small><b>${event.sc_yaw_deg.toFixed(1)}° · shift ${event.best_shift}</b></div>`;
  const qualityVisible=mode!=='candidate';$('#qualityBar').style.width=qualityVisible?`${Math.min(100,event.gicp_quality*100)}%`:'0';$('#qualityValue').textContent=qualityVisible?event.gicp_quality.toFixed(4):'pending';
  $('#eventMessage').textContent=mode==='candidate'?'Rank-1 candidate selected; GICP has not accepted a map edge yet.':mode==='verifying'?`Measured quality ${event.gicp_quality.toFixed(4)}; checking against threshold ${D.metadata.gicp_threshold.toFixed(4)}.`:`Loop ${event.index}/75 accepted; candidate → query metric transform is available.`;drawCloud(event,mode);
}
function cloudBounds(arrays){const pts=arrays.flat();if(!pts.length)return[-1,-1,1,1];let xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);let m=Math.max(6,...xs.map(Math.abs),...ys.map(Math.abs));return[-m,-m,m,m]}
function cloudXY(p,b,w,h){return[(p[0]-b[0])/(b[2]-b[0])*w,h-(p[1]-b[1])/(b[3]-b[1])*h]}
function drawCloud(event,mode='waiting'){
  for(const [id,aligned] of [['#cloudRaw',false],['#cloudAligned',true]]){const c=$(id),x=c.getContext('2d'),w=c.width,h=c.height;x.fillStyle='#07111e';x.fillRect(0,0,w,h);if(!event||aligned&&mode==='candidate')continue;const candidate=event[aligned?'candidate_gicp_aligned_xy':'candidate_raw_xy'],b=cloudBounds([event.query_cloud_xy,candidate]);x.globalAlpha=.85;x.fillStyle=COLORS.r1;event.query_cloud_xy.forEach(p=>{const u=cloudXY(p,b,w,h);x.fillRect(u[0],u[1],2.4,2.4)});x.fillStyle=aligned&&mode==='accepted'?COLORS.green:COLORS.r3;candidate.forEach(p=>{const u=cloudXY(p,b,w,h);x.fillRect(u[0],u[1],2.4,2.4)});x.globalAlpha=1}
}

function renderMap(){
  const selected=S.mapOverride||snapshot(),canvas=$('#mapCanvas'),empty=$('#mapEmpty');
  $$('.snapshot-dots button').forEach(b=>b.classList.toggle('active',b.dataset.map===selected));
  if(selected==='local'){canvas.style.display='none';empty.style.display='block';$('#mapTitle').textContent='Independent frames';$('#mapTag').textContent='LOCAL';return}
  canvas.style.display='block';empty.style.display='none';
  const points=D.map_points[selected],ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height,all=[...points.robot1,...points.robot3],xs=all.map(p=>p[0]),ys=all.map(p=>p[1]),pad=8,b=[Math.min(...xs)-pad,Math.min(...ys)-pad,Math.max(...xs)+pad,Math.max(...ys)+pad];
  ctx.fillStyle='#07111e';ctx.fillRect(0,0,w,h);
  for(const [robot,color] of [['robot1',COLORS.r1],['robot3',COLORS.r3]]){ctx.fillStyle=color;ctx.globalAlpha=.55;points[robot].forEach(p=>{const u=cloudXY(p,b,w,h);ctx.fillRect(u[0],u[1],1.4,1.4)})}
  ctx.globalAlpha=1;ctx.fillStyle='#b8cee1';ctx.font='12px system-ui';ctx.fillText(`${selected} frozen batch map · ${all.length.toLocaleString()} displayed points`,12,20);
  $('#mapTitle').textContent=selected==='K1'?'First common frame':selected==='K10'?'10-loop batch snapshot':'75-loop clean batch result';$('#mapTag').textContent=selected;$('#mapTag').className='state '+(selected==='K75'?'verified':'neutral');
}

function metric(direction){return D.metrics.filter(row=>row.direction===direction)}
function chartBase(canvas){const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);ctx.fillStyle='#091523';ctx.fillRect(0,0,w,h);ctx.font='11px system-ui';return{ctx,w,h,left:94,right:18,top:20,bottom:46}}
function renderAccuracy(){if(!D)return;const direction=$('#accuracyDirection').value,rows=metric(direction),rank1=rows.find(row=>row.policy==='gicp_rank1'),sc=rows.find(row=>row.policy==='sc_rank1'),g=chartBase($('#accuracyChart')),plotW=g.w-g.left-g.right,plotH=g.h-g.top-g.bottom,minValue=Math.min(...rows.flatMap(row=>[row.precision,row.correct_pair_recall]))<.75?0:.75,step=minValue===0?.25:.05;$('#rank1Precision').textContent=`${(rank1.precision*100).toFixed(2)}%`;$('#rank1Recall').textContent=`${(rank1.correct_pair_recall*100).toFixed(2)}%`;$('#rank1Hazards').textContent=`${rank1.mapping_hazard_event_count} events`;$('#accuracyInsight').innerHTML=`Rank-1 GICP reduces protocol false positives from <b>${sc.protocol_fp_query_count.toLocaleString()}</b> to <b>${rank1.protocol_fp_query_count}</b>; accepted edges retain a metric pose.`;g.ctx.strokeStyle='#20384e';g.ctx.fillStyle=COLORS.muted;for(let p=minValue;p<=1.0001;p+=step){const y=g.top+(1-p)/(1-minValue)*plotH;g.ctx.beginPath();g.ctx.moveTo(g.left,y);g.ctx.lineTo(g.w-g.right,y);g.ctx.stroke();g.ctx.fillText(`${Math.round(p*100)}%`,g.left-36,y+4)}const group=plotW/rows.length;rows.forEach((row,i)=>{const x=g.left+i*group+group*.18,bw=group*.25;for(const [j,key,color] of [[0,'precision',COLORS.green],[1,'correct_pair_recall',COLORS.purple]]){const v=row[key],height=Math.max(0,(v-minValue)/(1-minValue)*plotH),y=g.top+plotH-height;g.ctx.fillStyle=color;g.ctx.fillRect(x+j*(bw+4),y,bw,height);g.ctx.fillStyle='#d6e5f3';g.ctx.font='9px system-ui';g.ctx.textAlign='center';g.ctx.fillText(`${(v*100).toFixed(1)}%`,x+j*(bw+4)+bw/2,Math.max(g.top+10,y-5))}g.ctx.font='11px system-ui';g.ctx.fillStyle='#9bb2c7';g.ctx.textAlign='center';g.ctx.fillText(shortLabel(row.policy),x+bw,g.h-19)});g.ctx.textAlign='left'}
function shortLabel(policy){return({sc_rank1:'SC',sc_score_gate:'SC gate',gicp_rank1:'R1 GICP',gicp_top3:'Top3',selective_gicp:'Selective'})[policy]}
function renderEfficiency(){if(!D)return;const direction=$('#efficiencyDirection').value,rows=metric(direction),rank1=rows.find(row=>row.policy==='gicp_rank1'),top3=rows.find(row=>row.policy==='gicp_top3'),g=chartBase($('#efficiencyChart')),plotW=g.w-g.left-g.right,plotH=g.h-g.top-g.bottom,max=500;$('#rank1P95').textContent=`${rank1.p95_ms.toFixed(1)} ms`;$('#budgetHeadroom').textContent=`${(500-rank1.p95_ms).toFixed(1)} ms`;$('#rank1Calls').textContent=rank1.mean_gicp_calls.toFixed(2);$('#efficiencyInsight').innerHTML=`Rank-1 stays <b>${(500-rank1.p95_ms).toFixed(1)} ms</b> below the 500 ms budget; Top-3 raises p95 to <b>${top3.p95_ms.toFixed(1)} ms</b>.`;g.ctx.strokeStyle=COLORS.red;g.ctx.setLineDash([6,5]);g.ctx.beginPath();g.ctx.moveTo(g.left,g.top);g.ctx.lineTo(g.w-g.right,g.top);g.ctx.stroke();g.ctx.setLineDash([]);g.ctx.fillStyle=COLORS.red;g.ctx.fillText('500 ms',g.left-51,g.top+4);const group=plotW/rows.length;rows.forEach((row,i)=>{const x=g.left+i*group+group*.19,bw=group*.42,height=row.p95_ms/max*plotH,y=g.top+plotH-height;g.ctx.fillStyle=COLORS.cyan;g.ctx.fillRect(x,y,bw,height);g.ctx.fillStyle='#d6e5f3';g.ctx.textAlign='center';g.ctx.fillText(`${row.p95_ms.toFixed(0)} ms`,x+bw/2,Math.max(g.top+12,y-7));if(row.mean_gicp_calls>0){g.ctx.fillStyle=COLORS.yellow;g.ctx.beginPath();g.ctx.arc(x+bw/2,g.top+plotH-(row.mean_gicp_calls/3)*plotH,5,0,Math.PI*2);g.ctx.fill()}g.ctx.fillStyle='#9bb2c7';g.ctx.fillText(shortLabel(row.policy),x+bw/2,g.h-19)});g.ctx.textAlign='left'}
