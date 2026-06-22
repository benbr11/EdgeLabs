"use strict";
const D = window.WC_DATA, P = D.params, M = D.mods, T = D.teams;
const TEAMS = Object.keys(T).sort();
const el = (h) => { const d = document.createElement("div"); d.innerHTML = h.trim(); return d.firstChild; };
const pct = (x) => x.toFixed(1) + "%";

/* ----------------------------- prediction math (mirrors simulate.py) ------- */
function poisson(lam, mg){ const o=[]; let p=Math.exp(-lam); for(let k=0;k<=mg;k++){o.push(p); p*=lam/(k+1);} return o; }
function dcMatrix(lh, la){
  const mg = Math.max(12, Math.floor(lh+la)+8);
  const ph = poisson(lh,mg), pa = poisson(la,mg), Mx = [];
  for(let i=0;i<=mg;i++){ const r=[]; for(let j=0;j<=mg;j++) r.push(ph[i]*pa[j]); Mx.push(r); }
  const rho = P.rho;
  Mx[0][0]*=Math.max(0,1-lh*la*rho); Mx[0][1]*=Math.max(0,1+lh*rho);
  Mx[1][0]*=Math.max(0,1+la*rho);    Mx[1][1]*=Math.max(0,1-rho);
  let s=0; for(let i=0;i<=mg;i++)for(let j=0;j<=mg;j++)s+=Mx[i][j];
  for(let i=0;i<=mg;i++)for(let j=0;j<=mg;j++)Mx[i][j]/=s;
  return {Mx,mg};
}
function mods(team, avail, rest, stakes, vTemp, vAlt){
  let am=1, dm=1; const t=T[team];
  if(avail<1){ const f=M.avail_floor+(1-M.avail_floor)*avail; am*=f; dm*=f; }
  if(rest<4){ const f=Math.max(0,1-M.fatigue_per_day*(4-rest)); am*=f; dm*=f; }
  const sf=M.stakes[stakes]!=null?M.stakes[stakes]:1; am*=sf; dm*=sf;
  if(vAlt!=null && vAlt>t.home_alt+M.alt_buffer){ const f=Math.max(0,1-M.alt_pen_per_km*(vAlt-t.home_alt-M.alt_buffer)/1000); am*=f; dm*=f; }
  if(vTemp!=null && vTemp>t.home_temp+M.heat_buffer){ const f=Math.max(0,1-M.heat_pen_per_c*(vTemp-t.home_temp-M.heat_buffer)); am*=f; dm*=f; }
  return {am,dm};
}
function predict(A,B,o){
  o=o||{};
  const host=o.host||null;
  const vTemp = o.vTemp!=null?o.vTemp : (host?T[host].home_temp:null);
  const vAlt  = o.vAlt!=null ?o.vAlt  : (host?T[host].home_alt :null);
  const mA=mods(A, o.availA??1, o.restA??4, o.stakesA||"normal", vTemp, vAlt);
  const mB=mods(B, o.availB??1, o.restB??4, o.stakesB||"normal", vTemp, vAlt);
  const w=M.weather[o.weather||"clear"];
  const hfA=host===A?P.home_adv:1, hfB=host===B?P.home_adv:1;
  const lamA=P.avg*(T[A].att_mult*mA.am)*(T[B].dfn_mult/mB.dm)*hfA*w;
  const lamB=P.avg*(T[B].att_mult*mB.am)*(T[A].dfn_mult/mA.dm)*hfB*w;
  const {Mx,mg}=dcMatrix(lamA,lamB);
  let pA=0,pD=0,pB=0,exA=0,exB=0; const flat=[];
  for(let i=0;i<=mg;i++)for(let j=0;j<=mg;j++){ const p=Mx[i][j];
    if(i>j)pA+=p; else if(j>i)pB+=p; else pD+=p; exA+=i*p; exB+=j*p; flat.push([p,i,j]); }
  flat.sort((x,y)=>y[0]-x[0]);
  const r={A,B,lamA,lamB,pA:pA*100,pD:pD*100,pB:pB*100,exA,exB,
           top:flat.slice(0,3).map(([p,i,j])=>({i,j,p:p*100}))};
  if(o.knockout){
    const et=dcMatrix(lamA/3,lamB/3); let qa=0,qb=0,qd=0;
    for(let i=0;i<=et.mg;i++)for(let j=0;j<=et.mg;j++){const p=et.Mx[i][j]; if(i>j)qa+=p; else if(j>i)qb+=p; else qd+=p;}
    const share=(pA+pB)>0?pA/(pA+pB):0.5, psA=Math.min(0.55,Math.max(0.45,0.5+(share-0.5)*0.2));
    r.advA=100*(pA+pD*(qa+qd*psA)); r.advB=100*(pB+pD*(qb+qd*(1-psA)));
    r.pET=pD*100; r.pPen=pD*qd*100;
  }
  return r;
}
function stageOf(A,B){ const gA=T[A]&&T[A].group, gB=T[B]&&T[B].group;
  if(gA&&gA===gB) return "group"; return P.group_complete?"knockout":"unknown"; }
function hostOf(A,B){ if(P.hosts.includes(A))return A; if(P.hosts.includes(B))return B; return null; }
const stakeTag = (s)=> s==="clinched"?'<span class="tag clinch">clinched · rests</span>'
  : s==="eliminated"?'<span class="tag elim">eliminated</span>' : '<span class="tag live">live</span>';

/* --------------------------------- PREDICT screen -------------------------- */
function predictScreen(){
  const sec=document.getElementById("predict"); sec.innerHTML="";
  const opts=TEAMS.map(t=>`<option>${t}</option>`).join("");
  const card=el(`<div class="card">
    <div class="row"><div style="flex:1"><label>Team A</label><select id="selA">${opts}</select></div>
      <div class="vs">vs</div>
      <div style="flex:1"><label>Team B</label><select id="selB">${opts}</select></div></div>
    <div class="note" id="stageNote"></div>
    <details><summary>Match-day factors (optional — auto by default)</summary>
      <div class="grid2">
        <div><label>Knockout tie?</label><select id="ko"><option value="auto">Auto-detect</option><option value="yes">Yes (ET + pens)</option><option value="no">No (90 min)</option></select></div>
        <div><label>Neutral / host</label><select id="venue"><option value="auto">Auto</option><option value="neutral">Neutral</option><option value="A">Team A at home</option><option value="B">Team B at home</option></select></div>
        <div><label>A availability <span id="avAl">100%</span></label><input type="range" id="avA" min="50" max="100" value="100"></div>
        <div><label>B availability <span id="avBl">100%</span></label><input type="range" id="avB" min="50" max="100" value="100"></div>
        <div><label>A days rest</label><input type="number" id="rA" value="4" min="0" max="14"></div>
        <div><label>B days rest</label><input type="number" id="rB" value="4" min="0" max="14"></div>
        <div><label>Weather</label><select id="wx"><option>clear</option><option>rain</option><option>cold</option><option>heat</option></select></div>
        <div><label>Venue °C (blank=auto)</label><input type="number" id="vt" placeholder="auto"></div>
      </div>
      <h4>Betting edge (optional)</h4>
      <div class="grid2" style="grid-template-columns:1fr 1fr 1fr">
        <div><label>Odds A</label><input type="number" step="0.01" id="oA" placeholder="2.10"></div>
        <div><label>Odds Draw</label><input type="number" step="0.01" id="oD" placeholder="3.40"></div>
        <div><label>Odds B</label><input type="number" step="0.01" id="oB" placeholder="3.50"></div>
      </div>
      <div class="mini" style="margin-top:6px">Model is independent of the odds — it only flags where it sees value.</div>
    </details>
    <button class="btn" id="go">Predict</button>
    <div id="out"></div>
  </div>`);
  sec.appendChild(card);
  const $=(id)=>card.querySelector(id);
  $("#selA").selectedIndex=TEAMS.indexOf("Argentina"); $("#selB").selectedIndex=TEAMS.indexOf("Brazil");
  $("#avA").oninput=()=>$("#avAl").textContent=$("#avA").value+"%";
  $("#avB").oninput=()=>$("#avBl").textContent=$("#avB").value+"%";
  const refreshStage=()=>{
    const A=$("#selA").value,B=$("#selB").value;
    if(A===B){$("#stageNote").innerHTML="Pick two different teams.";return;}
    const st=stageOf(A,B);
    let s = st==="group" ? `Group stage · ${A} ${stakeTag(T[A].stakes)} &nbsp; ${B} ${stakeTag(T[B].stakes)}`
      : st==="knockout" ? `<span class="tag ko">Knockout</span> extra time + penalties → who advances`
      : `Stage <b>unknown</b> (knockout bracket not set yet) — set "Knockout tie?" if this is a KO.`;
    $("#stageNote").innerHTML=s;
  };
  $("#selA").onchange=refreshStage; $("#selB").onchange=refreshStage; refreshStage();
  $("#go").onclick=()=>{
    const A=$("#selA").value,B=$("#selB").value; if(A===B){$("#out").innerHTML='<div class="note">Pick two different teams.</div>';return;}
    const st=stageOf(A,B);
    let ko = $("#ko").value==="auto" ? st==="knockout" : $("#ko").value==="yes";
    let host=null; const vsel=$("#venue").value;
    if(vsel==="auto") host=hostOf(A,B); else if(vsel==="A")host=A; else if(vsel==="B")host=B;
    const o={knockout:ko, host:host,
      availA:+$("#avA").value/100, availB:+$("#avB").value/100, restA:+$("#rA").value, restB:+$("#rB").value,
      weather:$("#wx").value, vTemp: $("#vt").value!==""?+$("#vt").value:null };
    if(!ko && st==="group"){ o.stakesA=T[A].stakes; o.stakesB=T[B].stakes; }
    const r=predict(A,B,o);
    const odds=[+$("#oA").value,+$("#oD").value,+$("#oB").value];
    $("#out").innerHTML=renderResult(r,ko,odds);
  };
}
function renderResult(r,ko,odds){
  const {A,B}=r; let h="";
  if(ko){
    const aw=Math.max(2,r.advA), bw=Math.max(2,r.advB);
    h+=`<div class="probbar"><div class="sA" style="width:${aw}%">${A.slice(0,3).toUpperCase()} ${r.advA.toFixed(0)}%</div>
        <div class="sB" style="width:${bw}%">${B.slice(0,3).toUpperCase()} ${r.advB.toFixed(0)}%</div></div>
        <div class="big"><div><div class="n">${r.advA.toFixed(1)}%</div><div class="l">${A} advance</div></div>
        <div><div class="n">${r.advB.toFixed(1)}%</div><div class="l">${B} advance</div></div></div>
        <div class="xg">90 min: ${A} ${r.pA.toFixed(0)}% / draw ${r.pD.toFixed(0)}% / ${B} ${r.pB.toFixed(0)}% · extra time ${r.pET.toFixed(0)}% · penalties ${r.pPen.toFixed(1)}%</div>`;
  } else {
    h+=`<div class="probbar"><div class="sA" style="width:${Math.max(3,r.pA)}%">${r.pA.toFixed(0)}%</div>
        <div class="sD" style="width:${Math.max(3,r.pD)}%">${r.pD.toFixed(0)}%</div>
        <div class="sB" style="width:${Math.max(3,r.pB)}%">${r.pB.toFixed(0)}%</div></div>
        <div class="big"><div><div class="n">${r.pA.toFixed(1)}%</div><div class="l">${A} win</div></div>
        <div><div class="n">${r.pD.toFixed(1)}%</div><div class="l">draw</div></div>
        <div><div class="n">${r.pB.toFixed(1)}%</div><div class="l">${B} win</div></div></div>`;
  }
  h+=`<div class="xg">Expected goals — ${A} <b>${r.exA.toFixed(2)}</b> · ${B} <b>${r.exB.toFixed(2)}</b></div>`;
  h+=`<h4>Most likely scorelines</h4>`;
  h+=r.top.map(s=>`<div class="scoreline"><span>${A} ${s.i} – ${s.j} ${B}</span><span class="p">${s.p.toFixed(1)}%</span></div>`).join("");
  if(odds[0]>1&&odds[1]>1&&odds[2]>1){
    const raw=[1/odds[0],1/odds[1],1/odds[2]], ov=raw[0]+raw[1]+raw[2];
    const mp=[r.pA/100,r.pD/100,r.pB/100], lab=[A,"Draw",B];
    h+=`<h4>Edge vs market (overround ${((ov-1)*100).toFixed(1)}%)</h4><table><tr><th>Outcome</th><th>Model</th><th>Mkt</th><th>Odds</th><th>EV</th></tr>`;
    for(let i=0;i<3;i++){ const ev=mp[i]*odds[i]-1;
      h+=`<tr><td>${lab[i]}</td><td>${(mp[i]*100).toFixed(1)}%</td><td>${(raw[i]/ov*100).toFixed(1)}%</td><td>${odds[i].toFixed(2)}</td>
          <td class="${ev>0?'edge':'neg'}">${ev>0?'+':''}${(ev*100).toFixed(1)}%${ev>0?' ✓':''}</td></tr>`; }
    h+=`</table>`;
  }
  return h;
}

/* --------------------------------- GROUPS screen --------------------------- */
function groupsScreen(){
  const sec=document.getElementById("groups"); sec.innerHTML="";
  D.groups.forEach(g=>{
    let rows=g.table.map(r=>{const s=T[r.team].stakes;
      return `<tr><td>${r.team} ${r.P>=2?stakeTag(s):''}</td><td>${r.P}</td><td><b>${r.pts}</b></td><td>${r.gf}-${r.ga}</td><td>${r.gd>=0?'+':''}${r.gd}</td></tr>`;}).join("");
    let rem=g.remaining.length?`<div class="mini" style="margin-top:8px">Remaining: ${g.remaining.map(m=>m[0]+" v "+m[1]).join(" · ")}</div>`:`<div class="mini" style="margin-top:8px">Group complete</div>`;
    sec.appendChild(el(`<div class="card"><h3>${g.name}</h3>
      <table><tr><th>Team</th><th>P</th><th>Pts</th><th>GF-GA</th><th>GD</th></tr>${rows}</table>${rem}</div>`));
  });
}

/* --------------------------------- FIXTURES screen ------------------------- */
function slateScreen(){
  const sec=document.getElementById("slate"); sec.innerHTML="";
  const up=D.fixtures.filter(f=>f.status==="scheduled");
  const card=el(`<div class="card"><h3>Upcoming fixtures</h3><div class="mini">Auto-predicted with current situations. Tap a match for detail on the Predict tab.</div><div id="list"></div></div>`);
  sec.appendChild(card); const list=card.querySelector("#list");
  if(!up.length){ list.innerHTML='<div class="note">No upcoming fixtures in the data.</div>'; }
  let curDate="";
  up.forEach(f=>{
    if(f.date!==curDate){ curDate=f.date; list.appendChild(el(`<h4>${f.date}</h4>`)); }
    const A=f.home,B=f.away; if(!T[A]||!T[B])return;
    const ko=f.stage==="knockout"; const host=hostOf(A,B);
    const o={knockout:ko,host:host}; if(!ko&&f.stage==="group"){o.stakesA=T[A].stakes;o.stakesB=T[B].stakes;}
    const r=predict(A,B,o);
    let summary = ko
      ? `<span class="win">${r.advA>=r.advB?A:B}</span> ${(Math.max(r.advA,r.advB)).toFixed(0)}% adv`
      : `${A} ${r.pA.toFixed(0)} / D ${r.pD.toFixed(0)} / ${B} ${r.pB.toFixed(0)}`;
    list.appendChild(el(`<div class="slate-item"><div>${A} <span class="pill">v</span> ${B}${ko?' <span class="tag ko">KO</span>':''}</div>
      <div class="pill">${summary}</div></div>`));
  });
}

/* --------------------------------- TITLE ODDS (bracket sim) ---------------- */
const advCache={};
function advProb(A,B){ const k=A+"|"+B; if(k in advCache)return advCache[k];
  const v=predict(A,B,{knockout:true}).advA/100; advCache[k]=v; advCache[B+"|"+A]=1-v; return v; }
function projectedQualifiers(){
  // top 2 of each group + 8 best thirds, by (pts, gd, gf)
  const cmp=(x,y)=> y.pts-x.pts || y.gd-x.gd || y.gf-x.gf;
  let q=[], thirds=[];
  D.groups.forEach(g=>{ const tb=[...g.table].sort(cmp); q.push(tb[0].team,tb[1].team); if(tb[2])thirds.push(tb[2]); });
  thirds.sort(cmp); q.push(...thirds.slice(0,8).map(t=>t.team));
  return q;
}
const SEED32=[1,32,16,17,8,25,9,24,4,29,13,20,5,28,12,21,2,31,15,18,7,26,10,23,3,30,14,19,6,27,11,22];
function bracketScreen(){
  const sec=document.getElementById("bracket"); sec.innerHTML="";
  const q=projectedQualifiers();
  // seed by overall rating
  const seeded=[...new Set(q)].map(t=>({t,r:T[t].att100+T[t].def100})).sort((a,b)=>b.r-a.r).map(o=>o.t).slice(0,32);
  while(seeded.length<32) seeded.push(seeded[seeded.length-1]); // safety
  const order=SEED32.map(s=>seeded[s-1]);
  const N=8000, champ={}, finalist={}; seeded.forEach(t=>{champ[t]=0;finalist[t]=0;});
  for(let s=0;s<N;s++){
    let round=order.slice();
    while(round.length>1){
      const nxt=[];
      for(let i=0;i<round.length;i+=2){ const A=round[i],B=round[i+1];
        const pa=advProb(A,B); nxt.push(Math.random()<pa?A:B); }
      if(round.length===2){ finalist[round[0]]++; finalist[round[1]]++; }
      round=nxt;
    }
    champ[round[0]]++;
  }
  const rows=seeded.map(t=>({t,c:champ[t]/N*100,f:finalist[t]/N*100})).sort((a,b)=>b.c-a.c);
  const note = P.group_complete ? "Official-stage qualifiers." : "Projected from current standings (group stage in progress)";
  const card=el(`<div class="card"><h3>Title odds</h3>
    <div class="mini">${note} · rating-seeded 32-team bracket · ${N.toLocaleString()} simulations.
    Not the official bracket draw — a projection of who wins it all.</div><div id="ch"></div></div>`);
  sec.appendChild(card); const ch=card.querySelector("#ch");
  ch.innerHTML=rows.map(r=>`<div class="champ"><span>${r.t}</span><span style="text-align:right">
    <b>${r.c.toFixed(1)}%</b> <span class="mini">cup · ${r.f.toFixed(0)}% final</span>
    <div class="bar"><i style="width:${Math.min(100,r.c*2.5)}%"></i></div></span></div>`).join("");
}

/* --------------------------------- nav + init ------------------------------ */
function show(tab){
  document.querySelectorAll("nav button").forEach(b=>b.classList.toggle("on",b.dataset.tab===tab));
  document.querySelectorAll("section").forEach(s=>s.classList.toggle("on",s.id===tab));
  if(tab==="groups"&&!document.getElementById("groups").innerHTML) groupsScreen();
  if(tab==="slate"&&!document.getElementById("slate").innerHTML) slateScreen();
  if(tab==="bracket"&&!document.getElementById("bracket").innerHTML) bracketScreen();
}
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>show(b.dataset.tab));
document.getElementById("subtitle").textContent =
  `48 teams · Dixon-Coles + xG model · data through ${P.generated}`;
document.getElementById("foot").innerHTML =
  `Model: 3-source ratings (goals+Elo+FIFA), xG-adjusted, Dixon-Coles, backtested.<br>Data through ${P.generated}. Re-run <code>build_ratings.py --refresh</code> then <code>export_web.py</code> to update.`;
predictScreen();
