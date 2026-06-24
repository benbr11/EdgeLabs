"use strict";
const D = window.WC_DATA, P = D.params, M = D.mods, T = D.teams;
const TEAMS = Object.keys(T).sort();
const el = (h) => { const d = document.createElement("div"); d.innerHTML = h.trim(); return d.firstChild; };

/* ----------------------------- prediction math (mirrors simulate.py) ------- */
function poisson(lam, mg){ const o=[]; let p=Math.exp(-lam); for(let k=0;k<=mg;k++){o.push(p); p*=lam/(k+1);} return o; }
function rpois(l){ const L=Math.exp(-l); let k=0,p=1; do{k++;p*=Math.random();}while(p>L); return k-1; }
const VAR_BASE=6.0, VAR_SLOPE=0.34;   // variance-by-rating: weaker teams = more volatile
function lgamma(x){
  const c=[0.99999999999980993,676.5203681218851,-1259.1392167224028,771.32342877765313,
           -176.61502916214059,12.507343278686905,-0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
  if(x<0.5) return Math.log(Math.PI/Math.sin(Math.PI*x))-lgamma(1-x);
  x-=1; let a=c[0]; const t=x+7.5;
  for(let i=1;i<9;i++) a+=c[i]/(x+i);
  return 0.5*Math.log(2*Math.PI)+(x+0.5)*Math.log(t)-t+Math.log(a);
}
function nbpmf(mu, r, mg){ const o=[];   // negative-binomial; large r -> Poisson
  for(let k=0;k<=mg;k++) o.push(Math.exp(lgamma(k+r)-lgamma(r)-lgamma(k+1)+r*Math.log(r/(r+mu))+k*Math.log(mu/(r+mu))));
  return o;
}
function dcMatrix(lh, la, rh, ra){
  rh = rh || 50; ra = ra || 50;
  const mg = Math.max(12, Math.floor(lh+la)+8);
  const ph = nbpmf(lh,rh,mg), pa = nbpmf(la,ra,mg), Mx = [];
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
function lambdas(A,B,o){
  o=o||{}; const host=o.host||null;
  const vTemp=o.vTemp!=null?o.vTemp:(host?T[host].home_temp:null);
  const vAlt =o.vAlt!=null ?o.vAlt :(host?T[host].home_alt :null);
  const mA=mods(A, o.availA??1, o.restA??4, o.stakesA||"normal", vTemp, vAlt);
  const mB=mods(B, o.availB??1, o.restB??4, o.stakesB||"normal", vTemp, vAlt);
  const w=M.weather[o.weather||"clear"];
  const hfA=host===A?P.home_adv:1, hfB=host===B?P.home_adv:1;
  return [P.avg*(T[A].att_mult*mA.am)*(T[B].dfn_mult/mB.dm)*hfA*w,
          P.avg*(T[B].att_mult*mB.am)*(T[A].dfn_mult/mA.dm)*hfB*w];
}
function predict(A,B,o){
  o=o||{};
  const [lamA,lamB]=lambdas(A,B,o);
  const dA=VAR_BASE+VAR_SLOPE*((T[A].att100+T[A].def100)/2), dB=VAR_BASE+VAR_SLOPE*((T[B].att100+T[B].def100)/2);
  const {Mx,mg}=dcMatrix(lamA,lamB,dA,dB), rng=[...Array(mg+1).keys()];
  let pA=0,pD=0,pB=0,exA=0,exB=0,pH0=0,pA0=0; const flat=[];
  for(let i=0;i<=mg;i++)for(let j=0;j<=mg;j++){ const p=Mx[i][j];
    if(i>j)pA+=p; else if(j>i)pB+=p; else pD+=p; exA+=i*p; exB+=j*p;
    if(i===0)pH0+=p; if(j===0)pA0+=p; flat.push([p,i,j]); }
  flat.sort((x,y)=>y[0]-x[0]);
  const btts=Math.max(0,1-pH0-pA0+Mx[0][0]);
  const r={A,B,lamA,lamB,pA:pA*100,pD:pD*100,pB:pB*100,exA,exB,
           btts:btts*100, csA:pA0*100, csB:pH0*100,
           top:flat.slice(0,3).map(([p,i,j])=>({i,j,p:p*100}))};
  if(o.knockout){
    const et=dcMatrix(lamA/3,lamB/3,dA,dB); let qa=0,qb=0,qd=0;
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

/* group advance odds: Monte-Carlo the remaining group games -> % to reach knockouts
   (top 2 of each group + the 8 best third-placed teams). */
function groupAdvanceOdds(){
  if(window._advCache) return window._advCache;
  const N=5000, rem=[];
  D.groups.forEach((g,gi)=>g.remaining.forEach(([h,a])=>{
    const [lh,la]=lambdas(h,a,{host:hostOf(h,a)}); rem.push({gi,h,la,lh,a}); }));
  const base=D.groups.map(g=>g.table.map(r=>({team:r.team,pts:r.pts,gf:r.gf,ga:r.ga})));
  const adv={}; D.groups.forEach(g=>g.table.forEach(r=>adv[r.team]=0));
  const cmp=(x,y)=> y.pts-x.pts || (y.gf-y.ga)-(x.gf-x.ga) || y.gf-x.gf;
  for(let s=0;s<N;s++){
    const st=base.map(grp=>grp.map(r=>({team:r.team,pts:r.pts,gf:r.gf,ga:r.ga})));
    const idx={}; st.forEach(grp=>grp.forEach(r=>idx[r.team]=r));
    rem.forEach(m=>{ const hh=rpois(m.lh), aa=rpois(m.la), H=idx[m.h], A=idx[m.a];
      H.gf+=hh;H.ga+=aa;A.gf+=aa;A.ga+=hh;
      if(hh>aa)H.pts+=3; else if(aa>hh)A.pts+=3; else {H.pts++;A.pts++;} });
    const thirds=[];
    st.forEach(grp=>{ const o=[...grp].sort(cmp); adv[o[0].team]++; adv[o[1].team]++; if(o[2])thirds.push(o[2]); });
    thirds.sort(cmp); thirds.slice(0,8).forEach(t=>adv[t.team]++);
  }
  const out={}; for(const t in adv) out[t]=adv[t]/N*100;
  window._advCache=out; return out;
}

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
  $("#selA").selectedIndex=Math.max(0,TEAMS.indexOf("Argentina")); $("#selB").selectedIndex=Math.max(0,TEAMS.indexOf("Brazil"));
  $("#avA").oninput=()=>$("#avAl").textContent=$("#avA").value+"%";
  $("#avB").oninput=()=>$("#avBl").textContent=$("#avB").value+"%";
  const refreshStage=()=>{
    const A=$("#selA").value,B=$("#selB").value;
    if(A===B){$("#stageNote").innerHTML="Pick two different teams.";return;}
    const st=stageOf(A,B);
    $("#stageNote").innerHTML = st==="group" ? `Group stage · ${A} ${stakeTag(T[A].stakes)} &nbsp; ${B} ${stakeTag(T[B].stakes)}`
      : st==="knockout" ? `<span class="tag ko">Knockout</span> extra time + penalties → who advances`
      : `Stage <b>unknown</b> (knockout bracket not set yet) — set "Knockout tie?" if this is a KO.`;
  };
  $("#selA").onchange=refreshStage; $("#selB").onchange=refreshStage; refreshStage();
  $("#go").onclick=()=>{
    const A=$("#selA").value,B=$("#selB").value; if(A===B){$("#out").innerHTML='<div class="note">Pick two different teams.</div>';return;}
    const st=stageOf(A,B);
    const ko = $("#ko").value==="auto" ? st==="knockout" : $("#ko").value==="yes";
    let host=null; const vsel=$("#venue").value;
    if(vsel==="auto") host=hostOf(A,B); else if(vsel==="A")host=A; else if(vsel==="B")host=B;
    const o={knockout:ko, host:host, availA:+$("#avA").value/100, availB:+$("#avB").value/100,
      restA:+$("#rA").value, restB:+$("#rB").value, weather:$("#wx").value, vTemp:$("#vt").value!==""?+$("#vt").value:null};
    if(!ko && st==="group"){ o.stakesA=T[A].stakes; o.stakesB=T[B].stakes; }
    $("#out").innerHTML=renderResult(predict(A,B,o), ko, [+$("#oA").value,+$("#oD").value,+$("#oB").value]);
  };
}
function renderResult(r,ko,odds){
  const {A,B}=r; let h="";
  if(ko){
    h+=`<div class="probbar"><div class="sA" style="width:${Math.max(2,r.advA)}%">${r.advA.toFixed(0)}%</div>
        <div class="sB" style="width:${Math.max(2,r.advB)}%">${r.advB.toFixed(0)}%</div></div>
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
  h+=`<div class="statline"><span>Both teams to score: <b>${r.btts.toFixed(0)}%</b></span>
      <span>Clean sheet — ${A} <b>${r.csA.toFixed(0)}%</b></span><span>${B} <b>${r.csB.toFixed(0)}%</b></span></div>`;
  h+=`<h4>Most likely scorelines</h4>`;
  h+=r.top.map(s=>`<div class="scoreline"><span>${A} ${s.i} – ${s.j} ${B}</span><span class="p">${s.p.toFixed(1)}%</span></div>`).join("");
  if(odds[0]>1&&odds[1]>1&&odds[2]>1){
    const raw=[1/odds[0],1/odds[1],1/odds[2]], ov=raw[0]+raw[1]+raw[2], mp=[r.pA/100,r.pD/100,r.pB/100], lab=[A,"Draw",B];
    h+=`<h4>Edge vs market (overround ${((ov-1)*100).toFixed(1)}%)</h4><table><tr><th>Outcome</th><th>Model</th><th>Mkt</th><th>Odds</th><th>EV</th></tr>`;
    for(let i=0;i<3;i++){ const ev=mp[i]*odds[i]-1;
      h+=`<tr><td>${lab[i]}</td><td>${(mp[i]*100).toFixed(1)}%</td><td>${(raw[i]/ov*100).toFixed(1)}%</td><td>${odds[i].toFixed(2)}</td>
          <td class="${ev>0?'edge':'neg'}">${ev>0?'+':''}${(ev*100).toFixed(1)}%${ev>0?' ✓':''}</td></tr>`; }
    h+=`</table>`;
  }
  return h;
}

/* --------------------------------- GROUPS screen (with advance %) ---------- */
function groupsScreen(){
  const sec=document.getElementById("groups"); sec.innerHTML="";
  sec.appendChild(el(`<div class="card"><h3>Group standings</h3><div class="mini">“Adv%” = chance to reach the knockouts (top 2 of the group, or one of the 8 best third-placed teams), from simulating every remaining group game.</div></div>`));
  const adv=groupAdvanceOdds();
  D.groups.forEach(g=>{
    const rows=g.table.map(r=>{const s=T[r.team].stakes; const a=adv[r.team]??0;
      return `<tr><td>${r.team} ${r.P>=2?stakeTag(s):''}</td><td>${r.P}</td><td><b>${r.pts}</b></td><td>${r.gf}-${r.ga}</td><td>${r.gd>=0?'+':''}${r.gd}</td><td class="adv">${a.toFixed(0)}%</td></tr>`;}).join("");
    const rem=g.remaining.length?`<div class="mini" style="margin-top:8px">Remaining: ${g.remaining.map(m=>m[0]+" v "+m[1]).join(" · ")}</div>`:`<div class="mini" style="margin-top:8px">Group complete</div>`;
    sec.appendChild(el(`<div class="card"><h3>${g.name}</h3>
      <table><tr><th>Team</th><th>P</th><th>Pts</th><th>GF-GA</th><th>GD</th><th>Adv%</th></tr>${rows}</table>${rem}</div>`));
  });
}

/* --------------------------- KNOCKOUT BRACKET (replaces Groups once over) --- */
const ROUND_ORDER=["Round of 32","Round of 16","Quarter-finals","Quarter-final","Semi-finals","Semi-final","Final","Play-off for third place","Third place play-off"];
function koBracketScreen(){
  const sec=document.getElementById("groups"); sec.innerHTML="";
  const kos=D.fixtures.filter(f=>f.stage==="knockout");
  if(!kos.length){ sec.appendChild(el(`<div class="card"><h3>Knockout bracket</h3><div class="note">The bracket will appear here automatically once the group stage ends and the draw is set.</div></div>`)); return; }
  const byRound={}; kos.forEach(f=>{const r=f.round||"Knockouts"; (byRound[r]=byRound[r]||[]).push(f);});
  const rounds=Object.keys(byRound).sort((a,b)=>{const ia=ROUND_ORDER.indexOf(a),ib=ROUND_ORDER.indexOf(b);return (ia<0?99:ia)-(ib<0?99:ib);});
  sec.appendChild(el(`<div class="card"><h3>Knockout bracket</h3><div class="mini">Each tie shows the model’s % to advance (extra time + penalties included).</div></div>`));
  rounds.forEach(rd=>{
    const ties=byRound[rd].map(f=>{
      if(!T[f.home]||!T[f.away]) return `<div class="ko-tie"><span>${f.home} v ${f.away}</span><span class="pill">—</span></div>`;
      const r=predict(f.home,f.away,{knockout:true,host:hostOf(f.home,f.away)});
      const w=r.advA>=r.advB?f.home:f.away, wp=Math.max(r.advA,r.advB);
      return `<div class="ko-tie"><span>${f.home} v ${f.away}</span><span><span class="win">${w}</span> <span class="pill">${wp.toFixed(0)}%</span></span></div>`;
    }).join("");
    sec.appendChild(el(`<div class="card"><h3>${rd}</h3>${ties}</div>`));
  });
}

/* --------------------------------- FIXTURES screen ------------------------- */
function slateScreen(){
  const sec=document.getElementById("slate"); sec.innerHTML="";
  const up=D.fixtures.filter(f=>f.status==="scheduled");
  const card=el(`<div class="card"><h3>Upcoming fixtures</h3><div class="mini">Auto-predicted with current situations.</div><div id="list"></div></div>`);
  sec.appendChild(card); const list=card.querySelector("#list");
  if(!up.length){ list.innerHTML='<div class="note">No upcoming fixtures in the data.</div>'; return; }
  let curDate="";
  up.forEach(f=>{
    if(f.date!==curDate){ curDate=f.date; list.appendChild(el(`<h4>${f.date}</h4>`)); }
    const A=f.home,B=f.away; if(!T[A]||!T[B])return;
    const ko=f.stage==="knockout", host=hostOf(A,B), o={knockout:ko,host:host};
    if(!ko&&f.stage==="group"){o.stakesA=T[A].stakes;o.stakesB=T[B].stakes;}
    const r=predict(A,B,o);
    const summary = ko ? `<span class="win">${r.advA>=r.advB?A:B}</span> ${Math.max(r.advA,r.advB).toFixed(0)}% adv`
      : `${A} ${r.pA.toFixed(0)} / D ${r.pD.toFixed(0)} / ${B} ${r.pB.toFixed(0)}`;
    list.appendChild(el(`<div class="slate-item"><div>${A} <span class="pill">v</span> ${B}${ko?' <span class="tag ko">KO</span>':''}</div><div class="pill">${summary}</div></div>`));
  });
}

/* --------------------------------- TITLE ODDS (bracket sim) ---------------- */
const advCache={};
function advProb(A,B){ const k=A+"|"+B; if(k in advCache)return advCache[k];
  const v=predict(A,B,{knockout:true}).advA/100; advCache[k]=v; advCache[B+"|"+A]=1-v; return v; }
function projectedQualifiers(){
  const cmp=(x,y)=> y.pts-x.pts || y.gd-x.gd || y.gf-x.gf; let q=[],thirds=[];
  D.groups.forEach(g=>{ const tb=[...g.table].sort(cmp); q.push(tb[0].team,tb[1].team); if(tb[2])thirds.push(tb[2]); });
  thirds.sort(cmp); q.push(...thirds.slice(0,8).map(t=>t.team)); return q;
}
const SEED32=[1,32,16,17,8,25,9,24,4,29,13,20,5,28,12,21,2,31,15,18,7,26,10,23,3,30,14,19,6,27,11,22];
function titleOddsScreen(){
  const sec=document.getElementById("bracket"); sec.innerHTML="";
  const q=projectedQualifiers();
  const seeded=[...new Set(q)].map(t=>({t,r:T[t].att100+T[t].def100})).sort((a,b)=>b.r-a.r).map(o=>o.t).slice(0,32);
  while(seeded.length<32) seeded.push(seeded[seeded.length-1]);
  const order=SEED32.map(s=>seeded[s-1]);
  const N=8000, champ={}, finalist={}; seeded.forEach(t=>{champ[t]=0;finalist[t]=0;});
  for(let s=0;s<N;s++){ let round=order.slice();
    while(round.length>1){ const nxt=[];
      for(let i=0;i<round.length;i+=2){ nxt.push(Math.random()<advProb(round[i],round[i+1])?round[i]:round[i+1]); }
      if(round.length===2){ finalist[round[0]]++; finalist[round[1]]++; } round=nxt; }
    champ[round[0]]++; }
  const rows=seeded.map(t=>({t,c:champ[t]/N*100,f:finalist[t]/N*100})).sort((a,b)=>b.c-a.c);
  const note = P.group_complete ? "Qualified teams." : "Projected from current standings (group stage in progress)";
  const card=el(`<div class="card"><h3>Title odds</h3><div class="mini">${note} · rating-seeded 32-team bracket · ${N.toLocaleString()} simulations. A projection of who wins it all (not the official draw).</div><div id="ch"></div></div>`);
  sec.appendChild(card);
  card.querySelector("#ch").innerHTML=rows.map(r=>`<div class="champ"><span>${r.t}</span><span style="text-align:right">
    <b>${r.c.toFixed(1)}%</b> <span class="mini">cup · ${r.f.toFixed(0)}% final</span>
    <div class="bar"><i style="width:${Math.min(100,r.c*2.5)}%"></i></div></span></div>`).join("");
}

/* --------------------------------- theme + nav + init ---------------------- */
function setTheme(light){ document.body.classList.toggle("light",light);
  document.getElementById("themeBtn").textContent=light?"☀️":"🌙";
  try{ localStorage.setItem("wc-theme",light?"light":"dark"); }catch(e){} }
document.getElementById("themeBtn").onclick=()=>setTheme(!document.body.classList.contains("light"));
try{ setTheme(localStorage.getItem("wc-theme")==="light"); }catch(e){}

const groupsComplete=P.group_complete;
document.getElementById("tab2").textContent = groupsComplete ? "Bracket" : "Groups";
function show(tab){
  document.querySelectorAll("nav button").forEach(b=>b.classList.toggle("on",b.dataset.tab===tab));
  document.querySelectorAll("section").forEach(s=>s.classList.toggle("on",s.id===tab));
  if(tab==="groups"&&!document.getElementById("groups").innerHTML){ groupsComplete?koBracketScreen():groupsScreen(); }
  if(tab==="slate"&&!document.getElementById("slate").innerHTML) slateScreen();
  if(tab==="bracket"&&!document.getElementById("bracket").innerHTML) titleOddsScreen();
}
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>show(b.dataset.tab));
document.getElementById("subtitle").textContent=`48 teams · Dixon-Coles + xG model · data through ${P.generated}`;
document.getElementById("foot").innerHTML=`Model: 3-source ratings (goals+Elo+FIFA), xG-adjusted, Dixon-Coles, backtested.<br>Data through ${P.generated}. Updates automatically twice daily.`;
predictScreen();
