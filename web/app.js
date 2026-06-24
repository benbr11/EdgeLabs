"use strict";
const D = window.WC_DATA, P = D.params, M = D.mods, T = D.teams;
const TEAMS = Object.keys(T).sort();
const FLAG_CODE={Canada:"ca",Mexico:"mx","United States":"us",Australia:"au",Iran:"ir",Iraq:"iq",Japan:"jp",Jordan:"jo",Qatar:"qa","Saudi Arabia":"sa","South Korea":"kr",Uzbekistan:"uz",Algeria:"dz","Cape Verde":"cv","DR Congo":"cd",Egypt:"eg",Ghana:"gh","Ivory Coast":"ci",Morocco:"ma",Senegal:"sn","South Africa":"za",Tunisia:"tn","Curaçao":"cw",Haiti:"ht",Panama:"pa",Argentina:"ar",Brazil:"br",Colombia:"co",Ecuador:"ec",Paraguay:"py",Uruguay:"uy","New Zealand":"nz",Austria:"at",Belgium:"be","Bosnia and Herzegovina":"ba",Croatia:"hr","Czech Republic":"cz",England:"gb-eng",France:"fr",Germany:"de",Netherlands:"nl",Norway:"no",Portugal:"pt",Scotland:"gb-sct",Spain:"es",Sweden:"se",Switzerland:"ch",Turkey:"tr"};
function flag(t){ const c=FLAG_CODE[t]; return c?`<img class="flag" src="https://flagcdn.com/w40/${c}.png" alt="" loading="lazy" onerror="this.style.display='none'">`:""; }
const TEAM_COLOR={Canada:"#D52B1E",Mexico:"#006847","United States":"#2A3C7D",Australia:"#00843D",Iran:"#239F40",Iraq:"#1A8A4A",Japan:"#BC002D",Jordan:"#007A3D",Qatar:"#8A1538","Saudi Arabia":"#1B7A3D","South Korea":"#003478",Uzbekistan:"#1EB53A",Algeria:"#1B7A3D","Cape Verde":"#0A3A8B","DR Congo":"#1077E8",Egypt:"#CE1126",Ghana:"#007B3F","Ivory Coast":"#FF7900",Morocco:"#006233",Senegal:"#00853F","South Africa":"#007749",Tunisia:"#E70013","Curaçao":"#00248F",Haiti:"#00269A",Panama:"#0049A5",Argentina:"#6CA6DC",Brazil:"#1FAA52",Colombia:"#E0B100",Ecuador:"#E8A200",Paraguay:"#C8102E",Uruguay:"#0038A8","New Zealand":"#3A3A3A",Austria:"#ED2939",Belgium:"#C9A227","Bosnia and Herzegovina":"#1B3A8B",Croatia:"#D81E2C","Czech Republic":"#11457E",England:"#CF142B",France:"#21407F",Germany:"#3A3A3A",Netherlands:"#F36C21",Norway:"#BA0C2F",Portugal:"#0A6634",Scotland:"#005EB8",Spain:"#C60B1E",Sweden:"#D9A400",Switzerland:"#D52B1E",Turkey:"#E30A17"};
function colorOf(t){ return TEAM_COLOR[t]||"#3b82f6"; }
function txtOn(hex){ const n=parseInt(hex.slice(1),16); return (0.299*(n>>16)+0.587*((n>>8)&255)+0.114*(n&255))>150?"#0e1b30":"#ffffff"; }
function colDist(a,b){ const x=parseInt(a.slice(1),16),y=parseInt(b.slice(1),16); return Math.abs((x>>16)-(y>>16))+Math.abs(((x>>8)&255)-((y>>8)&255))+Math.abs((x&255)-(y&255)); }
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
  const fixtures=D.fixtures.filter(f=>f.status==="scheduled"&&T[f.home]&&T[f.away]);  // upcoming, in order
  let idx=fixtures.length?0:-1, fxVenue=null;
  const card=el(`<div class="card">
    <div class="navrow">
      <button class="navbtn" id="prevM">◀ Prev</button>
      <div class="mlabel" id="mLabel"></div>
      <button class="navbtn" id="nextM">Next ▶</button>
    </div>
    <div class="row"><div style="flex:1"><label>Team A</label><select id="selA">${opts}</select></div>
      <div class="vs">vs</div>
      <div style="flex:1"><label>Team B</label><select id="selB">${opts}</select></div></div>
    <div class="note" id="stageNote"></div>
    <div id="out"></div>
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
      <button class="btn" id="go" style="margin-top:10px">Update with these factors</button>
    </details>
  </div>`);
  sec.appendChild(card);
  const $=(id)=>card.querySelector(id);
  $("#avA").oninput=()=>$("#avAl").textContent=$("#avA").value+"%";
  $("#avB").oninput=()=>$("#avBl").textContent=$("#avB").value+"%";
  function refreshStage(){
    const A=$("#selA").value,B=$("#selB").value;
    if(A===B){$("#stageNote").innerHTML="Pick two different teams.";return;}
    const st=stageOf(A,B);
    $("#stageNote").innerHTML = st==="group" ? `Group stage · ${A} ${stakeTag(T[A].stakes)} &nbsp; ${B} ${stakeTag(T[B].stakes)}`
      : st==="knockout" ? `<span class="tag ko">Knockout</span> extra time + penalties → who advances`
      : `Stage <b>unknown</b> (knockout bracket not set yet) — set "Knockout tie?" if this is a KO.`;
    if(fxVenue){ $("#stageNote").innerHTML += ` <span class="vtag">📍 ${fxVenue.name}${fxVenue.alt>=500?' · '+fxVenue.alt+'m':''}${fxVenue.temp>=29?' · '+fxVenue.temp+'°C':''}</span>`; }
  }
  function runPredict(){
    const A=$("#selA").value,B=$("#selB").value;
    if(A===B){$("#out").innerHTML='<div class="note">Pick two different teams.</div>';return;}
    const st=stageOf(A,B);
    const ko=$("#ko").value==="auto"?st==="knockout":$("#ko").value==="yes";
    let host=null; const vsel=$("#venue").value;
    if(vsel==="auto")host=hostOf(A,B); else if(vsel==="A")host=A; else if(vsel==="B")host=B;
    const o={knockout:ko,host:host,availA:+$("#avA").value/100,availB:+$("#avB").value/100,
      restA:+$("#rA").value,restB:+$("#rB").value,weather:$("#wx").value,
      vTemp:$("#vt").value!==""?+$("#vt").value:(fxVenue?fxVenue.temp:null), vAlt:fxVenue?fxVenue.alt:null};
    if(!ko&&st==="group"){o.stakesA=T[A].stakes;o.stakesB=T[B].stakes;}
    $("#out").innerHTML=renderResult(predict(A,B,o),ko,[+$("#oA").value,+$("#oD").value,+$("#oB").value]);
  }
  function syncNav(){
    $("#prevM").disabled = idx<=0;
    $("#nextM").disabled = idx<0 || idx>=fixtures.length-1;
    $("#mLabel").textContent = idx>=0 ? `${fixtures[idx].date} · match ${idx+1} of ${fixtures.length}` : "Custom matchup";
  }
  function resetFactors(){
    $("#ko").value="auto"; $("#venue").value="auto"; $("#wx").value="clear";
    $("#avA").value=100; $("#avB").value=100; $("#avAl").textContent="100%"; $("#avBl").textContent="100%";
    $("#rA").value=4; $("#rB").value=4; $("#vt").value=""; $("#oA").value=""; $("#oD").value=""; $("#oB").value="";
  }
  function loadFixture(i){
    if(i<0||i>=fixtures.length) return;
    idx=i; resetFactors();
    fxVenue = fixtures[i].valt!=null ? {alt:fixtures[i].valt,temp:fixtures[i].vtemp,name:fixtures[i].venue} : null;
    $("#selA").value=fixtures[i].home; $("#selB").value=fixtures[i].away;
    syncNav(); refreshStage(); runPredict();
  }
  $("#prevM").onclick=()=>loadFixture(idx-1);
  $("#nextM").onclick=()=>loadFixture(idx+1);
  $("#selA").onchange=()=>{idx=-1;fxVenue=null;syncNav();refreshStage();runPredict();};
  $("#selB").onchange=()=>{idx=-1;fxVenue=null;syncNav();refreshStage();runPredict();};
  $("#go").onclick=runPredict;
  window.predictMatchup=(A,B,ven)=>{ resetFactors(); fxVenue=ven||null; $("#selA").value=A; $("#selB").value=B; idx=-1; syncNav(); refreshStage(); runPredict(); };
  if(idx>=0){ loadFixture(0); }                       // auto-open the next fixture
  else { $("#prevM").style.display="none"; $("#nextM").style.display="none"; $("#mLabel").textContent="Pick any two teams";
         $("#selA").value="Argentina"; $("#selB").value="Brazil"; refreshStage(); runPredict(); }
}
function renderResult(r,ko,odds){
  const {A,B}=r; let cA=colorOf(A),cB=colorOf(B); if(colDist(cA,cB)<90)cB="#f59e0b"; let h="";
  if(ko){
    h+=`<div class="probbar"><div class="seg" style="width:${Math.max(2,r.advA)}%;background:${cA};color:${txtOn(cA)}">${r.advA.toFixed(0)}%</div>
        <div class="seg" style="width:${Math.max(2,r.advB)}%;background:${cB};color:${txtOn(cB)}">${r.advB.toFixed(0)}%</div></div>
        <div class="big"><div><div class="n">${r.advA.toFixed(1)}%</div><div class="l">${flag(A)} ${A}</div></div>
        <div><div class="n">${r.advB.toFixed(1)}%</div><div class="l">${flag(B)} ${B}</div></div></div>
        <div class="xg">90 min: ${A} ${r.pA.toFixed(0)}% / draw ${r.pD.toFixed(0)}% / ${B} ${r.pB.toFixed(0)}% · extra time ${r.pET.toFixed(0)}% · penalties ${r.pPen.toFixed(1)}%</div>`;
  } else {
    h+=`<div class="probbar"><div class="seg" style="width:${Math.max(3,r.pA)}%;background:${cA};color:${txtOn(cA)}">${r.pA.toFixed(0)}%</div>
        <div class="seg sD" style="width:${Math.max(3,r.pD)}%">${r.pD.toFixed(0)}%</div>
        <div class="seg" style="width:${Math.max(3,r.pB)}%;background:${cB};color:${txtOn(cB)}">${r.pB.toFixed(0)}%</div></div>
        <div class="big"><div><div class="n">${r.pA.toFixed(1)}%</div><div class="l">${flag(A)} ${A}</div></div>
        <div><div class="n">${r.pD.toFixed(1)}%</div><div class="l">draw</div></div>
        <div><div class="n">${r.pB.toFixed(1)}%</div><div class="l">${flag(B)} ${B}</div></div></div>`;
  }
  h+=`<div class="pills"><span class="pillstat">Exp. goals <b>${r.exA.toFixed(2)}</b> – <b>${r.exB.toFixed(2)}</b></span>`+
     `<span class="pillstat">Both score <b>${r.btts.toFixed(0)}%</b></span>`+
     `<span class="pillstat">Clean sheet <b>${r.csA.toFixed(0)}%</b> / <b>${r.csB.toFixed(0)}%</b></span></div>`;
  h+=`<h4>Most likely scorelines</h4>`;
  const tmax=Math.max.apply(null,r.top.map(s=>s.p))||1;
  h+=r.top.map(s=>`<div class="scoreline"><span class="sl-teams">${flag(A)} <b>${s.i}</b>–<b>${s.j}</b> ${flag(B)}</span><span class="sl-bar"><i style="width:${(s.p/tmax*100).toFixed(0)}%"></i></span><span class="p">${s.p.toFixed(1)}%</span></div>`).join("");
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
    const rows=g.table.map((r,i)=>{const s=T[r.team].stakes; const a=adv[r.team]??0;
      return `<tr class="${i<2?'adv-pos':''}"><td>${flag(r.team)} ${r.team} ${r.P>=2?stakeTag(s):''}</td><td>${r.P}</td><td><b>${r.pts}</b></td><td>${r.gf}-${r.ga}</td><td>${r.gd>=0?'+':''}${r.gd}</td><td class="adv">${a.toFixed(0)}%</td></tr>`;}).join("");
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
      return `<div class="ko-tie"><span>${flag(f.home)} ${f.home} v ${flag(f.away)} ${f.away}</span><span><span class="win">${w}</span> <span class="pill">${wp.toFixed(0)}%</span></span></div>`;
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
    const ko=f.stage==="knockout", host=hostOf(A,B), o={knockout:ko,host:host,vTemp:f.vtemp??null,vAlt:f.valt??null};
    if(!ko&&f.stage==="group"){o.stakesA=T[A].stakes;o.stakesB=T[B].stakes;}
    const r=predict(A,B,o);
    const summary = ko ? `<span class="win">${r.advA>=r.advB?A:B}</span> ${Math.max(r.advA,r.advB).toFixed(0)}% adv`
      : `${A} ${r.pA.toFixed(0)} / D ${r.pD.toFixed(0)} / ${B} ${r.pB.toFixed(0)}`;
    const item=el(`<div class="slate-item" style="cursor:pointer"><div>${flag(A)} ${A} <span class="pill">v</span> ${flag(B)} ${B}${ko?' <span class="tag ko">KO</span>':''}</div><div class="pill">${summary} ›</div></div>`);
    item.onclick=()=>{ if(window.predictMatchup){ window.predictMatchup(A,B, f.valt!=null?{alt:f.valt,temp:f.vtemp,name:f.venue}:null); document.querySelector('nav button[data-tab="predict"]').click(); window.scrollTo(0,0); } };
    list.appendChild(item);
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
  const medal=["🥇","🥈","🥉"];
  card.querySelector("#ch").innerHTML=rows.map((r,i)=>`<div class="champ"><span>${medal[i]||""} ${flag(r.t)} ${r.t}</span><span style="text-align:right">
    <b>${r.c.toFixed(1)}%</b> <span class="mini">cup · ${r.f.toFixed(0)}% final</span>
    <div class="bar"><i style="width:${Math.min(100,r.c*2.5)}%"></i></div></span></div>`).join("");
}

/* --------------------------------- PLAYER ODDS ----------------------------- */
function playerOdds(team, teamLambda){
  const base = P.avg * T[team].att_mult * (P.avgdfn||1); // team's exp. goals vs an AVERAGE defence (incl. mean dfn mult)
  const scale = base > 0 ? teamLambda/base : 1;        // scale to this matchup (opponent defence); ~1 for an average opponent
  const af = {FWD:0.55, MID:1.0, DEF:0.7};             // assist-to-goal ratio by role (estimate)
  const CAL_FLOOR = 0.045, CAL_GAMMA = 0.55;           // backtest calibration (backtest_players.py): baseline hazard + discount; beats baseline log-loss
  return (D.players[team]||[]).map(p=>{
    // open-play + direct free kicks scale with the matchup defence; penalties (p.peng,
    // each player's real recency-weighted penalty-goal rate) do NOT — a pen is a pen.
    let gx = (p.op||0)*scale + (p.fk||0)*scale + (p.peng||0);
    const ax = (p.ar!=null ? p.ar : (p.op||0)*(af[p.pos]||0.6)) * scale;  // REAL assist rate (StatsBomb) where available, else role estimate
    const gh = CAL_FLOOR + CAL_GAMMA*gx, ah = CAL_GAMMA*ax;   // calibrated hazards (cures raw over-confidence; floor only on goals)
    return {...p, pg:(1-Math.exp(-gh))*100, pa:(1-Math.exp(-ah))*100, pga:(1-Math.exp(-(gh+ah)))*100};
  }).filter(x=>x.pga>=8).sort((a,b)=>b.pga-a.pga).slice(0,6);
}
function playersScreen(){
  const sec=document.getElementById("players"); sec.innerHTML="";
  sec.appendChild(el(`<div class="card"><h3>Player odds — ${P.stage_label}</h3>
    <div class="mini">Anytime <b>goal</b> / <b>assist</b> / <b>goal-or-assist</b> chances for the main contributors. Built from each player's <b>recency-weighted real international scoring rate</b> (every team, via 47k+ logged goals) refined by <b>StatsBomb shot quality</b> where available, scaled to the matchup defence, and calibrated against history. A <span class="ppos low">low data</span> tag means we have little real data on that player — treat any gap vs the bookmaker as our blind spot, not an edge. Assists now use <b>real StatsBomb assist + expected-assist data</b> where we have it (a role-based estimate otherwise).</div></div>`));
  const up=D.fixtures.filter(f=>f.status==="scheduled"&&T[f.home]&&T[f.away]);
  if(!up.length){ sec.appendChild(el(`<div class="card"><div class="note">No upcoming fixtures right now.</div></div>`)); return; }
  const rows=(team,lam)=>{ const ps=playerOdds(team,lam);
    return ps.length ? ps.map(p=>{ const low=(p.conf!=null&&p.conf<0.35);
      return `<div class="prow${low?' lowconf':''}"><span class="pn">${flag(team)} ${p.n} <span class="ppos">${p.pos}</span>${p.pen?' <span class="ppos pk">PK</span>':''}${low?' <span class="ppos low" title="Limited real data on this player — low confidence">low data</span>':''}</span><span class="pg">${p.pg.toFixed(0)}%</span><span class="pg">${p.pa.toFixed(0)}%</span><span class="pg ga">${p.pga.toFixed(0)}%</span></div>`; }).join("")
      : `<div class="mini" style="padding:6px 2px">${flag(team)} ${team} — no standout scorer.</div>`; };
  up.forEach(f=>{
    const A=f.home,B=f.away, ko=f.stage==="knockout", host=hostOf(A,B);
    const o={knockout:ko,host:host,vTemp:f.vtemp??null,vAlt:f.valt??null};
    if(!ko&&f.stage==="group"){o.stakesA=T[A].stakes;o.stakesB=T[B].stakes;}
    const r=predict(A,B,o);
    sec.appendChild(el(`<div class="card"><h4 style="margin:0 0 8px">${flag(A)} ${A} <span class="pill">v</span> ${flag(B)} ${B} <span class="mini">· ${f.date}</span></h4>
      <div class="prow phead"><span class="pn">Player</span><span class="pg">Goal</span><span class="pg">Assist</span><span class="pg ga">G+A</span></div>
      ${rows(A,r.exA)}${rows(B,r.exB)}</div>`));
  });
}

/* --------------------------------- theme + nav + init ---------------------- */
function setTheme(light){ document.body.classList.toggle("light",light);
  document.getElementById("themeBtn").textContent=light?"☀️":"🌙";
  try{ localStorage.setItem("wc-theme",light?"light":"dark"); }catch(e){} }
document.getElementById("themeBtn").onclick=()=>setTheme(!document.body.classList.contains("light"));
try{ setTheme(localStorage.getItem("wc-theme")==="light"); }catch(e){}

const infoModal=document.getElementById("infoModal");
document.getElementById("infoBtn").onclick=()=>infoModal.classList.add("show");
infoModal.onclick=(e)=>{ if(e.target===infoModal||e.target.classList.contains("modal-close")) infoModal.classList.remove("show"); };
document.addEventListener("keydown",(e)=>{ if(e.key==="Escape") infoModal.classList.remove("show"); });
{ const sd=document.getElementById("srcDate"); if(sd) sd.textContent=P.generated; }

const groupsComplete=P.group_complete;
document.getElementById("tab2").textContent = groupsComplete ? "Bracket" : "Groups";
function show(tab){
  document.querySelectorAll("nav button").forEach(b=>b.classList.toggle("on",b.dataset.tab===tab));
  document.querySelectorAll("section").forEach(s=>s.classList.toggle("on",s.id===tab));
  if(tab==="groups"&&!document.getElementById("groups").innerHTML){ groupsComplete?koBracketScreen():groupsScreen(); }
  if(tab==="slate"&&!document.getElementById("slate").innerHTML) slateScreen();
  if(tab==="bracket"&&!document.getElementById("bracket").innerHTML) titleOddsScreen();
  if(tab==="players"&&!document.getElementById("players").innerHTML) playersScreen();
}
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>show(b.dataset.tab));
document.getElementById("subtitle").textContent=`48 teams · Dixon-Coles + xG model · data through ${P.generated}`;
document.getElementById("foot").innerHTML=`Model: 3-source ratings (goals+Elo+FIFA), xG-adjusted, Dixon-Coles, backtested.<br>Data through ${P.generated}. Updates automatically twice daily.`;
predictScreen();
