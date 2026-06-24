/* Club-league predictor (EPL, La Liga, Serie A, Bundesliga, Ligue 1).
   One generic screen, switched per league. Poisson + Dixon-Coles, same as the soccer model. */
(function(){
  if(!window.LEAGUES_DATA){ return; }
  const LD=window.LEAGUES_DATA.leagues, GEN=window.LEAGUES_DATA.generated;
  const $=id=>document.getElementById(id);
  let cur=null;

  function predict(L,home,away){
    const T=L.teams,P=L.params,rho=P.rho;
    let lh=P.avg*T[home].att*T[away].dfn*P.home_adv, la=P.avg*T[away].att*T[home].dfn;
    const fac=k=>{let f=1;for(let i=2;i<=k;i++)f*=i;return f;};
    const Po=(k,l)=>Math.exp(-l)*Math.pow(l,k)/fac(k);
    let pH=0,pD=0,pA=0,pH0=0,pA0=0,btts=0,cells=[];
    for(let i=0;i<10;i++)for(let j=0;j<10;j++){
      const tau=(i===0&&j===0)?1-lh*la*rho:(i===0&&j===1)?1+lh*rho:(i===1&&j===0)?1+la*rho:(i===1&&j===1)?1-rho:1;
      const m=Po(i,lh)*Po(j,la)*tau;
      if(i>j)pH+=m; else if(i===j)pD+=m; else pA+=m;
      if(j===0)pA0+=m; if(i===0)pH0+=m; if(i>=1&&j>=1)btts+=m;
      cells.push([m,i,j]);
    }
    const s=pH+pD+pA||1; cells.sort((x,y)=>y[0]-x[0]);
    return {lh,la,pH:pH/s*100,pD:pD/s*100,pA:pA/s*100,btts:btts/s*100,csA:pA0/s*100,csB:pH0/s*100,
            top:cells.slice(0,3).map(c=>({p:c[0]/s*100,i:c[1],j:c[2]}))};
  }

  function renderResult(L,home,away){
    const r=predict(L,home,away), cA="var(--accent)", cB="#f59e0b", tmax=Math.max.apply(null,r.top.map(s=>s.p))||1;
    return `<div class="probbar">
        <div class="seg" style="width:${Math.max(3,r.pH)}%;background:${cA};color:#fff">${r.pH.toFixed(0)}%</div>
        <div class="seg sD" style="width:${Math.max(3,r.pD)}%">${r.pD.toFixed(0)}%</div>
        <div class="seg" style="width:${Math.max(3,r.pA)}%;background:${cB};color:#0e1b30">${r.pA.toFixed(0)}%</div></div>
      <div class="big"><div><div class="n">${r.pH.toFixed(1)}%</div><div class="l">${home}</div></div>
        <div><div class="n">${r.pD.toFixed(1)}%</div><div class="l">draw</div></div>
        <div><div class="n">${r.pA.toFixed(1)}%</div><div class="l">${away}</div></div></div>
      <div class="pills">
        <span class="pillstat">Exp. goals <b>${r.lh.toFixed(2)}</b> – <b>${r.la.toFixed(2)}</b></span>
        <span class="pillstat">Both score <b>${r.btts.toFixed(0)}%</b></span>
        <span class="pillstat">Clean sheet <b>${r.csA.toFixed(0)}%</b> / <b>${r.csB.toFixed(0)}%</b></span></div>
      <h4>Most likely scorelines</h4>
      ${r.top.map(s=>`<div class="scoreline"><span class="sl-teams">${home} <b>${s.i}</b>–<b>${s.j}</b> ${away}</span><span class="sl-bar"><i style="width:${(s.p/tmax*100).toFixed(0)}%"></i></span><span class="p">${s.p.toFixed(1)}%</span></div>`).join('')}`;
  }
  function run(){ const L=LD[cur]; const h=$('lgA').value, a=$('lgB').value;
    $('lgResult').innerHTML = (h===a)?`<div class="note" style="padding:12px">Pick two different teams.</div>`:renderResult(L,h,a); }

  function buildPredict(){
    const L=LD[cur], order=Object.keys(L.teams).sort((a,b)=>(L.teams[b].att100+L.teams[b].def100)-(L.teams[a].att100+L.teams[a].def100));
    const opt=sel=>order.map(t=>`<option${t===sel?' selected':''}>${t}</option>`).join('');
    $('lgPredict').innerHTML=`<div class="card">
      <div class="mini" style="margin-bottom:10px">Pick any two teams — home side on the left. Poisson + Dixon-Coles, recency-weighted; home advantage included.</div>
      <div class="nhlform">
        <div><label>Home</label><select id="lgA">${opt(order[0])}</select></div>
        <div><label>Away</label><select id="lgB">${opt(order[1])}</select></div>
      </div><div id="lgResult"></div></div>`;
    $('lgA').onchange=run; $('lgB').onchange=run; run();
  }
  function buildRatings(){
    const L=LD[cur], order=Object.keys(L.teams).sort((a,b)=>(L.teams[b].att100+L.teams[b].def100)-(L.teams[a].att100+L.teams[a].def100));
    let rows=order.map((t,i)=>`<tr><td>${i+1}</td><td>${t}</td><td><b>${L.teams[t].att100.toFixed(0)}</b></td><td><b>${L.teams[t].def100.toFixed(0)}</b></td><td>${L.teams[t].elo}</td></tr>`).join('');
    $('lgRatings').innerHTML=`<div class="card"><h3>${L.name} — power ratings</h3>
      <div class="mini">Attack &amp; defense from club results (Poisson/Dixon-Coles + Elo), recency-weighted and self-updating. Data through ${GEN}.</div>
      <table><tr><th>#</th><th>Team</th><th>ATK</th><th>DEF</th><th>Elo</th></tr>${rows}</table></div>`;
  }
  function lshow(tab){ document.querySelectorAll('#lgNav button').forEach(b=>b.classList.toggle('on',b.dataset.ltab===tab));
    document.querySelectorAll('#app-league .nsec').forEach(s=>s.classList.toggle('on',s.id===tab)); }

  window.openLeague=function(code){
    if(!LD[code]) return;
    cur=code; $('lgTitle').textContent=LD[code].name;
    buildPredict(); buildRatings(); lshow('lgPredict');
    if($('lgFoot')) $('lgFoot').innerHTML=`${LD[code].name} model — club results from football-data.co.uk, Poisson + Dixon-Coles, self-updating weekly. Data through ${GEN}.`;
    if(window.ELshow) window.ELshow('league');
  };

  document.querySelectorAll('#lgNav button').forEach(b=>b.onclick=()=>lshow(b.dataset.ltab));
  const md=$('lgInfoModal');
  if(md){ $('lgInfoBtn').onclick=()=>md.classList.add('show');
    md.onclick=e=>{ if(e.target===md||e.target.classList.contains('modal-close'))md.classList.remove('show'); }; }
})();
