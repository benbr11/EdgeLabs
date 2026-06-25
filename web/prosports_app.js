/* NFL / NBA (Gaussian point-margin) + MLB (Poisson runs + starting pitcher).
   Reads window.NFL_DATA / NBA_DATA / MLB_DATA. Styled to match the rest of EdgeLab. */
(function(){
  const SP={
    nfl:{D:window.NFL_DATA, name:"NFL", kind:"margin", unit:"pts"},
    nba:{D:window.NBA_DATA, name:"NBA", kind:"margin", unit:"pts"},
    mlb:{D:window.MLB_DATA, name:"MLB", kind:"runs", unit:"runs"},
  };
  function erf(x){ const s=x<0?-1:1; x=Math.abs(x); const t=1/(1+0.3275911*x);
    const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x); return s*y; }
  const cdf=x=>0.5*(1+erf(x/Math.SQRT2));
  const $=id=>document.getElementById(id);
  const nmf=(D,t)=>(D.teams[t]&&D.teams[t].name)||t;

  function predMargin(D,h,a){ const T=D.teams,P=D.params;
    const eH=P.lg+T[h].off+T[a].dfn+P.hfa/2, eA=P.lg+T[a].off+T[h].dfn-P.hfa/2;
    const m=eH-eA; return {eH,eA,m,tot:eH+eA,winH:cdf(m/P.sd_m),winA:1-cdf(m/P.sd_m)}; }
  function predRuns(D,h,a,ph,pa){ const T=D.teams,P=D.params,PIT=D._pit;
    let lh=P.avg*T[h].att*T[a].dfn*P.home, la=P.avg*T[a].att*T[h].dfn;
    const adj=f=>f?(0.6*f+0.4):1;                       // SP ≈ 60% of a game's run prevention
    if(ph&&PIT[ph]) la*=adj(PIT[ph].factor); if(pa&&PIT[pa]) lh*=adj(PIT[pa].factor);
    lh=Math.max(0.5,lh); la=Math.max(0.5,la);
    const fac=k=>{let f=1;for(let i=2;i<=k;i++)f*=i;return f;}, Po=(k,l)=>Math.exp(-l)*Math.pow(l,k)/fac(k);
    let pH=0,pT=0,pA=0,ov=0,rl=0; const LINE=8.5;
    for(let i=0;i<16;i++)for(let j=0;j<16;j++){ const m=Po(i,lh)*Po(j,la);
      if(i>j)pH+=m; else if(i===j)pT+=m; else pA+=m; if(i+j>LINE)ov+=m; if(i-j>=2)rl+=m; }
    const share=lh/(lh+la); return {lh,la,winH:pH+pT*share,winA:pA+pT*(1-share),tot:lh+la,over:ov,line:LINE,rl}; }

  function teamOpts(D,order,sel){ return order.map(t=>`<option value="${t}"${t===sel?' selected':''}>${nmf(D,t)}</option>`).join(''); }
  function fillPit(code,which,team){ const D=SP[code].D, s=$(code+'-'+which), ps=(D._byteam[team]||[]).slice().sort((a,b)=>a.ra9-b.ra9);
    s.innerHTML=`<option value="">Team-average starter</option>`+ps.map(p=>`<option value="${p.n}">${p.n} — ${p.ra9.toFixed(2)} RA9</option>`).join('');
    if(ps.length)s.value=ps[0].n; }
  function renderRes(code){
    const S=SP[code],D=S.D, h=$(code+'-A').value, a=$(code+'-B').value, res=$(code+'-res');
    if(h===a){ res.innerHTML='<div class="note" style="padding:12px">Pick two different teams.</div>'; return; }
    if(S.kind==='margin'){ const r=predMargin(D,h,a), wh=r.winH*100, wa=r.winA*100, fav=r.m>=0?h:a;
      res.innerHTML=`<div class="probbar"><div class="seg" style="width:${Math.max(3,wh)}%;background:var(--accent);color:#fff">${wh.toFixed(0)}%</div><div class="seg" style="width:${Math.max(3,wa)}%;background:#f59e0b;color:#0e1b30">${wa.toFixed(0)}%</div></div>
        <div class="big"><div><div class="n">${wh.toFixed(1)}%</div><div class="l">${nmf(D,h)} (home)</div></div><div><div class="n">${wa.toFixed(1)}%</div><div class="l">${nmf(D,a)}</div></div></div>
        <div class="pills"><span class="pillstat">Projected <b>${r.eH.toFixed(0)}</b> – <b>${r.eA.toFixed(0)}</b></span>
        <span class="pillstat">Spread <b>${nmf(D,fav)} −${Math.abs(r.m).toFixed(1)}</b></span>
        <span class="pillstat">Total <b>${r.tot.toFixed(1)}</b> ${S.unit}</span></div>
        <div class="mini" style="margin-top:8px">Win % from a normal model of the point margin (SD ${D.params.sd_m.toFixed(1)}); home edge ${D.params.hfa.toFixed(1)} ${S.unit}.</div>`;
    } else { const ph=$(code+'-ph').value, pa=$(code+'-pa').value, r=predRuns(D,h,a,ph,pa), wh=r.winH*100, wa=r.winA*100;
      res.innerHTML=`<div class="probbar"><div class="seg" style="width:${Math.max(3,wh)}%;background:var(--accent);color:#fff">${wh.toFixed(0)}%</div><div class="seg" style="width:${Math.max(3,wa)}%;background:#f59e0b;color:#0e1b30">${wa.toFixed(0)}%</div></div>
        <div class="big"><div><div class="n">${wh.toFixed(1)}%</div><div class="l">${nmf(D,h)} (home)</div></div><div><div class="n">${wa.toFixed(1)}%</div><div class="l">${nmf(D,a)}</div></div></div>
        <div class="pills"><span class="pillstat">Exp. runs <b>${r.lh.toFixed(2)}</b> – <b>${r.la.toFixed(2)}</b></span>
        <span class="pillstat">Total <b>${r.tot.toFixed(1)}</b></span>
        <span class="pillstat">Over ${r.line} <b>${(r.over*100).toFixed(0)}%</b></span>
        <span class="pillstat">${nmf(D,h)} −1.5 <b>${(r.rl*100).toFixed(0)}%</b></span></div>
        <div class="mini" style="margin-top:8px">Starting pitcher factored in (≈60% of a game's run prevention). Win % includes extra innings.</div>`;
    }
  }
  function buildPredict(code,order){
    const S=SP[code],D=S.D;
    const pitch = S.kind==='runs' ? `<div><label>🧤 Home starter</label><select id="${code}-ph"></select></div><div><label>🧤 Away starter</label><select id="${code}-pa"></select></div>` : '';
    $(code+'-pred').innerHTML=`<div class="card">
      <div class="mini" style="margin-bottom:10px">${S.kind==='runs'?'Pick teams and starting pitchers — the starter is the biggest single-game factor in baseball.':'Pick a matchup — home team on the left.'}</div>
      <div class="nhlform"><div><label>Home</label><select id="${code}-A">${teamOpts(D,order,order[0])}</select></div>
      <div><label>Away</label><select id="${code}-B">${teamOpts(D,order,order[1])}</select></div>${pitch}</div>
      <div id="${code}-res"></div></div>`;
    $(code+'-A').onchange=()=>{ if(S.kind==='runs')fillPit(code,'ph',$(code+'-A').value); renderRes(code); };
    $(code+'-B').onchange=()=>{ if(S.kind==='runs')fillPit(code,'pa',$(code+'-B').value); renderRes(code); };
    if(S.kind==='runs'){ fillPit(code,'ph',order[0]); fillPit(code,'pa',order[1]); $(code+'-ph').onchange=()=>renderRes(code); $(code+'-pa').onchange=()=>renderRes(code); }
    renderRes(code);
  }
  function buildRatings(code,order){
    const S=SP[code],D=S.D; let rows,head;
    if(S.kind==='margin'){ head=`<th>#</th><th>Team</th><th>Net</th><th>For</th><th>Agst</th>`;
      rows=order.map((t,i)=>`<tr><td>${i+1}</td><td>${nmf(D,t)}</td><td><b>${D.teams[t].net>=0?'+':''}${D.teams[t].net.toFixed(1)}</b></td><td>${(D.params.lg+D.teams[t].off).toFixed(1)}</td><td>${(D.params.lg+D.teams[t].dfn).toFixed(1)}</td></tr>`).join('');
    } else { head=`<th>#</th><th>Team</th><th>R/G for</th><th>R/G agst</th>`;
      rows=order.map((t,i)=>`<tr><td>${i+1}</td><td>${t}</td><td>${(D.params.avg*D.teams[t].att).toFixed(2)}</td><td>${(D.params.avg*D.teams[t].dfn).toFixed(2)}</td></tr>`).join(''); }
    $(code+'-rate').innerHTML=`<div class="card"><h3>${S.name} power ratings</h3><div class="mini">Opponent-adjusted, recency-weighted, self-updating. Data through ${D.generated}.</div><table><tr>${head}</tr>${rows}</table></div>`;
  }
  function build(code){
    const S=SP[code],D=S.D; if(!D) return;
    if(S.kind==='runs'){ D._pit={}; D._byteam={}; (D.pitchers||[]).forEach(p=>{ D._pit[p.n]=p; (D._byteam[p.team]=D._byteam[p.team]||[]).push(p); }); }
    const order=Object.keys(D.teams).sort((a,b)=> S.kind==='margin' ? D.teams[b].net-D.teams[a].net : (D.teams[b].att-D.teams[b].dfn)-(D.teams[a].att-D.teams[a].dfn));
    $(code+'-body').innerHTML=`<nav class="nhlnav" id="${code}Nav"><button data-t="${code}-pred" class="on">Predict</button><button data-t="${code}-rate">Power ratings</button></nav>
      <main><div id="${code}-pred" class="nsec on"></div><div id="${code}-rate" class="nsec"></div></main>
      <div class="foot">${S.name} model · opponent-adjusted, self-updating · data through ${D.generated}</div>`;
    document.querySelectorAll('#'+code+'Nav button').forEach(b=>b.onclick=()=>{
      document.querySelectorAll('#'+code+'Nav button').forEach(x=>x.classList.toggle('on',x===b));
      document.querySelectorAll('#app-'+code+' .nsec').forEach(s=>s.classList.toggle('on',s.id===b.dataset.t)); });
    buildPredict(code,order); buildRatings(code,order);
  }
  ['nfl','nba','mlb'].forEach(build);
})();
