/* NFL / NBA (Gaussian point-margin) + MLB (Poisson runs + starting pitcher).
   Reads window.NFL_DATA / NBA_DATA / MLB_DATA. Offense & defense are modelled and shown
   INDEPENDENTLY (separate opponent-adjusted ratings), plus an overall ranking. */
(function(){
  const SP={
    nfl:{D:window.NFL_DATA, name:"NFL", kind:"margin", unit:"pts", epa:true,  tabs:["pred","rank","players","kick"]},
    nba:{D:window.NBA_DATA, name:"NBA", kind:"margin", unit:"pts", epa:false, tabs:["pred","rank","players"]},
    mlb:{D:window.MLB_DATA, name:"MLB", kind:"runs",  unit:"runs", tabs:["pred","rank","hitters","pitchers"]},
  };
  const TABNM={pred:"Predict", rank:"Rankings", players:"Players", kick:"Kicking", pitchers:"Pitchers", hitters:"Hitters"};
  function erf(x){ const s=x<0?-1:1; x=Math.abs(x); const t=1/(1+0.3275911*x);
    const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x); return s*y; }
  const cdf=x=>0.5*(1+erf(x/Math.SQRT2));
  const $=id=>document.getElementById(id);
  const nmf=(D,t)=>(D.teams[t]&&D.teams[t].name)||t;
  const pc=x=>(x*100).toFixed(0)+'%', p1=x=>(x*100).toFixed(1);

  // ---------- prediction ----------
  function predMargin(D,h,a){ const T=D.teams,P=D.params,k=P.kp||1;
    const eH=P.lg+k*(T[h].off+T[a].dfn)+P.hfa/2, eA=P.lg+k*(T[a].off+T[h].dfn)-P.hfa/2;
    const m=eH-eA; return {eH,eA,m,tot:eH+eA,winH:cdf(m/P.sd_m),winA:1-cdf(m/P.sd_m)}; }
  function predRuns(D,h,a,ph,pa){ const T=D.teams,P=D.params,PIT=D._pit;
    let lh=P.avg*T[h].att*T[a].dfn*P.home, la=P.avg*T[a].att*T[h].dfn;
    const adj=f=>f?(0.6*f+0.4):1;
    if(ph&&PIT[ph]) la*=adj(PIT[ph].factor); if(pa&&PIT[pa]) lh*=adj(PIT[pa].factor);
    lh=Math.max(0.5,lh); la=Math.max(0.5,la);
    const fac=k=>{let f=1;for(let i=2;i<=k;i++)f*=i;return f;}, Po=(k,l)=>Math.exp(-l)*Math.pow(l,k)/fac(k);
    let pH=0,pT=0,pA=0,ov=0,rl=0; const LINE=8.5;
    for(let i=0;i<16;i++)for(let j=0;j<16;j++){ const m=Po(i,lh)*Po(j,la);
      if(i>j)pH+=m; else if(i===j)pT+=m; else pA+=m; if(i+j>LINE)ov+=m; if(i-j>=2)rl+=m; }
    const share=lh/(lh+la); return {lh,la,winH:pH+pT*share,winA:pA+pT*(1-share),tot:lh+la,over:ov,line:LINE,rl}; }

  // ---------- orderings ----------
  function orders(code){ const D=SP[code].D, T=D.teams, ts=Object.keys(T);
    if(SP[code].kind==='runs'){
      return { off:[...ts].sort((a,b)=>T[b].att-T[a].att),
               def:[...ts].sort((a,b)=>T[a].dfn-T[b].dfn),
               net:[...ts].sort((a,b)=>(T[b].att-T[b].dfn)-(T[a].att-T[a].dfn)) };
    }
    return { off:[...ts].sort((a,b)=>T[b].off-T[a].off),
             def:[...ts].sort((a,b)=>T[a].dfn-T[b].dfn),
             net:[...ts].sort((a,b)=>(T[b].net!=null?T[b].net:T[b].off-T[b].dfn)-(T[a].net!=null?T[a].net:T[a].off-T[a].dfn)) }; }

  function teamOpts(D,order,sel){ return order.map(t=>`<option value="${t}"${t===sel?' selected':''}>${nmf(D,t)}</option>`).join(''); }
  function fillPit(code,which,team){ const D=SP[code].D, s=$(code+'-'+which), ps=(D._byteam[team]||[]).slice().sort((a,b)=>a.ra9-b.ra9);
    s.innerHTML=`<option value="">Team-average starter</option>`+ps.map(p=>`<option value="${p.n}">${p.n} — ${p.ra9.toFixed(2)} RA9</option>`).join('');
    if(ps.length)s.value=ps[0].n; }

  function renderRes(code){
    const S=SP[code],D=S.D, h=$(code+'-A').value, a=$(code+'-B').value, res=$(code+'-res');
    if(h===a){ res.innerHTML='<div class="note" style="padding:12px">Pick two different teams.</div>'; return; }
    if(S.kind==='margin'){ const r=predMargin(D,h,a), wh=r.winH*100, wa=r.winA*100, fav=r.m>=0?h:a;
      const O=orders(code), orank=t=>O.off.indexOf(t)+1, drank=t=>O.def.indexOf(t)+1;
      const edge = S.epa ? `<div class="mini" style="margin-top:8px">Matchup — <b>${nmf(D,h)}</b>: offense #${orank(h)}, defense #${drank(h)} &nbsp;·&nbsp; <b>${nmf(D,a)}</b>: offense #${orank(a)}, defense #${drank(a)}</div>` : '';
      res.innerHTML=`<div class="probbar"><div class="seg" style="width:${Math.max(3,wh)}%;background:var(--accent);color:#fff">${wh.toFixed(0)}%</div><div class="seg" style="width:${Math.max(3,wa)}%;background:var(--teamB);color:#0e1b30">${wa.toFixed(0)}%</div></div>
        <div class="big"><div><div class="n">${wh.toFixed(1)}%</div><div class="l">${nmf(D,h)} (home)</div></div><div><div class="n">${wa.toFixed(1)}%</div><div class="l">${nmf(D,a)}</div></div></div>
        <div class="pills"><span class="pillstat">Projected <b>${r.eH.toFixed(0)}</b> – <b>${r.eA.toFixed(0)}</b></span>
        <span class="pillstat">Spread <b>${nmf(D,fav)} −${Math.abs(r.m).toFixed(1)}</b></span>
        <span class="pillstat">Total <b>${r.tot.toFixed(1)}</b> ${S.unit}</span></div>${edge}
        <div class="mini" style="margin-top:6px">Win % from a normal model of the point margin (SD ${D.params.sd_m.toFixed(1)}); home edge ${D.params.hfa.toFixed(1)} ${S.unit}.</div>`;
    } else { const ph=$(code+'-ph').value, pa=$(code+'-pa').value, r=predRuns(D,h,a,ph,pa), wh=r.winH*100, wa=r.winA*100;
      res.innerHTML=`<div class="probbar"><div class="seg" style="width:${Math.max(3,wh)}%;background:var(--accent);color:#fff">${wh.toFixed(0)}%</div><div class="seg" style="width:${Math.max(3,wa)}%;background:var(--teamB);color:#0e1b30">${wa.toFixed(0)}%</div></div>
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
    const pitch = S.kind==='runs' ? `<div><label>Home starter</label><select id="${code}-ph"></select></div><div><label>Away starter</label><select id="${code}-pa"></select></div>` : '';
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

  // ---------- rankings (offense / defense / overall, independent) ----------
  function rankTables(code){
    const S=SP[code],D=S.D,T=D.teams,O=orders(code);
    const orank={},drank={}; O.off.forEach((t,i)=>orank[t]=i+1); O.def.forEach((t,i)=>drank[t]=i+1);
    let off,def,ovr;
    if(S.kind==='runs'){
      off=`<tr><th>#</th><th>Team</th><th>Runs/G</th></tr>`+O.off.map((t,i)=>`<tr><td>${i+1}</td><td>${nmf(D,t)}</td><td><b>${(D.params.avg*T[t].att).toFixed(2)}</b></td></tr>`).join('');
      def=`<tr><th>#</th><th>Team</th><th>Runs/G allowed</th></tr>`+O.def.map((t,i)=>`<tr><td>${i+1}</td><td>${nmf(D,t)}</td><td><b>${(D.params.avg*T[t].dfn).toFixed(2)}</b></td></tr>`).join('');
      ovr=`<tr><th>#</th><th>Team</th><th>Off rank</th><th>Pitch rank</th><th>Run diff</th></tr>`+O.net.map((t,i)=>`<tr><td>${i+1}</td><td>${nmf(D,t)}</td><td>${orank[t]}</td><td>${drank[t]}</td><td><b>${(D.params.avg*(T[t].att-T[t].dfn)).toFixed(2)}</b></td></tr>`).join('');
    } else if(S.epa){
      off=`<tr><th>#</th><th>Team</th><th>EPA/play</th><th>Pts/G</th><th>Success</th><th>Comp%</th><th>Y/att</th><th>Y/carry</th><th>3rd%</th><th>Score%</th><th>Giveaway</th></tr>`+
        O.off.map((t,i)=>{const x=T[t];return `<tr><td>${i+1}</td><td>${nmf(D,t)}</td><td><b>${x.off>=0?'+':''}${x.off.toFixed(3)}</b></td><td>${x.ppf.toFixed(1)}</td><td>${pc(x.osucc)}</td><td>${pc(x.cmp)}</td><td>${x.ypa.toFixed(1)}</td><td>${x.ypc.toFixed(1)}</td><td>${pc(x.third)}</td><td>${pc(x.dsc)}</td><td>${x.gv.toFixed(1)}</td></tr>`;}).join('');
      def=`<tr><th>#</th><th>Team</th><th>EPA/play</th><th>Pts/G</th><th>Success</th><th>Comp%</th><th>Y/att</th><th>Y/carry</th><th>3rd%</th><th>Score%</th><th>Takeaway</th><th>Sacks</th></tr>`+
        O.def.map((t,i)=>{const x=T[t];return `<tr><td>${i+1}</td><td>${nmf(D,t)}</td><td><b>${x.dfn>=0?'+':''}${x.dfn.toFixed(3)}</b></td><td>${x.ppa.toFixed(1)}</td><td>${pc(x.dsucc)}</td><td>${pc(x.cmpA)}</td><td>${x.ypaA.toFixed(1)}</td><td>${x.ypcA.toFixed(1)}</td><td>${pc(x.thirdA)}</td><td>${pc(x.dscA)}</td><td>${x.tk.toFixed(1)}</td><td>${x.sk.toFixed(1)}</td></tr>`;}).join('');
      ovr=`<tr><th>#</th><th>Team</th><th>Off</th><th>Def</th><th>Net EPA</th><th>Pts for</th><th>Pts agst</th><th>FG%</th><th>KR</th><th>PR</th></tr>`+
        O.net.map((t,i)=>{const x=T[t];return `<tr><td>${i+1}</td><td>${nmf(D,t)}</td><td>#${orank[t]}</td><td>#${drank[t]}</td><td><b>${x.net>=0?'+':''}${x.net.toFixed(3)}</b></td><td>${x.ppf.toFixed(1)}</td><td>${x.ppa.toFixed(1)}</td><td>${pc(x.fg)}</td><td>${x.kr.toFixed(1)}</td><td>${x.pr.toFixed(1)}</td></tr>`;}).join('');
    } else { // nba (points, no granular yet)
      off=`<tr><th>#</th><th>Team</th><th>Pts/G</th></tr>`+O.off.map((t,i)=>`<tr><td>${i+1}</td><td>${nmf(D,t)}</td><td><b>${(D.params.lg+T[t].off).toFixed(1)}</b></td></tr>`).join('');
      def=`<tr><th>#</th><th>Team</th><th>Pts/G allowed</th></tr>`+O.def.map((t,i)=>`<tr><td>${i+1}</td><td>${nmf(D,t)}</td><td><b>${(D.params.lg+T[t].dfn).toFixed(1)}</b></td></tr>`).join('');
      ovr=`<tr><th>#</th><th>Team</th><th>Off rank</th><th>Def rank</th><th>Net</th></tr>`+O.net.map((t,i)=>{const net=T[t].net!=null?T[t].net:T[t].off-T[t].dfn;return `<tr><td>${i+1}</td><td>${nmf(D,t)}</td><td>${orank[t]}</td><td>${drank[t]}</td><td><b>${net>=0?'+':''}${net.toFixed(1)}</b></td></tr>`;}).join('');
    }
    return {off,def,ovr};
  }
  function buildRank(code){
    const S=SP[code], t=rankTables(code), oLbl=S.kind==='runs'?'Pitching':'Defense';
    $(code+'-rank').innerHTML=`<div class="card">
      <div class="nhlnav" id="${code}RankNav"><button data-r="ovr" class="on">Overall</button><button data-r="off">Offense</button><button data-r="def">${oLbl}</button></div>
      <div class="mini" style="margin:8px 0 4px">Opponent-adjusted &amp; recency-weighted. Offense and ${oLbl.toLowerCase()} are estimated independently. Data through ${S.D.generated}.</div>
      <div class="rwrap"><table id="${code}-rtbl">${t.ovr}</table></div></div>`;
    const set=r=>{ $(code+'-rtbl').innerHTML=t[r];
      document.querySelectorAll('#'+code+'RankNav button').forEach(b=>b.classList.toggle('on',b.dataset.r===r)); };
    document.querySelectorAll('#'+code+'RankNav button').forEach(b=>b.onclick=()=>set(b.dataset.r));
  }

  // ---------- players (NBA — flat per-game leaderboard) ----------
  function buildNbaPlayers(code){
    const D=SP[code].D, L=(D.players||[]).slice(); const byTeam={};
    L.forEach(p=>{(byTeam[p.team]=byTeam[p.team]||[]).push(p);});
    const teamSel=`<select id="${code}-pfilter"><option value="">All teams — top 40 by value</option>${Object.keys(D.teams).sort((a,b)=>nmf(D,a)<nmf(D,b)?-1:1).map(t=>`<option value="${t}">${nmf(D,t)}</option>`).join('')}</select>`;
    function table(list){ return `<table><tr><th>#</th><th>Player</th><th>Team</th><th>Pos</th><th>Value</th><th>PPG</th><th>RPG</th><th>APG</th><th>SPG</th><th>BPG</th></tr>`+
      list.map((p,i)=>`<tr><td>${i+1}</td><td>${p.n}</td><td>${p.team}</td><td>${p.pos}</td><td><b>${p.val.toFixed(1)}</b></td><td>${p.ppg.toFixed(1)}</td><td>${p.rpg.toFixed(1)}</td><td>${p.apg.toFixed(1)}</td><td>${p.spg.toFixed(1)}</td><td>${p.bpg.toFixed(1)}</td></tr>`).join('')+`</table>`; }
    $(code+'-players').innerHTML=`<div class="card"><h3>Player values</h3>
      <div class="mini" style="margin-bottom:8px">Value = all-around per-game impact (pts + rebounds + assists + steals + blocks − turnovers), current season.</div>
      <div class="nhlform"><div><label>Filter</label>${teamSel}</div></div>
      <div class="rwrap" id="${code}-ptbl">${table(L.slice(0,40))}</div></div>`;
    $(code+'-pfilter').onchange=function(){ const tm=this.value;
      $(code+'-ptbl').innerHTML = table(tm ? (byTeam[tm]||[]).slice().sort((a,b)=>b.val-a.val) : L.slice(0,40)); };
  }
  // ---------- players (NFL) ----------
  function buildPlayers(code){
    const D=SP[code].D; if(Array.isArray(D.players)) return buildNbaPlayers(code);
    const P=D.players||{}; const flat=[];
    Object.keys(P).forEach(tm=>P[tm].forEach(p=>flat.push(Object.assign({tm},p))));
    flat.sort((a,b)=>b.val-a.val);
    const teamSel=`<select id="${code}-pfilter"><option value="">All teams — top 40</option>${Object.keys(D.teams).sort((a,b)=>nmf(D,a)<nmf(D,b)?-1:1).map(t=>`<option value="${t}">${nmf(D,t)}</option>`).join('')}</select>`;
    function table(list){ return `<table><tr><th>#</th><th>Player</th><th>Pos</th><th>Team</th><th>Value</th><th>Pass</th><th>Rush</th><th>Rec</th><th>TD</th></tr>`+
      list.map((p,i)=>`<tr><td>${i+1}</td><td>${p.n}</td><td>${p.pos}</td><td>${p.tm}</td><td><b>${p.val.toFixed(1)}</b></td><td>${p.py||''}</td><td>${p.ry||''}</td><td>${p.recy||''}</td><td>${p.td||''}</td></tr>`).join('')+`</table>`; }
    $(code+'-players').innerHTML=`<div class="card"><h3>Player values</h3>
      <div class="mini" style="margin-bottom:8px">Value = expected points added (EPA) over the last two seasons, recency-weighted — who actually drives each team.</div>
      <div class="nhlform"><div><label>Filter</label>${teamSel}</div></div>
      <div class="rwrap" id="${code}-ptbl">${table(flat.slice(0,40))}</div></div>`;
    $(code+'-pfilter').onchange=function(){ const tm=this.value;
      $(code+'-ptbl').innerHTML = tm ? table((P[tm]||[]).slice().sort((a,b)=>b.val-a.val).map(p=>Object.assign({tm},p))) : table(flat.slice(0,40)); };
  }
  // ---------- kicking (NFL) ----------
  function buildKick(code){
    const D=SP[code].D, K=D.kickers||[];
    const rows=K.map(k=>`<tr><td>${k.k}</td><td>${k.team}</td><td>${k.u30!==''?pc(+k.u30):'—'}</td><td>${k.b39!==''?pc(+k.b39):'—'}</td><td>${k.b49!==''?pc(+k.b49):'—'}</td><td>${k.b50!==''?pc(+k.b50):'—'}</td><td>${k.att}</td></tr>`).join('');
    $(code+'-kick').innerHTML=`<div class="card"><h3>Kicker accuracy by distance</h3>
      <div class="mini" style="margin-bottom:8px">Field-goal make rate by range — feeds special-teams value.</div>
      <div class="rwrap"><table><tr><th>Kicker</th><th>Team</th><th>&lt;30</th><th>30–39</th><th>40–49</th><th>50+</th><th>Att</th></tr>${rows}</table></div></div>`;
  }
  // ---------- hitters (MLB) ----------
  function buildHitters(code){
    const D=SP[code].D, H=(D.hitters||[]).slice(); const byTeam={};
    H.forEach(h=>{(byTeam[h.team]=byTeam[h.team]||[]).push(h);});
    const teamSel=`<select id="${code}-hfilter"><option value="">All teams — top 40 by OPS</option>${Object.keys(D.teams).sort((a,b)=>nmf(D,a)<nmf(D,b)?-1:1).map(t=>`<option value="${t}">${nmf(D,t)}</option>`).join('')}</select>`;
    function table(list){ return `<table><tr><th>#</th><th>Hitter</th><th>Team</th><th>OPS</th><th>AVG</th><th>HR</th><th>RBI</th><th>R</th><th>SB</th></tr>`+
      list.map((p,i)=>`<tr><td>${i+1}</td><td>${p.n}</td><td>${p.team}</td><td><b>${p.ops.toFixed(3)}</b></td><td>${p.avg.toFixed(3)}</td><td>${p.hr}</td><td>${p.rbi}</td><td>${p.r}</td><td>${p.sb}</td></tr>`).join('')+`</table>`; }
    $(code+'-hitters').innerHTML=`<div class="card"><h3>Hitter values</h3>
      <div class="mini" style="margin-bottom:8px">Ranked by OPS (on-base + slugging), recency-weighted over five seasons. Counting stats are per-season.</div>
      <div class="nhlform"><div><label>Filter</label>${teamSel}</div></div>
      <div class="rwrap" id="${code}-htbl">${table(H.slice(0,40))}</div></div>`;
    $(code+'-hfilter').onchange=function(){ const tm=this.value;
      $(code+'-htbl').innerHTML = table(tm ? (byTeam[tm]||[]).slice().sort((a,b)=>b.ops-a.ops) : H.slice(0,40)); };
  }
  // ---------- pitchers (MLB) ----------
  function buildPitchers(code){
    const D=SP[code].D, ps=(D.pitchers||[]).slice().sort((a,b)=>a.ra9-b.ra9);
    const rows=ps.map((p,i)=>`<tr><td>${i+1}</td><td>${p.n}</td><td>${p.team}</td><td><b>${p.ra9.toFixed(2)}</b></td><td>${(p.factor).toFixed(2)}</td><td>${p.ip}</td></tr>`).join('');
    $(code+'-pitchers').innerHTML=`<div class="card"><h3>Starting pitchers — run prevention</h3>
      <div class="mini" style="margin-bottom:8px">RA9 = runs allowed per 9 (regressed by innings). Factor &lt;1 suppresses opponent runs. Lower is better.</div>
      <div class="rwrap"><table><tr><th>#</th><th>Pitcher</th><th>Team</th><th>RA9</th><th>Factor</th><th>IP</th></tr>${rows}</table></div></div>`;
  }

  function build(code){
    const S=SP[code],D=S.D; if(!D) return;
    if(S.kind==='runs'){ D._pit={}; D._byteam={}; (D.pitchers||[]).forEach(p=>{ D._pit[p.n]=p; (D._byteam[p.team]=D._byteam[p.team]||[]).push(p); }); }
    const order=orders(code).net;
    const nav=S.tabs.map((t,i)=>`<button data-t="${code}-${t}"${i===0?' class="on"':''}>${TABNM[t]}</button>`).join('');
    const secs=S.tabs.map((t,i)=>`<div id="${code}-${t}" class="nsec${i===0?' on':''}"></div>`).join('');
    $(code+'-body').innerHTML=`<nav class="nhlnav" id="${code}Nav">${nav}</nav><main>${secs}</main>
      <div class="foot">${S.name} model · ${S.epa?'play-by-play (EPA + granular components)':'opponent-adjusted, self-updating'} · data through ${D.generated}</div>`;
    document.querySelectorAll('#'+code+'Nav button').forEach(b=>b.onclick=()=>{
      document.querySelectorAll('#'+code+'Nav button').forEach(x=>x.classList.toggle('on',x===b));
      document.querySelectorAll('#app-'+code+' .nsec').forEach(s=>s.classList.toggle('on',s.id===b.dataset.t)); });
    buildPredict(code,order); buildRank(code);
    if(S.tabs.includes('players')) buildPlayers(code);
    if(S.tabs.includes('kick')) buildKick(code);
    if(S.tabs.includes('hitters')) buildHitters(code);
    if(S.tabs.includes('pitchers')) buildPitchers(code);
  }
  ['nfl','nba','mlb'].forEach(build);
})();
