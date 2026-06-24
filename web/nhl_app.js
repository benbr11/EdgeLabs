/* NHL section — styled to match the soccer app (team-colored bars, logos, pills,
   likely scorelines). Math ported from nhl_predict.py; reads window.NHL_DATA. */
(function(){
  if(!window.NHL_DATA){ return; }
  const D=window.NHL_DATA, P=D.params, T=D.teams, SOG=P.sog||29;
  const $=id=>document.getElementById(id);
  const teamsByRating=Object.keys(T).sort((a,b)=>(T[b].att100+T[b].def100)-(T[a].att100+T[a].def100));
  const GByTeam={}, GMAP={};
  (D.goalies||[]).forEach(g=>{ GMAP[g.n]=g; (GByTeam[g.team]=GByTeam[g.team]||[]).push(g); });

  const NHL_COLORS={ANA:"#F47A38",BOS:"#FFB81C",BUF:"#003087",CGY:"#C8102E",CAR:"#CC0000",CHI:"#CF0A2C",
    COL:"#6F263D",CBJ:"#002654",DAL:"#006847",DET:"#CE1126",EDM:"#FF4C00",FLA:"#B9975B",LAK:"#A2AAAD",
    MIN:"#154734",MTL:"#AF1E2D",NSH:"#FFB81C",NJD:"#CE1126",NYI:"#00539B",NYR:"#0038A8",OTT:"#DA1A32",
    PHI:"#F74902",PIT:"#FCB514",SJS:"#006D75",SEA:"#68A2B9",STL:"#002F87",TBL:"#002868",TOR:"#00205B",
    UTA:"#71AFE5",VAN:"#00843D",VGK:"#B4975A",WSH:"#C8102E",WPG:"#041E42"};
  const colorN=t=>NHL_COLORS[t]||"#3b82f6";
  const txtOn=hex=>{const n=parseInt(hex.slice(1),16);return (0.299*(n>>16)+0.587*((n>>8)&255)+0.114*(n&255))>150?"#0e1b30":"#fff";};
  const colDist=(a,b)=>{const x=parseInt(a.slice(1),16),y=parseInt(b.slice(1),16);return Math.abs((x>>16)-(y>>16))+Math.abs(((x>>8)&255)-((y>>8)&255))+Math.abs((x&255)-(y&255));};
  const logoN=t=>T[t].logo?`<img class="flag" src="${T[t].logo}" alt="" loading="lazy" onerror="this.style.display='none'">`:"";
  const nm=t=>T[t].name;

  function restF(d){ d=+d; if(d<=1)return 0.95; if(d===2)return 0.99; return 1.0; }
  function predict(home,away,o){
    o=o||{}; const a=T[home], b=T[away]; const hf=o.neutral?1:P.home_adv;
    let lh=P.avg*a.att*b.dfn*hf, la=P.avg*b.att*a.dfn;
    lh*=(o.availH??1)*restF(o.restH); la*=(o.availA??1)*restF(o.restA);
    lh*=Math.sqrt(2-restF(o.restA)); la*=Math.sqrt(2-restF(o.restH));
    const gadj=(g,t)=> (g&&GMAP[g])?(T[t].gsax-GMAP[g].gsax)*SOG:0;
    la+=gadj(o.goalieH,home); lh+=gadj(o.goalieA,away);
    lh=Math.max(0.5,lh); la=Math.max(0.5,la);
    const fac=k=>{let f=1;for(let i=2;i<=k;i++)f*=i;return f;};
    const Po=(k,l)=>Math.exp(-l)*Math.pow(l,k)/fac(k);
    let pH=0,pT=0,pA=0,o55=0,o65=0,pl=0,cells=[];
    for(let i=0;i<14;i++)for(let j=0;j<14;j++){ const m=Po(i,lh)*Po(j,la);
      if(i>j)pH+=m; else if(i===j)pT+=m; else pA+=m;
      if(i+j>5.5)o55+=m; if(i+j>6.5)o65+=m; if(i-j>=2)pl+=m;
      if(i<9&&j<9)cells.push([m,i,j]); }
    const fav=(pH+pA)?pH/(pH+pA):0.5; const winH=pH+pT*(0.5+(fav-0.5)*0.35);
    cells.sort((x,y)=>y[0]-x[0]);
    return {lh,la,winH,winA:1-winH,regH:pH,regOT:pT,regA:pA,o55,o65,pl,
            top:cells.slice(0,3).map(c=>({p:c[0]*100,i:c[1],j:c[2]}))};
  }
  const restOpts=`<option value="2">Normal rest</option><option value="1">Back-to-back</option><option value="4">Rested (3+ days)</option>`;
  const availOpts=`<option value="1">Full lineup</option><option value="0.92">Missing a key scorer</option><option value="0.85">Several out</option>`;
  const teamOpts=sel=>teamsByRating.map(t=>`<option value="${t}"${t===sel?' selected':''}>${nm(t)}</option>`).join('');
  function fillGoalies(selId,team){
    const s=$(selId), gs=GByTeam[team]||[];
    s.innerHTML=`<option value="">Team-average goalie</option>`+gs.map(g=>
      `<option value="${g.n}">${g.n} — ${g.sv.toFixed(3)} sv, GSAx ${(g.gsax>=0?'+':'')}${(g.gsax*100).toFixed(1)}</option>`).join('');
    if(gs.length) s.value=gs[0].n;
  }

  function render(){
    const home=$('nh').value, away=$('na').value, neutral=$('nneu').value==='1';
    if(home===away){ $('nresult').innerHTML=`<div class="note" style="padding:14px">Pick two different teams.</div>`; return; }
    const r=predict(home,away,{goalieH:$('ngh').value,goalieA:$('nga').value,
      restH:$('nrh').value,restA:$('nra').value,availH:+$('nih').value,availA:+$('nia').value,neutral});
    let cA=colorN(home),cB=colorN(away); if(colDist(cA,cB)<90)cB="#f59e0b";
    const wh=r.winH*100, wa=r.winA*100, tmax=Math.max.apply(null,r.top.map(s=>s.p))||1;
    $('nresult').innerHTML=`
      <div class="probbar">
        <div class="seg" style="width:${Math.max(3,wh)}%;background:${cA};color:${txtOn(cA)}">${wh.toFixed(0)}%</div>
        <div class="seg" style="width:${Math.max(3,wa)}%;background:${cB};color:${txtOn(cB)}">${wa.toFixed(0)}%</div>
      </div>
      <div class="big">
        <div><div class="n">${wh.toFixed(1)}%</div><div class="l">${logoN(home)} ${home}${neutral?'':' (home)'}</div></div>
        <div><div class="n">${wa.toFixed(1)}%</div><div class="l">${logoN(away)} ${away}</div></div>
      </div>
      <div class="xg">Regulation: ${home} ${(r.regH*100).toFixed(0)}% · tie→OT/SO ${(r.regOT*100).toFixed(0)}% · ${away} ${(r.regA*100).toFixed(0)}% <span style="opacity:.7">(win % above includes OT/shootout)</span></div>
      <div class="pills">
        <span class="pillstat">Exp. goals <b>${r.lh.toFixed(2)}</b> – <b>${r.la.toFixed(2)}</b></span>
        <span class="pillstat">Total <b>${(r.lh+r.la).toFixed(1)}</b></span>
        <span class="pillstat">Over 5.5 <b>${(r.o55*100).toFixed(0)}%</b></span>
        <span class="pillstat">Over 6.5 <b>${(r.o65*100).toFixed(0)}%</b></span>
        <span class="pillstat">${home} −1.5 <b>${(r.pl*100).toFixed(0)}%</b></span>
      </div>
      <h4>Most likely final scores</h4>
      ${r.top.map(s=>`<div class="scoreline"><span class="sl-teams">${logoN(home)} <b>${s.i}</b>–<b>${s.j}</b> ${logoN(away)}</span><span class="sl-bar"><i style="width:${(s.p/tmax*100).toFixed(0)}%"></i></span><span class="p">${s.p.toFixed(1)}%</span></div>`).join('')}`;
  }
  function predictScreen(){
    $('npredict').innerHTML=`<div class="card">
      <div class="mini" style="margin-bottom:10px">Pick a matchup. Set the <b>starting goalie</b> (the biggest single-game factor), rest, injuries and venue — the model updates instantly.</div>
      <div class="nhlform">
        <div><label>Home team</label><select id="nh">${teamOpts(teamsByRating[0])}</select></div>
        <div><label>Away team</label><select id="na">${teamOpts(teamsByRating[1])}</select></div>
        <div><label>🧤 Home goalie</label><select id="ngh"></select></div>
        <div><label>🧤 Away goalie</label><select id="nga"></select></div>
        <div><label>Home rest</label><select id="nrh">${restOpts}</select></div>
        <div><label>Away rest</label><select id="nra">${restOpts}</select></div>
        <div><label>Home availability</label><select id="nih">${availOpts}</select></div>
        <div><label>Away availability</label><select id="nia">${availOpts}</select></div>
        <div><label>Venue</label><select id="nneu"><option value="0">Home ice</option><option value="1">Neutral</option></select></div>
      </div></div>
      <div class="card" id="nresultcard"><div id="nresult"></div></div>`;
    fillGoalies('ngh',teamsByRating[0]); fillGoalies('nga',teamsByRating[1]);
    $('nh').onchange=()=>{ fillGoalies('ngh',$('nh').value); render(); };
    $('na').onchange=()=>{ fillGoalies('nga',$('na').value); render(); };
    ['ngh','nga','nrh','nra','nih','nia','nneu'].forEach(id=>$(id).onchange=render);
    render();
  }
  function ratingsScreen(){
    let rows=teamsByRating.map((t,i)=>{ const x=T[t],c=colorN(t);
      return `<tr><td>${i+1}</td><td><span class="tdot" style="background:${c}"></span>${logoN(t)} ${nm(t)}</td>
        <td><b>${x.att100.toFixed(0)}</b></td><td><b>${x.def100.toFixed(0)}</b></td>
        <td>${x.xgf.toFixed(2)}</td><td>${x.xga.toFixed(2)}</td><td>${x.elo}</td><td>${x.pp.toFixed(1)}</td><td>${x.pk.toFixed(1)}</td></tr>`;
    }).join('');
    $('nratings').innerHTML=`<div class="card"><h3>Power ratings</h3>
      <div class="mini">A blend of goals, Elo, shot-quality expected goals (xGF/xGA), goaltending and special teams — recency-weighted and self-updating.</div>
      <table><tr><th>#</th><th>Team</th><th>ATK</th><th>DEF</th><th>xGF</th><th>xGA</th><th>Elo</th><th>PP%</th><th>PK%</th></tr>${rows}</table></div>`;
  }
  function goaliesScreen(){
    const gs=(D.goalies||[]).slice().sort((a,b)=>b.gsax-a.gsax).slice(0,30);
    let rows=gs.map((g,i)=>`<tr><td>${i+1}</td><td>${logoN(g.team)} ${g.n}</td><td>${g.team}</td>
      <td>${g.sv.toFixed(3)}</td><td class="${g.gsax>=0?'edge':'neg'}">${g.gsax>=0?'+':''}${(g.gsax*100).toFixed(1)}</td></tr>`).join('');
    $('ngoalies').innerHTML=`<div class="card"><h3>Goaltending — GSAx leaders</h3>
      <div class="mini">Goals Saved Above Expected per 100 shots — stopping high-danger chances at a higher clip than an average goalie. The biggest single-game swing factor.</div>
      <table><tr><th>#</th><th>Goalie</th><th>Team</th><th>Sv%</th><th>GSAx/100</th></tr>${rows}</table></div>`;
  }
  function nshow(tab){ document.querySelectorAll('.nhlnav button').forEach(b=>b.classList.toggle('on',b.dataset.ntab===tab));
    document.querySelectorAll('#app-nhl .nsec').forEach(s=>s.classList.toggle('on',s.id===tab)); }
  document.querySelectorAll('.nhlnav button').forEach(b=>b.onclick=()=>nshow(b.dataset.ntab));
  const md=$('nhlInfoModal');
  if(md){ $('nhlInfoBtn').onclick=()=>md.classList.add('show');
    md.onclick=e=>{ if(e.target===md||e.target.classList.contains('modal-close'))md.classList.remove('show'); }; }
  if($('nhlSrcDate')) $('nhlSrcDate').textContent=D.generated;
  if($('nhlfoot')) $('nhlfoot').innerHTML=`NHL model: goals + Elo + shot-quality xG (MoneyPuck) + GSAx goaltending + special teams. Self-updating daily; rolls forward each season. Data through ${D.generated}.`;
  predictScreen(); ratingsScreen(); goaliesScreen();
})();
