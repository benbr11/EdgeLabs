/* Best Bets — per-sport box of the model's HIGHEST-CONFIDENCE picks: the selective
   tier that historically hits ABOVE the market (that's the real edge; the straight-up
   average only matches the market). Auto-populates from the current slate; where a sport
   is between seasons it shows the tier's out-of-sample hit-rate and fills in when games
   are scheduled. Injected after each sport's header so no per-section HTML is needed. */
(function(){
  function slot(appId){
    var app=document.getElementById(appId); if(!app) return null;
    var s=app.querySelector('.bb-slot');
    if(!s){ s=document.createElement('div'); s.className='bb-slot'; s.style.margin='0 0 4px';
      var head=app.querySelector('.sporthead');
      if(head && head.nextSibling) app.insertBefore(s, head.nextSibling);
      else app.insertBefore(s, app.firstChild); }
    return s;
  }
  function render(cfg){
    var s=slot(cfg.app); if(!s) return;
    var picks=[]; try{ picks=cfg.getPicks?cfg.getPicks():[]; }catch(e){ picks=[]; }
    var head='<div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap">'
      +'<div style="font-size:15px;font-weight:800">⭐ Best Bets</div>'
      +'<div class="mini" style="color:var(--good)">this tier hits <b>'+cfg.hit+'%</b> out-of-sample · market ~'+cfg.mkt+'%</div></div>';
    var body;
    if(picks.length){
      body=picks.slice(0,6).map(function(p){
        return '<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-top:1px solid var(--line)">'
          +'<div style="min-width:46px;text-align:center;font-weight:800;font-size:18px;color:var(--good)">'+p.conf+'%</div>'
          +'<div style="flex:1"><div style="font-weight:700">'+p.pick+'</div><div class="mini">'+p.ctx+'</div></div></div>';
      }).join('');
    } else {
      var n = cfg.slate ? (cfg.slate()||0) : 0;
      var msg = n>0
        ? 'No best-bet-tier picks on the current '+cfg.name+' slate — the scheduled games are too close to call for the ≥'+cfg.thr+'% tier (we only surface picks that clear the market). This fills in as matchups firm up.'
        : 'No games scheduled right now — your highest-confidence '+cfg.name+' picks (the ≥'+cfg.thr+'% tier that beats the market) post here automatically the moment the slate is live.';
      body='<div class="mini" style="padding:8px 0 2px;line-height:1.5">'+msg+'</div>';
    }
    s.innerHTML='<div class="card" style="border:1px solid rgba(52,211,153,.45)">'+head+body+'</div>';
  }

  // ---- soccer / World Cup: current LOCK-tier picks (knockout advance >=70%, or group win >=65%) ----
  function soccerPicks(){
    if(typeof predict!=='function' || !window.WC_DATA) return [];
    var D=window.WC_DATA, hosts=(D.params&&D.params.hosts)||[], out=[];
    (D.fixtures||[]).forEach(function(f){
      if(f.status!=='scheduled' || !D.teams[f.home] || !D.teams[f.away]) return;
      var ko=f.stage==='knockout';
      var host=hosts.indexOf(f.home)>-1?f.home:(hosts.indexOf(f.away)>-1?f.away:null);
      var r=predict(f.home,f.away,{knockout:ko,host:host});
      if(ko){ var adv=Math.max(r.advA,r.advB), who=r.advA>=r.advB?f.home:f.away;
        if(adv>=70) out.push({conf:adv.toFixed(0),pick:who+' to advance',ctx:f.home+' v '+f.away+' · '+f.date}); }
      else { var w=Math.max(r.pA,r.pB), wt=r.pA>=r.pB?f.home:f.away;
        if(w>=65) out.push({conf:w.toFixed(0),pick:wt+' to win',ctx:f.home+' v '+f.away+' · '+f.date}); }
    });
    return out.sort(function(a,b){return b.conf-a.conf;});
  }
  // ---- UFC: the exporter already tiers each bout; "best" = the >=75% tier ----
  function ufcPicks(){
    var C=window.UFC_CARD; if(!C) return [];
    return (C.bouts||[]).filter(function(b){return !b.dataGap && b.tier==='best';})
      .map(function(b){ var wp=Math.max(b.winA,b.winB)*100;
        return {conf:wp.toFixed(0),pick:b.favored,ctx:b.a+' v '+b.b+' · '+C.event}; })
      .sort(function(a,b){return b.conf-a.conf;});
  }

  function soccerSlate(){ if(!window.WC_DATA) return 0; var D=window.WC_DATA;
    return (D.fixtures||[]).filter(function(f){return f.status==='scheduled'&&D.teams[f.home]&&D.teams[f.away];}).length; }
  function ufcSlate(){ var C=window.UFC_CARD; return C?(C.bouts||[]).filter(function(b){return !b.dataGap;}).length:0; }
  var CFG=[
    {app:'app-soccer', name:'World Cup', thr:70, hit:87.5, mkt:55, getPicks:soccerPicks, slate:soccerSlate},
    {app:'app-ufc',    name:'UFC',       thr:75, hit:82.5, mkt:66, getPicks:ufcPicks, slate:ufcSlate},
    {app:'app-nba',    name:'NBA',       thr:80, hit:83.7, mkt:69},
    {app:'app-nfl',    name:'NFL',       thr:75, hit:75.0, mkt:66},
    {app:'app-mlb',    name:'MLB',       thr:80, hit:75.0, mkt:59},
    {app:'app-nhl',    name:'NHL',       thr:80, hit:68.5, mkt:58},
    {app:'app-league', name:'league',    thr:80, hit:87.5, mkt:55}
  ];
  function renderAll(){ CFG.forEach(render); }
  renderAll();
  window.EL_renderBestBets=renderAll;   // callable after data refresh / navigation
})();
