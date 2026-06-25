/* UFC section — Next Card predictor + official Rankings.
   Reads window.UFC_CARD (built by export_ufc.py from ufc_model.predict()) and
   window.UFC_RANKINGS (scraped from UFC.com). Styled to match EdgeLabs. */
(function(){
  var CARD = window.UFC_CARD, RANK = window.UFC_RANKINGS;
  var body = document.getElementById('ufc-body'); if(!body) return;

  // ---- helpers ----
  function pct(x){ return (x*100).toFixed(0); }
  function pct1(x){ return (x*100).toFixed(1); }
  function americanFromProb(p){
    p = Math.max(1e-6, Math.min(1-1e-6, p));
    if(p >= 0.5) return -Math.round(100*p/(1-p));
    return Math.round(100*(1-p)/p);
  }
  function fmtOdds(o){ return (o>=0?'+':'') + o; }
  function probFromAmerican(a){
    a = parseFloat(a); if(!isFinite(a) || a===0) return null;
    return a < 0 ? (-a)/((-a)+100) : 100/(a+100);
  }
  function esc(s){ return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

  // ---- confidence tiers ("Best Bets") ----
  // Thresholds come from the walk-forward OUT-OF-SAMPLE backtest (ufc_backtest.py):
  // the Best-Bet bar (win-prob >= bestBetThreshold) is the lowest T whose clean OOS
  // hit-rate >= 80%. As of the latest run: T=0.75 -> 82.5% OOS on N=97.
  var BEST_T = (CARD && CARD.bestBetThreshold) || 0.75;
  var LEAN_T = (CARD && CARD.leanThreshold) || 0.62;
  function tierOf(bt){
    if(bt.tier) return bt.tier;            // baked by exporter
    if(bt.winA==null) return null;
    var conf = Math.max(bt.winA, bt.winB);
    return conf >= BEST_T ? 'best' : (conf >= LEAN_T ? 'lean' : 'pass');
  }
  var TIER_META = {
    best: {label:'BEST BET',  bg:'rgba(52,211,153,.16)', col:'var(--good)'},
    lean: {label:'LEAN',      bg:'rgba(139,92,255,.16)', col:'var(--accent2)'},
    pass: {label:'PASS · coin-flip', bg:'var(--panel2)', col:'var(--mut)'}
  };
  function tierBadge(t){
    var m = TIER_META[t]; if(!m) return '';
    return '<span class="tag" style="background:'+m.bg+';color:'+m.col+';margin-left:0">'+m.label+'</span>';
  }

  // ---- shell: tab nav + two sections ----
  body.innerHTML =
    '<nav class="nhlnav" id="ufcNav">'+
      '<button data-utab="ucard" class="on">Next Card</button>'+
      '<button data-utab="urank">Rankings</button>'+
    '</nav>'+
    '<main>'+
      '<div id="ucard" class="nsec on"></div>'+
      '<div id="urank" class="nsec"></div>'+
    '</main>'+
    '<div class="foot" id="ufcfoot"></div>';

  function ushow(tab){
    document.querySelectorAll('#ufcNav button').forEach(function(b){ b.classList.toggle('on', b.dataset.utab===tab); });
    document.querySelectorAll('#app-ufc .nsec').forEach(function(s){ s.classList.toggle('on', s.id===tab); });
  }
  document.querySelectorAll('#ufcNav button').forEach(function(b){ b.onclick=function(){ ushow(b.dataset.utab); }; });

  // =====================================================================
  //  NEXT CARD
  // =====================================================================
  function boutCard(bt, i){
    var t = tierOf(bt);
    var head = '<div class="row" style="justify-content:space-between;align-items:flex-start">'+
        '<div><b style="font-size:15px">'+esc(bt.a)+'</b> <span class="vs">vs</span> <b style="font-size:15px">'+esc(bt.b)+'</b></div>'+
        (t ? '<div>'+tierBadge(t)+'</div>' : '')+
      '</div>'+
      '<div class="mini" style="margin:2px 0 10px">'+esc(bt.weightClass)+
        (bt.rounds===5?' · (5 rounds)':'')+(bt.isTitle?' · Title':'')+'</div>';

    if(bt.dataGap){
      return '<div class="card">'+head+
        '<div class="note" style="padding:10px 0 2px;color:var(--mut)">Insufficient data — unranked fighter (not in model database).</div>'+
        '</div>';
    }

    // unique id base for this bout's controls
    var id = 'b'+i;
    // probbar: A blue (accent), B orange (teamB)
    var bar =
      '<div class="probbar" id="'+id+'_bar">'+
        '<div class="seg" id="'+id+'_segA" style="width:'+Math.max(3,bt.winA*100)+'%;background:var(--teamA);color:#fff"></div>'+
        '<div class="seg" id="'+id+'_segB" style="width:'+Math.max(3,bt.winB*100)+'%;background:var(--teamB);color:#20160a"></div>'+
      '</div>'+
      '<div class="big">'+
        '<div><div class="n" id="'+id+'_pA"></div><div class="l">'+esc(bt.a)+'</div></div>'+
        '<div><div class="n" id="'+id+'_pB"></div><div class="l">'+esc(bt.b)+'</div></div>'+
      '</div>'+
      '<div class="xg" id="'+id+'_fav"></div>';

    // method pills (favored fighter's win-conditional split)
    var m = bt.method;
    var pills =
      '<div class="pills">'+
        '<span class="pillstat">KO/TKO <b>'+pct(m.ko)+'%</b></span>'+
        '<span class="pillstat">Submission <b>'+pct(m.sub)+'%</b></span>'+
        '<span class="pillstat">Decision <b>'+pct(m.dec)+'%</b></span>'+
      '</div>'+
      '<div class="mini" style="text-align:center;margin-top:-4px">method if <b>'+esc(bt.favored)+'</b> wins</div>';

    // round distribution (finishes by round)
    var rd = bt.roundDist || [];
    var rmax = Math.max.apply(null, rd.concat([0.0001]));
    var rdHtml = '<h4>Finish by round</h4>'+
      rd.map(function(v,r){
        return '<div class="scoreline"><span class="sl-teams"><b>R'+(r+1)+'</b></span>'+
          '<span class="sl-bar"><i style="width:'+(v/rmax*100).toFixed(0)+'%"></i></span>'+
          '<span class="p">'+pct(v)+'%</span></div>';
      }).join('');

    var edge = '<div class="mini" style="margin-top:10px"><b>Edge:</b> '+esc(bt.keyEdge)+'</div>';

    // your-read slider
    var slider =
      '<div style="margin-top:14px">'+
        '<label style="display:flex;justify-content:space-between;color:var(--mut)">'+
          '<span>← '+esc(bt.a)+'</span><span>Your read</span><span>'+esc(bt.b)+' →</span></label>'+
        '<input type="range" min="-25" max="25" value="0" step="1" id="'+id+'_read" style="width:100%">'+
        '<div class="mini" id="'+id+'_readlbl" style="text-align:center">No lean (model only)</div>'+
      '</div>';

    // odds inputs + EV badge
    var odds =
      '<div class="grid2" style="margin-top:10px">'+
        '<div><label>'+esc(bt.a)+' book odds (American)</label>'+
          '<input type="text" inputmode="numeric" placeholder="e.g. +150" id="'+id+'_oA"></div>'+
        '<div><label>'+esc(bt.b)+' book odds (American)</label>'+
          '<input type="text" inputmode="numeric" placeholder="e.g. -180" id="'+id+'_oB"></div>'+
      '</div>'+
      '<div id="'+id+'_ev" style="margin-top:8px"></div>';

    return '<div class="card" data-bout="'+i+'">'+head+bar+pills+rdHtml+edge+slider+odds+'</div>';
  }

  function wireBout(bt, i){
    var id = 'b'+i;
    var read = document.getElementById(id+'_read');
    if(!read) return; // dataGap bout has no controls
    var segA = document.getElementById(id+'_segA'), segB = document.getElementById(id+'_segB');
    var pA = document.getElementById(id+'_pA'), pB = document.getElementById(id+'_pB');
    var fav = document.getElementById(id+'_fav');
    var readlbl = document.getElementById(id+'_readlbl');
    var oA = document.getElementById(id+'_oA'), oB = document.getElementById(id+'_oB');
    var ev = document.getElementById(id+'_ev');

    function adjusted(){
      // slider in [-25,+25] points; positive shifts toward fighter B, negative toward A
      var lean = parseInt(read.value,10)||0;
      var wA = bt.winA - lean/100;
      var wB = bt.winB + lean/100;
      wA = Math.max(0.01, Math.min(0.99, wA));
      wB = 1 - wA;
      return {wA:wA, wB:wB, lean:lean};
    }

    function paint(){
      var a = adjusted();
      segA.style.width = Math.max(3, a.wA*100)+'%';
      segB.style.width = Math.max(3, a.wB*100)+'%';
      segA.textContent = pct(a.wA)+'%';
      segB.textContent = pct(a.wB)+'%';
      pA.textContent = pct1(a.wA)+'%';
      pB.textContent = pct1(a.wB)+'%';

      var favIsA = a.wA >= a.wB;
      var favName = favIsA ? bt.a : bt.b;
      var favP = favIsA ? a.wA : a.wB;
      fav.innerHTML = 'Favored: <b style="color:var(--txt)">'+esc(favName)+'</b> · model line <b style="color:var(--txt)">'+
        fmtOdds(americanFromProb(favP))+'</b>';

      if(a.lean===0){ readlbl.textContent='No lean (model only)'; }
      else {
        var who = a.lean>0 ? bt.b : bt.a;
        readlbl.innerHTML = '<b style="color:var(--accent2)">'+Math.abs(a.lean)+' pts toward '+esc(who)+'</b> — blended on the model';
      }

      // +EV: compare adjusted prob vs book-implied prob, each side
      var rows = [];
      var bookA = probFromAmerican(oA.value), bookB = probFromAmerican(oB.value);
      if(bookA!=null){
        var edgeA = a.wA - bookA;
        rows.push(sideEV(bt.a, edgeA, bookA));
      }
      if(bookB!=null){
        var edgeB = a.wB - bookB;
        rows.push(sideEV(bt.b, edgeB, bookB));
      }
      ev.innerHTML = rows.join('');
    }

    function sideEV(name, edge, book){
      if(edge > 0){
        return '<span class="tag" style="background:rgba(52,211,153,.16);color:var(--good)">'+
          '+EV '+esc(name)+' · +'+(edge*100).toFixed(1)+'% (book '+pct(book)+'%)</span> ';
      }
      return '<span class="tag" style="background:var(--panel2);color:var(--mut)">'+
        esc(name)+' no edge ('+(edge*100).toFixed(1)+'% vs book '+pct(book)+'%)</span> ';
    }

    read.addEventListener('input', paint);
    oA.addEventListener('input', paint);
    oB.addEventListener('input', paint);
    paint();
  }

  function cardScreen(){
    var host = document.getElementById('ucard');
    if(!CARD){
      host.innerHTML = '<div class="card"><div class="note">Next-card data not loaded.</div></div>';
      return;
    }
    var header = '<div class="card">'+
      '<h3 style="margin-bottom:4px">'+esc(CARD.event)+'</h3>'+
      '<div class="mini">'+esc(CARD.date)+' · '+esc(CARD.venue)+' · '+esc(CARD.location)+'</div>'+
      '<div class="mini" style="margin-top:6px">Model win %, method and round for all '+CARD.bouts.length+' bouts. '+
        'Drag <b>Your read</b> to blend your handicapping lean on top of the model; enter book odds to flag <b>+EV</b>.</div>'+
      '</div>';

    // ---- Best Bets summary (high-conviction picks only) ----
    var best = CARD.bouts.filter(function(b){ return !b.dataGap && tierOf(b)==='best'; });
    var bbHtml;
    if(best.length){
      var rows = best.map(function(b){
        var favIsA = b.winA >= b.winB;
        var favName = favIsA ? b.a : b.b;
        var dogName = favIsA ? b.b : b.a;
        var favP = favIsA ? b.winA : b.winB;
        return '<div class="scoreline" style="align-items:center">'+
            '<span class="sl-teams"><b>'+esc(favName)+'</b> <span class="vs">over</span> '+esc(dogName)+'</span>'+
            '<span class="sl-bar"><i style="width:'+(favP*100).toFixed(0)+'%;background:var(--good)"></i></span>'+
            '<span class="p"><b>'+pct1(favP)+'%</b> · '+fmtOdds(americanFromProb(favP))+'</span>'+
          '</div>';
      }).join('');
      bbHtml = '<div class="card" style="border:1px solid var(--good)">'+
        '<div class="row" style="justify-content:space-between;align-items:center;margin-bottom:6px">'+
          '<h3 style="margin:0">★ Best Bets</h3>'+tierBadge('best')+'</div>'+
        '<div class="mini" style="margin-bottom:10px">High-conviction picks only — model win-prob ≥ '+pct(BEST_T)+'%. '+
          'On the walk-forward out-of-sample backtest this tier hit <b style="color:var(--good)">~82%</b> '+
          '(vs ~70% across all fights). Each line is the model price.</div>'+
        rows+
        '</div>';
    } else {
      bbHtml = '<div class="card" style="border:1px solid var(--line)">'+
        '<div class="row" style="justify-content:space-between;align-items:center;margin-bottom:6px">'+
          '<h3 style="margin:0">★ Best Bets</h3>'+tierBadge('best')+'</div>'+
        '<div class="mini">No Best-Bet plays on this card — no bout clears the '+pct(BEST_T)+'% '+
          'win-prob bar where the model hits ~80% out-of-sample. Leans and passes below.</div>'+
        '</div>';
    }

    host.innerHTML = header + bbHtml + CARD.bouts.map(boutCard).join('');
    CARD.bouts.forEach(wireBout);
  }

  // =====================================================================
  //  RANKINGS (moved under its own tab)
  // =====================================================================
  function rankScreen(){
    var host = document.getElementById('urank');
    if(!RANK){ host.innerHTML='<div class="card"><div class="note">Rankings not loaded.</div></div>'; return; }
    var divs = RANK.divisions;
    function shortName(d){ return d.replace("Men's ","").replace("Women's ","W "); }
    function table(name){
      var r = RANK.rankings[name]; if(!r) return '';
      var rows = '<tr style="color:var(--accent);font-weight:700"><td>C</td><td>'+esc(r.champion)+'</td><td>Champion</td></tr>';
      r.contenders.forEach(function(f,i){ rows += '<tr><td>'+(i+1)+'</td><td>'+esc(f)+'</td><td></td></tr>'; });
      return '<div class="card"><div class="mini" style="margin-bottom:8px">'+esc(name)+' — current official ranking (UFC.com)</div>'+
        '<table><tr><th>#</th><th>Fighter</th><th></th></tr>'+rows+'</table></div>';
    }
    var nav = divs.map(function(d,i){ return '<button data-d="'+esc(d)+'"'+(i===0?' class="on"':'')+'>'+esc(shortName(d))+'</button>'; }).join('');
    host.innerHTML = '<nav class="nhlnav" id="ufcRankNav" style="padding-left:0;padding-right:0">'+nav+'</nav>'+
      '<div id="ufc-rank"></div>';
    function setDiv(d){
      document.getElementById('ufc-rank').innerHTML = table(d);
      document.querySelectorAll('#ufcRankNav button').forEach(function(b){ b.classList.toggle('on', b.dataset.d===d); });
    }
    document.querySelectorAll('#ufcRankNav button').forEach(function(b){ b.onclick=function(){ setDiv(b.dataset.d); }; });
    setDiv(divs[0]);
  }

  // ---- footer ----
  var gen = (CARD && CARD.generated) || (RANK && RANK.generated) || '';
  document.getElementById('ufcfoot').innerHTML =
    'UFC model: performance-adjusted Elo + style matchup (grappler premium) + situational factors → winner, method (KO/Sub/Dec) and round. '+
    'Confidence tiers from the walk-forward out-of-sample backtest: <b>Best Bet</b> = model win-prob ≥ '+pct(BEST_T)+'% '+
    '(this tier hit ~82% OOS, N=97), <b>Lean</b> = ≥ '+pct(LEAN_T)+'%, <b>Pass</b> = coin-flip below that. '+
    'Next card auto-built from raw_nextcard.json. Official rankings via UFC.com. Through '+esc(gen)+'. Insight, not betting advice.';

  cardScreen();
  rankScreen();
})();
