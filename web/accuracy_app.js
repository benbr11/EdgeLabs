/* Model Accuracy / Track Record — one section per sport, our out-of-sample
   walk-forward accuracy vs the Vegas benchmark. Numbers from the backtests.
   HONEST framing: these are straight-up winner accuracy at/near market level.
   On a large enough sample, no model here reliably beats the closing line — UFC
   was tested head-to-head vs de-vigged closing odds on 812 fights and sits at
   market (65.3% vs the book's 66.3%; model log-loss 0.630 vs book 0.615). */
(function(){
  var body = document.getElementById('accuracy-body'); if(!body) return;
  var S = [
    { icon:'🥊', name:'UFC',          ours:65.3, vegas:'~66%',   verdict:'At market', beat:false,
      tier:'High-confidence picks (≥80%): 82.0%', n:'812-fight walk-forward' },
    { icon:'🏀', name:'NBA',          ours:67.4, vegas:'~69%',   verdict:'Matches the books', beat:false,
      tier:'Top picks (≥80%): 83.7%', n:'2,462 games' },
    { icon:'🏈', name:'NFL',          ours:65.9, vegas:'~66%',   verdict:'Matches Vegas', beat:false,
      tier:'Top picks (≥75%): ~75%', n:'543 games' },
    { icon:'⚾', name:'MLB',          ours:56.4, vegas:'~58–60%', verdict:'Matches the market', beat:false,
      tier:'Top picks (≥80%): 75%', n:'4,857 games' },
    { icon:'🏒', name:'NHL',          ours:55.5, vegas:'~57–59%', verdict:'At market', beat:false,
      tier:'Top picks (≥80%): 68.5%', n:'2,792 games' },
    { icon:'⚽', name:'Soccer (1X2)', ours:52.1, vegas:'~50–55%', verdict:'Market-grade', beat:false,
      tier:'Top picks (≥80%): 87.5%', n:'3,444 games (3-way)' },
  ];
  var head = '<div class="card"><div class="mini" style="line-height:1.5">Every number here is <b>out-of-sample</b> — the model predicted each game using only data available <i>before</i> it happened, then we checked it against the real result (no hindsight). The benchmark is the <b>Vegas closing line</b>, the sharpest aggregate of every expert’s opinion. We sit <b>at market level</b> across the board: these are strong, honest predictions — but tested head-to-head on a large sample, <b>no model here reliably beats the closing line</b>. Treat them as a sharp second opinion, not a betting edge.</div></div>';
  var cards = S.map(function(s){
    var badge = s.beat
      ? 'style="background:rgba(52,211,153,.16);color:var(--good)"'
      : 'style="background:rgba(96,165,250,.16);color:var(--accent2)"';
    var bar = '<div class="probbar" style="height:10px;margin:10px 0 4px">'+
        '<div class="seg" style="width:'+s.ours+'%;background:var(--accent)"></div>'+
        '<div class="seg" style="width:'+(100-s.ours)+'%;background:var(--panel2)"></div>'+
      '</div>';
    return '<div class="card">'+
      '<div class="row" style="justify-content:space-between;align-items:center">'+
        '<div style="font-size:16px;font-weight:700">'+s.icon+' '+s.name+'</div>'+
        '<span class="tag" '+badge+'>'+s.verdict+'</span>'+
      '</div>'+
      '<div class="big" style="margin-top:8px">'+
        '<div><div class="n">'+s.ours.toFixed(1)+'%</div><div class="l">EdgeLabs · out-of-sample</div></div>'+
        '<div><div class="n" style="-webkit-text-fill-color:var(--mut);color:var(--mut);background:none">'+s.vegas+'</div><div class="l">Vegas line</div></div>'+
      '</div>'+ bar +
      '<div class="mini" style="text-align:center;margin-top:6px"><b>'+s.tier+'</b> &nbsp;·&nbsp; '+s.n+'</div>'+
    '</div>';
  }).join('');
  body.innerHTML = head + cards +
    '<div class="foot">Out-of-sample walk-forward backtests through June 2026 · straight-up winner accuracy at or near the market. The high-confidence tiers are where reliability is highest — but those are heavy favorites the market also nails, so they are a confidence filter, not a proven betting edge. Tested head-to-head vs de-vigged closing odds (UFC, 812 fights), the model sits at market, not above it.</div>';
})();
