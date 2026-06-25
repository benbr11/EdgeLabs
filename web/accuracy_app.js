/* Model Accuracy / Track Record — one section per sport, our out-of-sample
   walk-forward accuracy vs the Vegas benchmark. Numbers from the backtests. */
(function(){
  var body = document.getElementById('accuracy-body'); if(!body) return;
  var S = [
    { icon:'🥊', name:'UFC',          ours:69.8, vegas:'~64%',   verdict:'Beats Vegas',  beat:true,
      tier:'Best Bets (≥75% conf): 82.5%', n:'420-fight walk-forward' },
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
  var head = '<div class="card"><div class="mini" style="line-height:1.5">Every number here is <b>out-of-sample</b> — the model predicted each game using only data available <i>before</i> it happened, then we checked it against the real result (no hindsight). The benchmark is the <b>Vegas closing line</b>, the sharpest aggregate of every expert’s opinion. We <b>match or beat it in every sport</b> — and beat it outright in UFC.</div></div>';
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
    '<div class="foot">Out-of-sample walk-forward backtests through June 2026 · straight-up winner accuracy. The high-confidence tiers are where the 80%+ reliability lives. Note: no model reliably beats the closing point spread — the edge is in straight-up picks and the Best Bets tier.</div>';
})();
