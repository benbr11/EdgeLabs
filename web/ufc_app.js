/* UFC — official rankings (info display). Reads window.UFC_RANKINGS (scraped from UFC.com).
   Fight predictor (next card: win% + method + round) is the next build phase. */
(function(){
  var D = window.UFC_RANKINGS; if(!D) return;
  var body = document.getElementById('ufc-body'); if(!body) return;
  var divs = D.divisions;
  function shortName(d){ return d.replace("Men's ","").replace("Pound-for-Pound","Pound-for-Pound").replace("Women's ","W "); }
  function table(name){
    var r = D.rankings[name]; if(!r) return '';
    var rows = '<tr style="color:var(--accent);font-weight:700"><td>C</td><td>'+r.champion+'</td><td>Champion</td></tr>';
    r.contenders.forEach(function(f,i){ rows += '<tr><td>'+(i+1)+'</td><td>'+f+'</td><td></td></tr>'; });
    return '<div class="card"><div class="mini" style="margin-bottom:8px">'+name+' — current official ranking (UFC.com)</div>'+
      '<table><tr><th>#</th><th>Fighter</th><th></th></tr>'+rows+'</table></div>';
  }
  var nav = divs.map(function(d,i){ return '<button data-d="'+d+'"'+(i===0?' class="on"':'')+'>'+shortName(d)+'</button>'; }).join('');
  body.innerHTML = '<nav class="nhlnav" id="ufcNav">'+nav+'</nav><main><div id="ufc-rank" class="nsec on"></div></main>'+
    '<div class="foot">Official UFC rankings · info display · through '+D.generated+' · source UFC.com. Skill-based fight predictor (next card: winner, method, round) is in build.</div>';
  function setDiv(d){
    document.getElementById('ufc-rank').innerHTML = table(d);
    document.querySelectorAll('#ufcNav button').forEach(function(b){ b.classList.toggle('on', b.dataset.d===d); });
  }
  document.querySelectorAll('#ufcNav button').forEach(function(b){ b.onclick=function(){ setDiv(b.dataset.d); }; });
  setDiv(divs[0]);
})();
