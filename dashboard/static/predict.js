// Prediction studio: enter a case, pick a saved model, see the prediction and what moved it.
// Uses the helpers defined in app.js (plot, esc, num, pct, stats, select, chart, colors, $).
window.PredictPage=(()=>{
 let schema,mode='classification',last;
 const tasks={classification:{label:'Case outcome',verb:'Predict outcome',metrics:[['accuracy','Test accuracy',pct],['f1','Test weighted F1',num]]},
  regression:{label:'Resolution time',verb:'Predict duration',metrics:[['r2','Test R²',num],['rmse','Test RMSE',x=>days(x)]]}};
 const days=x=>Number.isFinite(x)?Math.round(x).toLocaleString()+' days':'—';
 const field=(label,html,hint='')=>`<label class="control">${label}${html}${hint?`<small>${hint}</small>`:''}</label>`;
 const options=(id,list,value)=>`<select id="p-${id}">${list.map(([v,n])=>`<option value="${esc(v)}"${String(v)===String(value)?' selected':''}>${esc(n)}</option>`).join('')}</select>`;
 const toggle=(id,label,on)=>`<label class="p-switch"><input type="checkbox" id="p-${id}"${on?' checked':''}><span>${label}</span></label>`;
 function form(t){const o=schema.options,cit=[['','Select…'],...o.citizenship];return `
 <div class="p-grid">
  <article class="p-card"><h3>Court &amp; filing</h3>${field('Court district',options('district',o.district,t.district),'<span id="p-circuit"></span>')}${field('Filing month',options('filing_month',o.filing_month,t.filing_month))}${field('How the case arrived',options('origin',o.origin,t.origin))}</article>
  <article class="p-card"><h3>Case type</h3>${field('Jurisdiction basis',options('juris',o.juris,t.juris))}${field('Nature of suit',options('nos_group',o.nos_group,t.nos_group))}<div id="p-diversity">${field('Plaintiff citizenship',options('plaintiff_cit',cit,''))}${field('Defendant citizenship',options('defendant_cit',cit,''))}</div>${toggle('classact','Class action',t.classact)}</article>
  <article class="p-card"><h3>Parties &amp; stakes</h3>${field('Amount demanded ($ thousands)',`<input id="p-demanded" type="number" min="0" max="9999" step="1" value="${t.demanded}">`,'0 = no amount entered · 9999 = over $10M · ≥100 counts as high stakes')}${field('Jury demand',options('jury',o.jury,t.jury))}${field('Self-represented parties',options('prose',o.prose,t.prose))}${toggle('ifp','Filed in forma pauperis',t.ifp)}</article>
  <article class="p-card"><h3>Progress</h3>${field('Procedural progress at disposition',options('procprog',o.procprog,t.procprog),'Recorded when the case closed, so it carries strong signal about the outcome.')}</article>
 </div>`;}
 function inputs(){const v=id=>$('p-'+id).value,x={district:v('district'),filing_month:+v('filing_month'),origin:+v('origin'),juris:+v('juris'),nos_group:v('nos_group'),classact:+$('p-classact').checked,demanded:$('p-demanded').value===''?0:+$('p-demanded').value,jury:v('jury'),prose:+v('prose'),ifp:+$('p-ifp').checked,procprog:+v('procprog')};
  if(x.juris===4&&v('plaintiff_cit')&&v('defendant_cit')){x.plaintiff_cit=+v('plaintiff_cit');x.defendant_cit=+v('defendant_cit');}return x;}
 function sync(){const o=schema.options;$('p-circuit').textContent=o.circuits[o.circuit_of[$('p-district').value]]+' Circuit';$('p-diversity').hidden=+$('p-juris').value!==4;}
 function modelPicker(){const m=schema.models[mode];return select('Fitted model','p-model',m.models.map(x=>[x.id,x.name+(x.id===m.best?' · best':'')]));}
 function metricsFor(id){const m=schema.models[mode].models.find(x=>x.id===id);return tasks[mode].metrics.map(([k,l,f])=>[l,f(m.test[k])]);}
 function render(){const box=$('content');box.innerHTML=`<span class="eyebrow">PREDICTION STUDIO</span><h1>Predict a case<span>.</span></h1><p class="lead">Describe a federal civil case, choose one of the fitted models, and see what it predicts, how confident it is, and which inputs drove the answer.</p>
 <div class="p-tabs" role="tablist">${Object.entries(tasks).map(([k,t])=>`<button role="tab" data-mode="${k}" aria-selected="${k===mode}" class="${k===mode?'active':''}">${t.label}<small>${k==='classification'?'Dismissed · Judgment · Settled · Statistical · Transferred':'Days from filing to termination'}</small></button>`).join('')}</div>
 <div class="section-heading"><h2>Inference engine</h2><span>${esc(schema.models[mode].models.length)} SAVED MODELS · RUN ${esc(schema.run_id)}</span></div>
 <div class="controls">${modelPicker()}<div class="stats mini p-model-stats" id="p-model-stats"></div></div>
 <div class="section-heading"><h2>Case details</h2><button class="subtle" id="p-reset">Reset to a typical case</button></div>${form(schema.typical)}
 <div class="p-actions"><button class="primary" id="p-go">${tasks[mode].verb} ↗</button><span class="caption" id="p-status"></span></div>
 <div id="p-result"></div>`;
 $('p-model').value=schema.models[mode].best;showModel();sync();
 document.querySelectorAll('.p-tabs button').forEach(b=>b.onclick=()=>{if(b.dataset.mode===mode)return;const keep=inputs();mode=b.dataset.mode;last=null;render();restore(keep);});
 $('p-model').onchange=()=>{showModel();if(last)run();};$('p-district').onchange=sync;$('p-juris').onchange=sync;
 $('p-reset').onclick=()=>{restore(schema.typical);$('p-plaintiff_cit').value='';$('p-defendant_cit').value='';sync();};$('p-go').onclick=run;}
 function restore(x){Object.entries(x).forEach(([k,v])=>{const el=$('p-'+k);if(!el||v===null||v===undefined)return;if(el.type==='checkbox')el.checked=!!+v;else el.value=v;});sync();}
 function showModel(){$('p-model-stats').innerHTML=stats(metricsFor($('p-model').value));}
 async function run(){const button=$('p-go'),status=$('p-status');button.disabled=true;status.textContent='Scoring with all '+schema.models[mode].models.length+' models… (the first run loads them from disk)';
  try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:mode,model_id:$('p-model').value,inputs:inputs()})}),x=await r.json();if(!r.ok)throw Error(typeof x.detail==='string'?x.detail:'Check the inputs and try again.');last=x;status.textContent='';show(x,true);}
  catch(e){status.textContent='';$('p-result').innerHTML=`<div class="notice">${esc(e.message)}</div>`;}finally{button.disabled=false;}}
 function show(x,scroll){const clf=x.task==='classification',r=x.result,box=$('p-result');
  const headline=clf?[['Predicted outcome',r.prediction],['Model confidence',pct(r.confidence)]]:[['Predicted time to resolution',days(Math.max(0,r.prediction))],['About',Math.max(0,r.prediction/365).toFixed(1)+' years']];
  const agree=clf?x.all_models.filter(m=>m.prediction===r.prediction).length:null;
  box.innerHTML=`<div class="section-heading"><h2>${clf?'Outcome estimate':'Duration estimate'}</h2><span>${esc(x.model.name.toUpperCase())}</span></div>
  <div class="stats p-headline">${stats([...headline,...metricsFor(x.model.id)])}</div>
  <p class="caption">${clf?`${agree} of ${x.all_models.length} models agree on <b>${esc(r.prediction)}</b>. `:`Typical error for this model on held-out cases is about ±${days(x.model.test.mae)} (MAE). `}${esc(x.case.circuit)} Circuit${x.case.diversity?' · diversity case (residence code '+x.case.residence_code+')':''}${x.case.high_stakes?' · high stakes (≥ $100k)':''}.${!clf&&r.prediction<0?' The model returned a negative value, shown as 0 days.':''}</p>
  <div class="charts">${clf?chart('p-proba','Class confidence probabilities','Share of the model\'s probability on each outcome'):''}${chart('p-effects','What moved this prediction',clf?`Change in P(${esc(r.prediction)}) versus resetting each input to the typical case`:'Days added (+) or removed (−) versus resetting each input to the typical case',!clf)}${chart('p-models','Every model on this case',clf?'Predicted outcome and confidence per model':'Predicted days per model',true)}</div>
  <details class="method"><summary>How to read these results ＋</summary><p>Inputs are rebuilt into the same scaled, one-hot and PCA features the notebooks used, then passed to the saved model. Nothing is retrained.</p><p>"What moved this prediction" swaps one input at a time back to the most common value in the dataset (${esc(schema.options.district.find(d=>d[0]===schema.typical.district)[1])}, federal question, civil rights, no amount demanded, plaintiff jury demand, no court action before issue joined) and re-scores the case. Inputs already at that value are not listed. It is a local what-if check, not a causal claim.</p><p>Test metrics come from the held-out sample in the current run. Accuracy across all five outcomes is modest, so treat any single prediction as an estimate.</p></details>`;
  if(clf){const cls=Object.keys(r.probabilities).sort((a,b)=>r.probabilities[a]-r.probabilities[b]);plot('p-proba',[{type:'bar',orientation:'h',y:cls,x:cls.map(c=>100*r.probabilities[c]),text:cls.map(c=>(100*r.probabilities[c]).toFixed(1)+'%'),textposition:'outside',cliponaxis:false,marker:{color:cls.map(c=>c===r.prediction?colors[0]:colors[1])},hovertemplate:'%{y}: %{x:.1f}%<extra></extra>'}],{height:300,margin:{l:95,r:50,t:10,b:45},xaxis:{title:{text:'Model confidence (%)'},range:[0,100]}});}
  const e=[...x.effects].reverse();
  if(e.length)plot('p-effects',[{type:'bar',orientation:'h',y:e.map(v=>v.input),x:e.map(v=>clf?100*v.effect:v.effect),marker:{color:e.map(v=>v.effect>=0?colors[0]:colors[3])},hovertemplate:clf?'%{y}: %{x:+.1f} points<extra></extra>':'%{y}: %{x:+.0f} days<extra></extra>'}],{height:Math.max(260,e.length*34+80),margin:{l:210,r:25,t:10,b:45},xaxis:{title:{text:clf?'Percentage points':'Days'},zeroline:true}});
  else $('p-effects').innerHTML='<p class="caption p-empty">Every input is at its typical value, so there is nothing to compare against. Change an input to see its effect.</p>';
  const ms=[...x.all_models].sort((a,b)=>clf?a.confidence-b.confidence:a.prediction-b.prediction);
  plot('p-models',clf?[...new Set(ms.map(m=>m.prediction))].map((label,i)=>{const rows=ms.filter(m=>m.prediction===label);return {type:'bar',orientation:'h',name:label,y:rows.map(m=>m.name),x:rows.map(m=>100*m.confidence),marker:{color:colors[Object.keys(r.probabilities).indexOf(label)%colors.length]},hovertemplate:'%{y}<br>'+esc(label)+' · %{x:.1f}%<extra></extra>'};})
   :[{type:'bar',orientation:'h',y:ms.map(m=>m.name),x:ms.map(m=>Math.max(0,m.prediction)),marker:{color:ms.map(m=>m.id===x.model.id?colors[0]:colors[1])},hovertemplate:'%{y}: %{x:.0f} days<extra></extra>'}],
   {height:Math.max(340,ms.length*30+90),margin:{l:230,r:25,t:10,b:60},barmode:'stack',xaxis:{title:{text:clf?'Confidence in predicted outcome (%)':'Predicted days'},...(clf?{range:[0,100]}:{})},yaxis:{categoryorder:'array',categoryarray:ms.map(m=>m.name)}});
  if(scroll)box.scrollIntoView({behavior:'smooth',block:'start'});}
 async function open(){const box=$('content');box.innerHTML='<div class="loading">Loading prediction studio…</div>';const r=await fetch('/api/predict/schema',{cache:'no-store'}),x=await r.json();if(!r.ok)throw Error(x.detail);schema=x;last=null;render();}
 return {open,redraw:()=>{if(last&&$('p-result'))show(last);}};
})();
