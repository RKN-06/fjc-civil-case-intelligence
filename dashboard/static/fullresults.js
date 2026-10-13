// Full-data notebook leaderboards (scripts/export_notebook_results.py). Shown above the sampled run.
window.FullResults=(()=>{
 const cols={classification:[['f1','Weighted F1'],['accuracy','Accuracy'],['precision','Precision'],['recall','Recall'],['roc_auc','ROC-AUC']],
  regression:[['r2','R²'],['rmse','RMSE (days)'],['mae','MAE (days)']],clustering:[['silhouette','Silhouette'],['davies_bouldin','Davies–Bouldin'],['calinski_harabasz','Calinski–Harabasz']]};
 const label={classification:'Classification',regression:'Regression',clustering:'Clustering'};
 function headline(task,f){if(task==='classification'){const b=f.models.find(m=>m.name===f.best)||f.models[0];return [b.name,'F1 '+num(b.f1)+' · accuracy '+pct(b.accuracy)];}
  if(task==='regression'){const b=f.tuned||f.models[0];return [b.name,'R² '+num(b.r2)+' · RMSE '+Math.round(b.rmse)+' days'];}
  const b=f.models.find(m=>m.name===f.best)||f.models[0];return [b.name,'Silhouette '+num(b.silhouette)];}
 function winners(full){return ['classification','regression','clustering'].map(t=>{const [n,s]=headline(t,full[t]);return `<article class="winner"><div class="winner-top"><span class="eyebrow">${label[t]}</span><span class="chip">BEST · FULL DATA</span></div><h3>${esc(n)}</h3><strong class="full-score">${esc(s)}</strong><p>${esc(full[t].selection)}</p></article>`;}).join('');}
 function overview(d){const f=d.full;if(!f)return;const anchor=document.querySelector('#content .section-heading');if(!anchor)return;
  anchor.insertAdjacentHTML('beforebegin',`<div class="section-heading"><h2>Best models · full data</h2><span>${num(f.training_rows)} TRAINING · ${num(f.test_rows)} TEST CASES</span></div><div class="winners">${winners(f)}</div><p class="caption">From a complete run of the project notebooks on all prepared cases. The section below is a quicker sampled run used for the interactive charts, so its scores and winners can differ.</p>`);
  anchor.querySelector('h2').textContent='Sampled run front-runners';}
 function table(task,rows,best,extra){const c=cols[task];return `<div class="table-wrap"><table><thead><tr><th>#</th><th>Model</th>${c.map(([,n])=>`<th>${n}</th>`).join('')}</tr></thead><tbody>${rows.map((m,i)=>`<tr${m.name===best?' class="full-best"':''}><td>${i+1}</td><td>${esc(m.name)}${m.name===best?' <span class="best-dot">●</span>':''}</td>${c.map(([k])=>`<td>${num(m[k])}</td>`).join('')}</tr>`).join('')}${extra||''}</tbody></table></div>`;}
 function task(d){const f=d.full,lead=document.querySelector('#content .lead');if(!f||!lead)return;const t=(d.models[0].id||'').startsWith('cluster')?'clustering':d.models[0].id.split('_')[0];
  let extra='',note='';
  if(t==='regression'&&f.tuned){extra=`<tr class="full-best"><td>★</td><td>${esc(f.tuned.name)} <span class="best-dot">●</span></td><td>${num(f.tuned.r2)}</td><td>${num(f.tuned.rmse)}</td><td>${num(f.tuned.mae)}</td></tr>`;note=' The tuned Random Forest (★) is the final pick.';}
  if(t==='classification'&&f.tuning){note=Object.keys(f.tuning.params||{}).length?` Tuning ${esc(f.tuning.model)} changed F1 from ${num(f.tuning.before.f1)} to ${num(f.tuning.after.f1)}.`:` The notebook has no tuning grid for ${esc(f.tuning.model)}, so it was kept as trained.`;}
  const [n,s]=headline(t,f);
  const extraTable=t==='clustering'&&f.k5_comparison?`<p class="caption">At k=5 the notebook also compares K-Means with Ward agglomerative clustering:</p>${table('clustering',f.k5_comparison,'')}`:'';
  lead.insertAdjacentHTML('afterend',`<div class="winner-strip full-strip"><span class="chip">BEST · FULL DATA</span><strong>${esc(n)}</strong><span>${esc(s)} · ${esc(f.selection)}</span></div><details class="method" open><summary>Full-data leaderboard · ${num(f.training_rows)} training · ${num(f.test_rows)} test cases${t==='clustering'?' · '+num(f.sample_rows)+'-case sample':''}</summary><p class="caption">${t==='clustering'?'Internal cluster scores on the notebook’s 15,000-case sample of the training data':'Scores on the held-out test set'} from a complete run of the project notebook.${note} The explorer below uses a smaller sampled run, so its numbers and winner can differ.</p>${table(t,f.models,t==='regression'?'':f.best,extra)}${extraTable}</details>`);
  const strip=document.querySelector('#content .winner-strip:not(.full-strip) .chip');if(strip)strip.textContent='SAMPLED RUN WINNER';}
 return {overview,task};
})();
