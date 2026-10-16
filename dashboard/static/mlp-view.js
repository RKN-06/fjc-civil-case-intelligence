// Neuron diagrams shown with the MLP: the single-layer perceptron (37 → 5, no hidden layer) beside
// the saved best MLP. Its layers and activation come from notebook_results.json (network), falling back to
// the notebook's MLPClassifier(hidden_layer_sizes=(64, 32)); scripts/train_perceptron.py trains the perceptron.
window.MlpView=(()=>{
 const HIDDEN=[64,32];
 const score=(d,name)=>{const r=d.full&&[...(d.full.tuned||[]),...d.full.models].find(m=>m.name===name);return r?`F1 ${num(r.f1)} · accuracy ${pct(r.accuracy)} (full data)`:'';};
 function draw(id,sizes,classes,act='ReLU'){
  const last=sizes.length-1,names=sizes.map((_,l)=>l===0?'Input':l===last?'Output':'Hidden layer '+l);
  const pos=sizes.map((n,l)=>Array.from({length:n},(_,i)=>[l,n===1?.5:i/(n-1)]));
  const ex=[],ey=[];
  for(let l=0;l<last;l++)for(const a of pos[l])for(const b of pos[l+1]){ex.push(a[0],b[0],null);ey.push(a[1],b[1],null);}
  const dark=document.body.classList.contains('navy');
  const colour=l=>l===0?'#7794B2':l===last?'#344B68':l===1?'#B69D74':'#C49B63';
  const nodes=pos.map((layer,l)=>({type:'scatter',mode:l===last?'markers+text':'markers',x:layer.map(p=>p[0]),y:layer.map(p=>p[1]),
   text:l===last?classes:undefined,textposition:'middle right',
   marker:{size:l===last?16:sizes[l]>40?6:8,color:colour(l),line:{width:1,color:dark?'#1F2839':'#FEFEFA'}},
   hovertext:layer.map((_,i)=>l===0?`Input ${i+1}: PC${i+1}`:l===last?`Output: ${classes[i]} (softmax)`:`${names[l]} · neuron ${i+1} · ${act}`),hoverinfo:'text'}));
  plot(id,[{type:'scattergl',mode:'lines',x:ex,y:ey,line:{width:.5,color:dark?'rgba(208,184,142,.10)':'rgba(52,75,104,.07)'},hoverinfo:'skip'},...nodes],
   {height:480,margin:{l:20,r:30,t:10,b:50},showlegend:false,
    xaxis:{showgrid:false,zeroline:false,tickmode:'array',tickvals:sizes.map((_,i)=>i),ticktext:sizes.map((n,i)=>`${names[i]}<br>${n} neurons`),range:[-.25,last+.75]},
    yaxis:{showgrid:false,zeroline:false,showticklabels:false,range:[-.04,1.04]}});
 }
 function attach(root,m,x,d){
  const old=document.getElementById('mlp-card');
  if(!/MLP/i.test(m.name)){if(old){['slp-net','mlp-net'].forEach(id=>Plotly.purge(id));old.remove();}return;}
  const inputs=(d.meta&&d.meta.features)||37,classes=x.classes||[];
  const net=d.full&&d.full.network,act=net&&net.activation==='tanh'?'tanh':'ReLU',mlpName=net?net.model:'MLP Classifier';
  const slp=[inputs,classes.length],mlp=net?net.layers:[inputs,...HIDDEN,classes.length];
  const params=s=>s.slice(0,-1).reduce((t,n,i)=>t+n*s[i+1]+s[i+1],0).toLocaleString();
  if(!old){const charts=root.querySelector('.charts');if(!charts)return;
   charts.insertAdjacentHTML('afterend',`<article class="chart-card wide" id="mlp-card" style="margin-top:18px"><div class="chart-heading"><h3>Network architecture · single-layer perceptron vs MLP</h3><p>Each dot is a neuron and each line a learned weight. Hover a neuron for its role.</p></div>
   <div style="display:grid;grid-template-columns:minmax(0,1fr) minmax(0,2fr);gap:10px">
    <div><p class="caption" style="padding:0 20px"><b>Single-layer perceptron</b> · ${slp.join(' → ')} · ${params(slp)} weights<br>${score(d,'Single-Layer Perceptron')}</p><div id="slp-net" class="chart" style="height:480px"></div></div>
    <div><p class="caption" style="padding:0 20px"><b>${esc(mlpName)}</b> · ${mlp.join(' → ')} · ${act} · ${params(mlp)} weights<br>${score(d,mlpName)}</p><div id="mlp-net" class="chart" style="height:480px"></div></div>
   </div>
   <p class="caption" style="padding:0 20px 18px">Both read a case as its ${inputs} PCA components. The perceptron connects them straight to the ${classes.length} outcomes, so it can only draw straight-line boundaries between them. The MLP's hidden neurons each add up their weighted inputs and apply ${act==='tanh'?'tanh (squashes to −1…1)':'ReLU (max(0, x))'}, which lets it learn curved boundaries; that is why it scores far higher. Outputs become probabilities through softmax.</p></article>`);}
  draw('slp-net',slp,classes);draw('mlp-net',mlp,classes,act);
 }
 return {attach};
})();
