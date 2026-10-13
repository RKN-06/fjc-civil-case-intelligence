import pandas as pd
import streamlit as st

import lab
import predictor

lab.setup('Outcome Predictor', '🎯')
st.title('Case Outcome Predictor')
st.caption('Estimate how a federal civil case ends: Dismissed, Judgment, Settled, Statistical closing or Transferred.')

models = lab.model_choices('classification')
names = {m['id']: m['name'] for m in models}
st.subheader('Inference engine')
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    model_id = st.selectbox('Select fitted classifier', list(names), format_func=lambda i: names[i] + (' · best' if i == models[0]['id'] else ''))
chosen = next(m for m in models if m['id'] == model_id)
c2.metric('Test accuracy', f"{chosen['test']['accuracy']:.1%}")
c3.metric('Test weighted F1', f"{chosen['test']['f1']:.3f}")
c4.metric('Test ROC-AUC', f"{chosen['test']['roc_auc']:.3f}")
rows = chosen.get('test_rows')
st.caption(f'Scored on {rows:,} held-out test cases.' if rows else 'Scored on the sampled run\'s 10,000 held-out test cases.')

st.subheader('Case details')
raw = lab.case_form('clf')
if st.button('Predict outcome', type='primary'):
    try:
        with st.spinner('Scoring the case with every classifier…'):
            x = predictor.predict('classification', lab.report()['meta']['run_id'], models, model_id, raw)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.session_state['clf'] = (model_id, raw, x)

if 'clf' in st.session_state:
    used_model, used_raw, x = st.session_state['clf']
    if (used_model, used_raw) != (model_id, raw):
        st.info('The inputs or model changed since this prediction. Press the button again to update it.')
    r = x['result']
    agree = sum(m['prediction'] == r['prediction'] for m in x['all_models'])
    st.divider()
    st.subheader('Outcome estimate')
    c1, c2, c3 = st.columns(3)
    c1.metric('Predicted outcome', r['prediction'])
    c2.metric('Model confidence', f"{r['confidence']:.1%}")
    c3.metric('Models in agreement', f"{agree} of {len(x['all_models'])}")
    notes = [f"{x['case']['circuit']} Circuit"]
    if x['case']['diversity']:
        notes.append(f"diversity case (residence code {x['case']['residence_code']})")
    if x['case']['high_stakes']:
        notes.append('high stakes (≥ $100k)')
    st.caption(' · '.join(notes))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Class confidence probabilities**')
        probs = sorted(r['probabilities'].items(), key=lambda kv: kv[1])
        st.plotly_chart(lab.bar([k for k, _ in probs], [100*v for _, v in probs], highlight=r['prediction'],
                                title='Model confidence (%)', fmt='%{x:.1f}%', xrange=[0, 100],
                                text=[f'{100*v:.1f}%' for _, v in probs]), width='stretch')
    with c2:
        st.markdown(f"**What moved this prediction**")
        if x['effects']:
            st.plotly_chart(lab.effects_chart(x['effects'], 100, 'Percentage points'), width='stretch')
            st.caption(f"Change in the chance of {r['prediction']} compared with resetting each input to the most common case "
                       '(no amount demanded, federal question, civil rights, plaintiff jury demand). A what-if check, not a causal claim.')
        else:
            st.info('Every input is at its most common value, so there is nothing to compare against. Change an input to see its effect.')

    st.markdown('**Every classifier on this case**')
    table = pd.DataFrame([{'Model': m['name'], 'Prediction': m['prediction'], 'Confidence': m['confidence'],
                           **{k: v for k, v in m['probabilities'].items()}} for m in x['all_models']])
    st.dataframe(table.style.format({c: '{:.1%}' for c in table.columns if c not in ('Model', 'Prediction')}), hide_index=True, width='stretch')
