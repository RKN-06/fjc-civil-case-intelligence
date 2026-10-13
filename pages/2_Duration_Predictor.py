import pandas as pd
import streamlit as st

import lab
import predictor

lab.setup('Duration Predictor', '⏱️')
st.title('Case Duration Predictor')
st.caption('Estimate the number of days from filing to termination for a federal civil case.')

models = lab.model_choices('regression')
names = {m['id']: m['name'] for m in models}
best = lab.report()['regression']['best']['id']
st.subheader('Inference engine')
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    model_id = st.selectbox('Select fitted regressor', list(names), index=list(names).index(best),
                            format_func=lambda i: names[i] + (' · best' if i == best else ''))
chosen = next(m for m in models if m['id'] == model_id)
c2.metric('Test R²', f"{chosen['test']['r2']:.3f}")
c3.metric('Test RMSE', f"{chosen['test']['rmse']:.0f} days")
c4.metric('Test MAE', f"{chosen['test']['mae']:.0f} days")
tuned = lab.full()['regression'].get('tuned')
st.caption('Scored on the sampled run\'s 10,000 held-out test cases. '
           + (f"The notebook's tuned Random Forest (R² {tuned['r2']:.3f}) is too large to host online, so this page uses the "
              'Random Forest from the sampled run.' if tuned else ''))

st.subheader('Case details')
raw = lab.case_form('reg')
if st.button('Predict duration', type='primary'):
    try:
        with st.spinner('Scoring the case with every regressor…'):
            x = predictor.predict('regression', lab.report()['meta']['run_id'], models, model_id, raw)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.session_state['reg'] = (model_id, raw, x)

if 'reg' in st.session_state:
    used_model, used_raw, x = st.session_state['reg']
    if (used_model, used_raw) != (model_id, raw):
        st.info('The inputs or model changed since this prediction. Press the button again to update it.')
    days = max(0.0, x['result']['prediction'])
    st.divider()
    st.subheader('Duration estimate')
    c1, c2, c3 = st.columns(3)
    c1.metric('Predicted time to resolution', f'{days:,.0f} days')
    c2.metric('About', f'{days/365:.1f} years')
    c3.metric('Typical error (MAE)', f"± {chosen['test']['mae']:.0f} days")
    if x['result']['prediction'] < 0:
        st.caption('The model returned a negative value, shown as 0 days.')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**What moved this prediction**')
        if x['effects']:
            st.plotly_chart(lab.effects_chart(x['effects'], 1, 'Days'), width='stretch')
            st.caption('Days added (+) or removed (−) compared with resetting each input to the most common case. A what-if check, not a causal claim.')
        else:
            st.info('Every input is at its most common value, so there is nothing to compare against. Change an input to see its effect.')
    with c2:
        st.markdown('**Every regressor on this case**')
        ms = sorted(x['all_models'], key=lambda m: m['prediction'])
        st.plotly_chart(lab.bar([m['name'] for m in ms], [max(0, m['prediction']) for m in ms], highlight=chosen['name'],
                                title='Predicted days', fmt='%{x:.0f} days'), width='stretch')
