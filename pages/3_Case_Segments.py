import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lab
import predictor

lab.setup('Case Segments', '🧩')
st.title('Case Segments')
seg = lab.load('segments.json')
st.caption(f"The notebook's best clustering, K-Means with k={seg['k']}, splits the {seg['training_cases']:,} training cases into segments. "
           'Describe a case to see which segment it falls in and how cases in that segment usually ended.')

profiles = seg['profiles']
order = ['Dismissed', 'Settled', 'Judgment', 'Transferred', 'Statistical']


@st.cache_resource
def kmeans():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return joblib.load(lab.ART/'models/kmeans_segments.pkl')


def profile_chart(selected=None):
    fig = go.Figure()
    for i, outcome in enumerate(order):
        fig.add_bar(name=outcome, x=[f"Segment {p['segment']}" for p in profiles],
                    y=[100*p['outcomes'].get(outcome, 0) for p in profiles], marker_color=lab.PALETTE[i],
                    hovertemplate=outcome + ': %{y:.1f}%<extra></extra>')
    fig.update_layout(barmode='stack', height=340, yaxis_title='Share of cases (%)', margin=dict(l=10, r=10, t=10, b=40),
                      legend=dict(orientation='h', y=-0.18), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    if selected is not None:
        fig.add_vrect(x0=selected - 0.45, x1=selected + 0.45, line_width=2, line_color=lab.GOLD)
    return fig


raw = lab.case_form('seg')
go_ = st.button('Find segment', type='primary')

selected = None
if go_:
    try:
        X = predictor.features(predictor.normalise(raw))
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    model = kmeans()
    selected = int(model.predict(np.asarray(X, dtype=float))[0])
    distances = model.transform(np.asarray(X, dtype=float))[0]
    p = next(p for p in profiles if p['segment'] == selected)
    top = max(p['outcomes'], key=p['outcomes'].get)
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Segment', f'Segment {selected}')
    c2.metric('Share of all cases', f"{p['share']:.1%}")
    c3.metric('Median duration', f"{p['median_days']:,.0f} days")
    c4.metric('Most common outcome', f"{top} ({p['outcomes'][top]:.0%})")
    others = sorted(range(len(distances)), key=lambda i: distances[i])
    st.caption('Distance to each segment centre (smaller is closer): ' +
               ' · '.join(f'Segment {i}: {distances[i]:.2f}' for i in others))

st.subheader('Segment profiles')
c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(profile_chart(selected), width='stretch')
with c2:
    table = pd.DataFrame([{'Segment': p['segment'], 'Cases': p['cases'], 'Share': p['share'],
                           'Median days': p['median_days'], 'Mean days': p['mean_days']} for p in profiles])
    st.dataframe(table.style.format({'Cases': '{:,}', 'Share': '{:.1%}', 'Median days': '{:,.0f}', 'Mean days': '{:,.0f}'}),
                 hide_index=True, width='stretch')
    st.caption('Outcomes and durations are the real results of the training cases assigned to each segment. '
               'Silhouette is about 0.23, so segments overlap and are a coarse grouping rather than distinct case types.')
