"""Shared helpers for the Streamlit pages: reports, the case form and charts."""
from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import predictor

HERE = Path(__file__).resolve().parent
ART = HERE/'artifacts'
GOLD, NAVY, TEAL, CLAY = '#B69D74', '#344B68', '#6D8D88', '#AB796D'
PALETTE = [GOLD, NAVY, TEAL, CLAY, '#8F83A0', '#C49B63', '#7794B2', '#8B906D']


def setup(title, icon='⚖️'):
    st.set_page_config(page_title=f'{title} | FJC Case Intelligence', page_icon=icon, layout='wide')
    st.markdown("""<style>
      .block-container{padding-top:2.2rem;max-width:1250px}
      div[data-testid="stMetric"]{background:rgba(182,157,116,.08);border:1px solid rgba(182,157,116,.35);
        border-radius:10px;padding:14px 16px}
      div[data-testid="stMetricLabel"] p{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase}
    </style>""", unsafe_allow_html=True)


@st.cache_data
def load(name):
    return json.loads((ART/name).read_text(encoding='utf-8'))


def report():
    return load('dashboard.json')


def full():
    return load('notebook_results.json')


@st.cache_data
def schema():
    return predictor.schema()


def model_choices(task):
    """Saved models for a task, best first: the notebook's tuned MLP, then the sampled run."""
    data = report()
    models = [dict(id=m['id'], name=m['name'], test=m.get('test', {})) for m in data[task]['models']]
    if task == 'classification':
        models = predictor.notebook_models() + models
    return models


def options(key):
    return schema()['options'][key]


def _select(label, key, form_key, default, help=None):
    opts = options(key)
    values = [v for v, _ in opts]
    names = dict((v, n) for v, n in opts)
    index = next((i for i, v in enumerate(values) if str(v) == str(default)), 0)
    return st.selectbox(label, values, index=index, format_func=lambda v: names[v], key=f'{form_key}-{key}', help=help)


def case_form(form_key):
    """The case details form; returns raw inputs for predictor.predict."""
    t = schema()['typical']
    st.markdown('##### Court & filing')
    c1, c2, c3 = st.columns(3)
    with c1:
        district = _select('Court district', 'district', form_key, t['district'])
        st.caption(f"{options('circuits')[options('circuit_of')[district]]} Circuit")
    with c2:
        month = _select('Filing month', 'filing_month', form_key, t['filing_month'])
    with c3:
        origin = _select('How the case arrived', 'origin', form_key, t['origin'])
    st.markdown('##### Case type')
    c1, c2, c3 = st.columns(3)
    with c1:
        juris = _select('Jurisdiction basis', 'juris', form_key, t['juris'])
    with c2:
        nos = _select('Nature of suit', 'nos_group', form_key, t['nos_group'])
    with c3:
        classact = st.toggle('Class action', key=f'{form_key}-classact')
    plaintiff = defendant = None
    if int(juris) == 4:
        c1, c2 = st.columns(2)
        cit = [None] + [v for v, _ in options('citizenship')]
        names = dict(options('citizenship'))
        fmt = lambda v: 'Select…' if v is None else names[v]
        with c1:
            plaintiff = st.selectbox('Plaintiff citizenship', cit, format_func=fmt, key=f'{form_key}-pcit')
        with c2:
            defendant = st.selectbox('Defendant citizenship', cit, format_func=fmt, key=f'{form_key}-dcit')
    st.markdown('##### Parties & stakes')
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    with c1:
        demanded = st.number_input('Amount demanded ($ thousands)', 0, 9999, int(t['demanded']), step=25, key=f'{form_key}-demanded',
                                   help='0 = no amount entered, 9999 = over $10M. 100 or more counts as high stakes.')
    with c2:
        jury = _select('Jury demand', 'jury', form_key, t['jury'])
    with c3:
        prose = _select('Self-represented parties', 'prose', form_key, t['prose'])
    with c4:
        st.write('')
        ifp = st.toggle('In forma pauperis', key=f'{form_key}-ifp')
    procprog = _select('Procedural progress at disposition', 'procprog', form_key, t['procprog'],
                       help='Recorded when the case closed, so it carries strong signal about the outcome.')
    raw = dict(district=district, filing_month=int(month), origin=int(origin), juris=int(juris), nos_group=nos,
               classact=int(classact), demanded=float(demanded), jury=jury, prose=int(prose), ifp=int(ifp),
               procprog=int(procprog))
    if plaintiff and defendant:
        raw.update(plaintiff_cit=int(plaintiff), defendant_cit=int(defendant))
    return raw


def bar(labels, values, highlight=None, title='', fmt='%{x:.1f}', height=None, colors=None, xrange=None, text=None):
    fig = go.Figure(go.Bar(orientation='h', y=labels, x=values, text=text, textposition='outside' if text else None,
                           cliponaxis=False, marker_color=colors or [GOLD if l == highlight else NAVY for l in labels],
                           hovertemplate='%{y}: ' + fmt + '<extra></extra>'))
    fig.update_layout(height=height or max(260, 34*len(labels) + 80), margin=dict(l=10, r=40, t=10, b=40),
                      xaxis_title=title, xaxis_range=xrange, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def effects_chart(effects, scale=1.0, unit=''):
    rows = list(reversed(effects))
    return bar([e['input'] for e in rows], [scale*e['effect'] for e in rows], title=unit, fmt='%{x:+.1f}',
               colors=[GOLD if e['effect'] >= 0 else CLAY for e in rows])


def leaderboard(rows, columns, highlight=None):
    frame = pd.DataFrame([{label: r.get(key) for key, label in [('name', 'Model')] + columns} for r in rows])
    frame.index = range(1, len(frame) + 1)
    styler = frame.style.format({label: '{:,.4f}' for _, label in columns}, na_rep='—')
    if highlight:
        styler = styler.apply(lambda row: ['color:#B69D74;font-weight:600' if row['Model'] == highlight else '' for _ in row], axis=1)
    return styler


def plot_path(url):
    """Map a dashboard plot URL (/results/<run>/plots/x.png) to the bundled file."""
    parts = url.strip('/').split('/')
    return ART/'runs'/parts[1]/'plots'/parts[-1]
