"""Entry point: `streamlit run streamlit_app.py` (also the default file Streamlit Cloud looks for).

Shows the project's original dashboard (website/frontend) unchanged inside a Streamlit component.
Its data comes from bundled JSON files (dashboard/data); predictions are computed here in Python.
"""
from pathlib import Path
import json

import streamlit as st
import streamlit.components.v1 as components

import predictor

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title='FJC — Interactive Model Lab', page_icon='◈', layout='wide', initial_sidebar_state='collapsed')
st.markdown("""<style>
  header[data-testid="stHeader"], footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
  [data-testid="stStatusWidget"], [data-testid="stSidebarCollapsedControl"] {display:none!important}
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background:#F5F5EF}
  .block-container, [data-testid="stMainBlockContainer"] {padding:0!important;max-width:none!important}
  [data-testid="stVerticalBlock"] {gap:0!important}
  iframe[title$="fjc_dashboard"] {display:block;border:0}
</style>""", unsafe_allow_html=True)

dashboard = components.declare_component('fjc_dashboard', path=str(HERE/'dashboard'))


@st.cache_data
def report():
    return json.loads((HERE/'artifacts/dashboard.json').read_text(encoding='utf-8'))


def predict(body):
    """Same contract as POST /api/predict in website/backend/main.py."""
    task, model_id = body.get('task'), body.get('model_id')
    if task not in ('classification', 'regression'):
        return 404, {'detail': 'Unknown task'}
    data = report()
    models = data[task]['models']
    if task == 'classification':
        models = predictor.notebook_models() + models
    if not any(m['id'] == model_id for m in models):
        return 404, {'detail': 'Unknown model'}
    try:
        return 200, predictor.predict(task, data['meta']['run_id'], models, model_id, body.get('inputs') or {})
    except ValueError as exc:
        return 422, {'detail': str(exc)}
    except (FileNotFoundError, ImportError) as exc:
        return 503, {'detail': str(exc)}


answers = st.session_state.setdefault('answers', {})
request = dashboard(responses=answers, key='fjc', default=None)
if request and request.get('id') and request['id'] not in answers:
    status, body = predict(request.get('body') or {})
    for old in list(answers)[:-4]:  # keep the payload small
        answers.pop(old)
    answers[request['id']] = {'status': status, 'body': body}
    st.rerun()
