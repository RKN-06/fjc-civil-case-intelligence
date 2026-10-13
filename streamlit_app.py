"""Entry point: `streamlit run streamlit_app.py` (also the default file Streamlit Cloud looks for)."""
import streamlit as st

st.navigation([
    st.Page('Home.py', title='Home', icon='⚖️', default=True),
    st.Page('pages/1_Outcome_Predictor.py', title='Outcome Predictor', icon='🎯'),
    st.Page('pages/2_Duration_Predictor.py', title='Duration Predictor', icon='⏱️'),
    st.Page('pages/3_Case_Segments.py', title='Case Segments', icon='🧩'),
    st.Page('pages/4_Model_Laboratory.py', title='Model Laboratory', icon='🧪'),
]).run()
