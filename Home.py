import streamlit as st

import lab

lab.setup('Home')
full, run = lab.full(), lab.report()

st.title('FJC Civil Case Intelligence')
st.caption('Machine learning on the Federal Judicial Center Integrated Database: '
           f"{full['training_rows'] + full['test_rows']:,} cleaned federal civil cases from the cv23 extract.")

st.markdown("""
Describe a federal civil case and the models estimate **how it will end** and **how long it will take**.
Use the pages in the sidebar:

- **Outcome Predictor**: Dismissed, Judgment, Settled, Statistical closing or Transferred, with class probabilities.
- **Duration Predictor**: days from filing to termination.
- **Case Segments**: which K-Means segment a case falls in, and how cases in that segment usually end.
- **Model Laboratory**: leaderboards for all 10 classifiers, 10 regressors and the clustering runs, with diagnostics.
""")

st.subheader('Best models (full data)')
clf = next(m for m in full['classification']['models'] if m['name'] == full['classification']['best'])
reg = full['regression']['tuned'] or full['regression']['models'][0]
clu = next(m for m in full['clustering']['models'] if m['name'] == full['clustering']['best'])
c1, c2, c3 = st.columns(3)
c1.metric('Classification', clf['name'], f"F1 {clf['f1']:.3f} · accuracy {clf['accuracy']:.1%}", delta_color='off')
c2.metric('Regression', reg['name'].replace(' Regressor', ''), f"R² {reg['r2']:.3f} · RMSE {reg['rmse']:.0f} days", delta_color='off')
c3.metric('Clustering', clu['name'], f"silhouette {clu['silhouette']:.3f}", delta_color='off')
st.caption(f"Trained on {full['training_rows']:,} cases and scored on {full['test_rows']:,} held-out cases by the project notebooks. "
           'Outcome accuracy across five classes is moderate, so treat any single prediction as an estimate.')

with st.expander('How the data was prepared'):
    st.markdown("""
- Source: `cv23.sas7bdat` from the FJC IDB, with labels from the Civil Codebook (1988 forward).
- Outcome: the 21 disposition codes grouped into 5 classes. Duration: days from filing to termination, capped at the 99th percentile.
- 16 case fields (court, origin, jurisdiction, nature of suit, jury demand, amount demanded, pro se, IFP, procedural progress and filing date)
  are scaled, one-hot encoded and reduced with PCA to 37 components that keep 95% of the variance.
- The prediction pages rebuild those exact features from what you enter, then score them with the saved models. Nothing is retrained online.
""")
    cols = st.columns(3)
    for col, name, cap in zip(cols, ['preprocessing_class_dist', 'preprocessing_pca_scree', 'preprocessing_pca_2d'],
                              ['Outcome classes', 'PCA explained variance', 'First two components']):
        path = lab.ART/'plots'/f'{name}.png'
        if path.is_file():
            col.image(str(path), caption=cap, width='stretch')
