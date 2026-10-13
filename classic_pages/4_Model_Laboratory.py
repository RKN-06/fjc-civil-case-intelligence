import plotly.graph_objects as go
import streamlit as st

import lab

lab.setup('Model Laboratory', '🧪')
st.title('Model Laboratory')
st.caption('Full benchmarking of every algorithm, test-set diagnostics and confusion matrices.')
full, run = lab.full(), lab.report()
meta = run['meta']

CLF = [('f1', 'Weighted F1'), ('accuracy', 'Accuracy'), ('precision', 'Precision'), ('recall', 'Recall'), ('roc_auc', 'ROC-AUC')]
REG = [('r2', 'R²'), ('rmse', 'RMSE (days)'), ('mae', 'MAE (days)')]
CLU = [('silhouette', 'Silhouette'), ('davies_bouldin', 'Davies–Bouldin'), ('calinski_harabasz', 'Calinski–Harabasz')]


def sampled(task, columns, split):
    rows = [dict(name=m['name'], **m.get(split, m.get('metrics', {}))) for m in run[task]['models']]
    return lab.leaderboard(rows, columns, run[task]['best']['name'])


def diagnostics(task):
    models = run[task]['models']
    names = {m['id']: m['name'] for m in models}
    model_id = st.selectbox('Inspect diagnostic plots for model', list(names), format_func=names.get, key=f'diag-{task}')
    m = next(x for x in models if x['id'] == model_id)
    if task == 'classification' and m.get('confusion_matrix'):
        cm, labels = m['confusion_matrix'], m['class_names']
        pct = [[100*v/max(1, sum(row)) for v in row] for row in cm]
        fig = go.Figure(go.Heatmap(z=pct, x=labels, y=labels, text=cm, texttemplate='%{text}', showscale=False,
                                   colorscale=[[0, '#F5F5EF'], [.5, lab.GOLD], [1, '#1F2839']],
                                   hovertemplate='Actual %{y}<br>Predicted %{x}<br>%{text} cases (%{z:.1f}% of row)<extra></extra>'))
        fig.update_layout(height=420, xaxis_title='Predicted', yaxis_title='Actual', yaxis_autorange='reversed',
                          margin=dict(l=10, r=10, t=10, b=40), paper_bgcolor='rgba(0,0,0,0)')
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"**Confusion matrix · {m['name']}** (sampled run, 10,000 test cases)")
            st.plotly_chart(fig, width='stretch')
        with c2:
            st.markdown('**Test metrics**')
            st.dataframe({label: [f"{m['test'][key]:.4f}"] for key, label in CLF}, hide_index=True, width='stretch')
    plots = [lab.plot_path(u) for u in m.get('plots', [])]
    plots = [p for p in plots if p.is_file() and not (task == 'classification' and p.name.endswith('_confusion.png'))]
    for col, path in zip(st.columns(max(1, len(plots))), plots):
        col.image(str(path), width='stretch')


tabs = st.tabs(['Classification Benchmarks', 'Regression Benchmarks', 'Clustering Diagnostics'])

with tabs[0]:
    f = full['classification']
    st.subheader('10-algorithm classification leaderboard')
    st.caption(f"Full data: {full['training_rows']:,} training and {full['test_rows']:,} held-out test cases, ranked by weighted F1.")
    st.dataframe(lab.leaderboard(f['models'], CLF, f['best']), width='stretch')
    saved = [m for m in lab.model_choices('classification') if m['id'] == 'classification_notebook']
    if saved:
        t = saved[0]['test']
        st.success(f"Best: **{f['best']}**. The tuned version saved in models/best_classifier.pkl (used by the Outcome Predictor) "
                   f"scores F1 {t['f1']:.4f} and accuracy {t['accuracy']:.1%} on the same test cases.")
    with st.expander('Sampled run used for the diagnostics below'):
        split = st.radio('Split', ['validation', 'test'], horizontal=True, key='clf-split')
        st.caption(f"{meta['training_rows']:,} training, {meta['validation_rows']:,} validation and {meta['test_rows']:,} test cases. "
                   f"Winner by validation F1: {run['classification']['best']['name']}.")
        st.dataframe(sampled('classification', CLF, split), width='stretch')
    st.subheader('Classification model diagnostics')
    diagnostics('classification')

with tabs[1]:
    f = full['regression']
    st.subheader('10-algorithm regression leaderboard')
    st.caption(f"Full data: {full['training_rows']:,} training and {full['test_rows']:,} held-out test cases, ranked by test R².")
    rows = ([dict(f['tuned'])] if f.get('tuned') else []) + f['models']
    st.dataframe(lab.leaderboard(rows, REG, f['tuned']['name'] if f.get('tuned') else f['models'][0]['name']), width='stretch')
    if f.get('tuned'):
        st.success(f"Best: **{f['tuned']['name']}** with R² {f['tuned']['r2']:.4f} and RMSE {f['tuned']['rmse']:.0f} days after tuning.")
    with st.expander('Sampled run used for the diagnostics below'):
        split = st.radio('Split', ['validation', 'test'], horizontal=True, key='reg-split')
        st.caption(f"Winner by validation RMSE: {run['regression']['best']['name']}.")
        st.dataframe(sampled('regression', REG, split), width='stretch')
    st.subheader('Regression model diagnostics')
    diagnostics('regression')

with tabs[2]:
    f = full['clustering']
    st.subheader('K-Means across k')
    st.caption(f"Scored on the notebook's {f['sample_rows']:,}-case sample. Higher silhouette and lower Davies–Bouldin are better.")
    st.dataframe(lab.leaderboard(f['models'], CLU, f['best']), width='stretch')
    if f.get('k5_comparison'):
        st.markdown('**K-Means vs Ward agglomerative at k=5**')
        st.dataframe(lab.leaderboard(f['k5_comparison'], CLU), width='stretch')
    with st.expander('Sampled run: K-Means and Ward for k = 2 to 8'):
        st.dataframe(lab.leaderboard([dict(name=m['name'], **m['metrics']) for m in run['clustering']['models']], CLU,
                                     run['clustering']['best']['name']), width='stretch')
    cols = st.columns(2)
    for i, name in enumerate(['cluster_elbow', 'cluster_silhouette', 'cluster_pca2d', 'cluster_dendrogram']):
        path = lab.ART/'plots'/f'{name}.png'
        if path.is_file():
            cols[i % 2].image(str(path), width='stretch')
