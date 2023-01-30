import plotly.graph_objects as go
import pandas as pd
import numpy as np

def _make_local_explain_df(data_shap, features, idx):
  df = pd.DataFrame(np.minimum(data_shap[idx], 0), index=features.columns, columns=["neg_shap"])
  df["pos_shap"] = np.maximum(data_shap[idx], 0)
  df["value"] = [f"{c} = {features.iloc[idx, i]}" for i, c in enumerate(features.columns.tolist())]
  df["feature"] = features.columns
  df.index = range(0, len(features.columns))
  return df

def plot_local_explain(data_shap, features, idx = 0):
  df = _make_local_explain_df(data_shap, features, idx)

  fig = go.Figure()
  fig.add_trace(go.Bar(
    y=df.feature,
    x=df.neg_shap,
    text=df.value,
    name='Negative Impact',
    orientation='h',
    marker=dict(color='rgba(235, 78, 57, 0.6)')
  ))
  fig.add_trace(go.Bar(
    y=df.feature,
    x=df.pos_shap,
    text=df.value,
    name='Positive Impact',
    orientation='h',
    marker=dict(color='rgba(88, 100, 243, 0.6)'),
  ))

  fig.update_layout(
    title=f"Shap explainability for sample",
    xaxis_title="Shap values",
    barmode='stack'
  )

  return fig

import plotly
import plotly.graph_objects as go

def plot_global_explain(data_shap, features):
    traces = []
    data_shap_f = data_shap.swapaxes(0,1)
    data_features_f = features.values.swapaxes(0,1)
    for feature_indx, feature_name in enumerate(features.columns):
        f = data_shap_f[feature_indx]
        v = data_features_f[feature_indx]
        traces.append({'meanline': {'visible': True},
                'name': feature_name, 'type': 'violin', 
                 'x': pd.Series([feature_name]*len(v)), 
                 'y': pd.Series(f)})
    
    return {'data': traces, 'layout': go.Layout()}
    # plotly.offline.iplot({'data': traces, 'layout': go.Layout(title = f'{model_name} SHAP values for features')})
     


