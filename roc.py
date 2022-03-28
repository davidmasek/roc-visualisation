"""Methods for ROC visualisation and computation."""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go


def generate_data():
    """Generate random data."""
    y = ['0', '0', '0', '1', '0', '1', '1']
    y_score = ['0.05', '0.1', '0.1', '0.4', '0.5', '0.7', '0.9']
    data = pd.DataFrame({'y': y, 'y_score': y_score})
    return data


def calculate_TPR_FPR(y, y_score, threshold):
    """Return tuple of (True Positive, Positive, False Positive, Negative).
    
    True Positive Rate and False Positive Rate can then be calculated as:
    TPR = TP / P,
    FPR = FP / N.
    
    The four values are returned so exact fractions can be displayed.
    """
    y_hat = np.array(y_score) > threshold
    y_hat = y_hat.astype(int)
    y = np.array(y)

    P = y.sum()
    N = len(y) - P

    TP = (y == 1) & (y == y_hat)
    TP = TP.sum()
    FP = (y == 1) & (y != y_hat)
    FP = FP.sum()

    return TP, P, FP, N


def roc_graph(y, y_score):
    """Plot Receiver Operator Characteristic curve."""
    fpr, tpr, thresholds = roc_curve(y, y_score)
    df = pd.DataFrame({
        'True Positive Rate': tpr,
        'False Positive Rate': fpr,
        'Threshold': thresholds,
    })

    fig = px.area(
        data_frame=df,
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.3f}), values from scikit-learn',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        hover_data=['Threshold', 'True Positive Rate', 'False Positive Rate'],
        # height=600,
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')

    return fig

def fixed_treshold_graph(y, y_score):
    """Currently returns inputs visualisation. Will be updated to visualise classification status based on dynamic threshold from slider."""
    # @TODO
    # - currently displays one "trace", should display always all four traces for TP,FP,TN,FN
    # - visuals (axes, etc) should be unified for different steps
    # - labels and texts should be updated

    # <Placeholder>
    fig = px.scatter(
        # data
        x=y_score, 
        y=np.zeros(len(y)), 
        color=[str(yi) for yi in y],
        # visual
        range_x=[0, 1.1], 
        size=np.ones(len(y))*10,
        height=250,
        labels={'x': 'y_score', 'y': '', 'color': 'class'},
        opacity=0.9,
        title='Inputs (sorted by prediction score)',
    )
    fig.update_xaxes(tickvals=y_score)
    fig.update_yaxes(showticklabels=False, nticks=1)
    # default hover is terrible, display score
    # fig.update_traces(hovertemplate='score: %{x}<extra></extra>')
    fig.update_traces(hovertemplate='<extra></extra>')
    return fig
    # </Placeholder>

    thresholds = np.unique(y_score)
    y = np.array(y)
    y_score = np.array(y_score)
    P = y.sum()
    N = len(y) - P

    classes = []
    for t in thresholds:
        y_hat = y_score > t
        y_hat = y_hat.astype(int)
        TP = (y_hat == 1) & (y == y_hat)    
        FP = (y_hat == 1) & (y != y_hat)    
        TN = (y_hat == 0) & (y == y_hat)    
        FN = (y_hat == 0) & (y != y_hat)
        classes.append({
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN,
        })
        
    fig = go.Figure()

    # Add traces, one for each slider step
    status_to_color = {
        'TP': 'blue',
        'FP': 'blue',
        'TN': 'red',
        'FN': 'red',
    }
    status_to_marker = {
        'TP': 'circle',
        'FP': 'x',
        'TN': 'circle',
        'FN': 'x',
    }
    for step, thresh in enumerate(thresholds):
        for status in ['TP', 'FP', 'TN', 'FN']:
            indices = classes[step][status]
            _y_score = y_score[indices]
            fig.add_trace(
                go.Scatter(
                    # data
                    x=_y_score, 
                    y=np.zeros(len(_y_score)), 
                    # markers
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=status_to_color[status],
                        symbol=status_to_marker[status],
                    ),
                    # other visuals
                    opacity=0.9,
                    visible=False,
                    name=f"Threshold={thresh:.2f}",
                ))

    # Make 1st trace visible
    fig.data[0].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": "Slider switched to step: " + str(i)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )


    # fig.update_xaxes(tickvals=y_score)
    # fig.update_yaxes(showticklabels=False, nticks=1)
    # fig.update_traces(hovertemplate='<extra></extra>')
    # fig.update_layout(hovermode="x unified")

    return fig

def inputs_graph(y, y_score):
    """Plot inputs on a line ordered by `y_score` and categorized by `y`. TPR and FPR lines are also plotted."""
    fig = px.scatter(
        # data
        x=y_score, 
        y=np.zeros(len(y)), 
        color=[str(yi) for yi in y],
        # visual
        range_x=[0, 1], 
        size=np.ones(len(y))*10,
        height=350,
        labels={'x': 'y_score', 'y': '', 'color': 'class'},
        opacity=0.9,
        title='True Positive and False Negative Rate given y_hat = y_score > threshold',
    )
    fig.update_xaxes(tickvals=y_score)
    fig.update_yaxes(showticklabels=False, nticks=1)
    # default hover is terrible, display score
    # fig.update_traces(hovertemplate='score: %{x}<extra></extra>')
    fig.update_traces(hovertemplate='<extra></extra>')


    fpr, tpr, thresholds = roc_curve(y, y_score)
    # add/recalculate edge values, scikit-learn does not automatically
    # provide value at thresh=0 and thresh=1
    fpr = fpr[thresholds < 1]
    fpr = np.concatenate([[0], fpr, [1]])
    tpr = tpr[thresholds < 1]
    tpr = np.concatenate([[0], tpr, [1]])
    thresholds = thresholds[thresholds < 1]
    thresholds = np.concatenate([[1], thresholds, [0]])

    fig.add_trace(go.Scatter(
        x=thresholds,
        y=fpr,
        name='FalsePR',
        hovertemplate='%{y:.2f}<extra>FPR</extra>',
        line_color='red',
    ))

    fig.add_trace(go.Scatter(
        x=thresholds,
        y=tpr,
        name='TruePR',
        hovertemplate='%{y:.2f}<extra>TPR</extra>',
        line_color='blue',
    ))
    fig.update_layout(hovermode="x unified")

    return fig
