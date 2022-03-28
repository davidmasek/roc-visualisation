"""UI for ROC visualisation tool."""
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output, State

import numpy as np
import os

import roc

external_stylesheets = [
    {
        'href': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/brands.min.css',
        'rel': 'stylesheet',
        'crossorigin': 'anonymous'
    },
]


app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
DEBUG = os.environ.get('ENV', '').lower().startswith('dev')

# styles are automatically loaded from `assets` directory
app.layout = html.Div([
    html.Header([
        html.H1(
            'ROC Visualisation',
         ),
        html.A(
                [html.Span(className='fa-brands fa-github'), ' github'],
                href='https://github.com/davidmasek/roc-visualisation',
                className='button github-button',
                target='_blank',
            ),
    ]),
    html.Div([
        html.H2('Inputs:'),
        dash_table.DataTable(
            id='inputs-table',
            columns=[
                {
                    'name': 'True Value',
                    'id': 'y',
                    'deletable': False,
                    'renamable': False,
                },
                {
                    'name': 'Prediction',
                    'id': 'y_score',
                }
            ],
            data=roc.generate_data().to_dict('records'),
            editable=True,
            row_deletable=True,
            # style_table={'height': '300px', 'overflowY': 'auto'},
            page_action='none',
            fixed_rows={'headers': True},
            sort_action='native',
            style_cell_conditional=[
                {
                'if': {'column_id': 'y'},
                'width': '130px'
                },
            ],
        ),

        html.Button('Add Row', id='add-row-button', n_clicks=0),
    ], id='inputs-table-wrap'),
    html.Div([html.Span('Status: '), html.Span(id='status')], id='status-wrap'),
    dcc.Graph(id='fixed-threshold-graph'),
    dcc.Graph(id='inputs-graph'),
    dcc.Slider(
            min=0, 
            max=1, 
            step=0.01,
            value=0.5,
            marks={x: f'{x:.2f}' for x in np.linspace(0, 1, 9)},
            id='threshold-slider',
    ),
    html.Div(id='slider-output-container'),
    dcc.Graph(id='roc-graph'),
], className='container')

@app.callback(
    Output('slider-output-container', 'children'),
    Input('threshold-slider', 'value'),
    State('inputs-table', 'data'),
)
def update_slider(threshold, rows):
    """Diplay TPR and FPR based on threshold value."""
    try:
        y, y_score = _parse_data(rows)
    except Exception as ex:
        return str(ex)
    
    TP, P, FP, N = roc.calculate_TPR_FPR(y, y_score, threshold)

    return dcc.Markdown(f'**TPR**: {TP}/{P}, **FPR**: {FP}/{N}')

@app.callback(
    Output('inputs-table', 'data'),
    Input('add-row-button', 'n_clicks'),
    State('inputs-table', 'data'),
    State('inputs-table', 'columns'))
def add_row(n_clicks, rows, columns):
    """Diplay row to inputs table on button click."""
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows

def _parse_data(rows):
    """Parse data from table (Dash DataTable) to targets and scores (predictions). 
    Throws error on invalid values."""
    y = []
    y_score = []
    for row in rows:
        yi = row['y']
        if yi not in ['0', '1', 0, 1]:
            raise ValueError(f'True Value must be 0 or 1, got "{yi}"')
        try:
            yi_score = float(row['y_score'])
        except ValueError:
            raise ValueError(f'"{yi_score}" not convertable to float.')
        # 0 is not allowed to simplify the situation
        # so P(Y=1|x) > 0 is always true
        if yi_score > 1 or yi_score <= 0:
            raise ValueError(f'"{yi_score}" not in (0,1] range (we assume P(Y=1|x) > 0).')
        y.append(int(yi))
        y_score.append(yi_score)
    return y, y_score

@app.callback(
    Output('roc-graph', 'figure'),
    Output('inputs-graph', 'figure'),
    Output('fixed-threshold-graph', 'figure'),
    Output('status', 'children'),
    Input('inputs-table', 'data'),
    Input('roc-graph', 'figure'),
    Input('inputs-graph', 'figure'),
    Input('fixed-threshold-graph', 'figure'),
)
def update_inputs(rows, prev_fig, prev_inputs_fig, fixed_tresh_fig):
    """Plot ROC and display status. Keeps previous figure on error."""
    try:
        y, y_score = _parse_data(rows)
    except Exception as ex:
        return prev_fig, prev_inputs_fig, fixed_tresh_fig, str(ex)


    return (
        roc.roc_graph(y, y_score), 
        roc.inputs_graph(y, y_score), 
        roc.fixed_treshold_graph(y, y_score), 
        'ok'
        )


if __name__ == '__main__':
    app.run_server(debug=DEBUG)
