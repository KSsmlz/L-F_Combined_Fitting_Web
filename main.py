import json
from os.path import split

import dash
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import pandas as pd
from dash import Input, Output, dcc, html, callback, State, clientside_callback
import plotly.express as px
import dash_bootstrap_components as dbc

data = {"concentration": [], "adsorption": [], "selected": []}
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Layout
app.layout = dbc.Container([
    html.H3("Adsorption Analysis"),
    html.Hr(),
    html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("1. Algorithm"),
                    dbc.CardBody([
                        dbc.Label("Fitting Model", html_for="fitting-model"),
                        dcc.Dropdown(id='fitting-model', options=[
                            {'label': 'Linear Model', 'value': 'linear-model'},
                            {'label': 'Freundlich Model', 'value': 'freundlich-model'},
                            {'label': 'Langmuir Model', 'value': 'langmuir-model'}
                        ], value='freundlich-model'),

                    ]),
                ], className='mb-3'),
                dbc.Card([
                    dbc.CardHeader("2. Data"),
                    dbc.CardBody(
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Concentration (C):", html_for="concentration-input"),
                                dbc.Input(type="text", id="concentration-input"),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Adsorption (S):", html_for="adsorption-input"),
                                dbc.Input(type="text", id="adsorption-input"),
                            ], width=6),
                        ], className='g-3')
                    ),
                    dbc.CardFooter(
                        dbc.Stack([
                            dbc.Button('Add Data', id='add-data', color='secondary'),
                            dbc.Button('Delete Last One', id='delete-last-one', color='secondary'),
                            dbc.Button('Delete All', id='delete-all', color='secondary'),
                        ], gap=2)
                    ),
                ], className='mb-3'),
                dbc.Card([
                    dbc.CardHeader("3. Select & Run"),
                    dbc.CardFooter(
                        dbc.Stack([
                            dbc.Button('Toggle Select Many', id='toggle-selected-many', color='secondary'),
                            dbc.Button('Fitting', id='run-fitting', color='secondary'),
                        ], gap=2)
                    ),
                ]),
            ], width=6),
            dbc.Col([
                dcc.Graph(
                    id='data-visualization', className='mb-3'
                    , figure=px.scatter(pd.DataFrame(data), x='concentration', y='adsorption', color='selected'),
                ),
                dbc.Card([
                    dbc.CardBody(id='fitting-report'),
                ]),
            ], width=6),
        ])
    ])
], className="p-5")


# Logic
@callback(
    Output('concentration-input', 'value'),
    Output('adsorption-input', 'value'),
    Output('data-visualization', 'figure', allow_duplicate=True),
    Input('add-data', 'n_clicks'),
    State('concentration-input', 'value'),
    State('adsorption-input', 'value'),
    prevent_initial_call=True)
def add_data(n_clicks, concentration, adsorption):
    if isinstance(concentration, (int, float)) and isinstance(adsorption, (int, float)):
        data['concentration'].append(float(concentration))
        data['adsorption'].append(float(adsorption))
        data['selected'].append(True)
        return None, None, px.scatter(pd.DataFrame(data), x='concentration', y='adsorption', color='selected')

    elif isinstance(concentration, str) and isinstance(adsorption, str):
        concentration = concentration.split('\t')
        adsorption = adsorption.split('\t')
        if len(concentration) == len(adsorption):
            data['concentration'].extend([float(c) for c in concentration])
            data['adsorption'].extend([float(s) for s in adsorption])
            data['selected'].extend([True] * len(concentration))
            return None, None, px.scatter(pd.DataFrame(data), x='concentration', y='adsorption', color='selected')


@callback(
    Output('data-visualization', 'figure', allow_duplicate=True),
    Input('delete-last-one', 'n_clicks'),
    prevent_initial_call=True)
def delete_last_one(n_clicks):
    if len(data['concentration']) > 0:
        data['concentration'].pop()
        data['adsorption'].pop()
        data['selected'].pop()
    return px.scatter(pd.DataFrame(data), x='concentration', y='adsorption', color='selected')


@callback(
    Output('data-visualization', 'figure', allow_duplicate=True),
    Input('delete-all', 'n_clicks'),
    prevent_initial_call=True)
def delete_all(n_clicks):
    data['concentration'] = []
    data['adsorption'] = []
    data['selected'] = []
    return px.scatter(pd.DataFrame(data), x='concentration', y='adsorption', color='selected')


@callback(
    Output('data-visualization', 'figure', allow_duplicate=True),
    Input('toggle-selected-many', 'n_clicks'),
    State('data-visualization', 'relayoutData'),
    prevent_initial_call=True
)
def toggle_relayout_data(n_clicks, relayoutData):
    if 'xaxis.range[0]' in relayoutData:
        for idx, (concentration, adsorption) in enumerate(zip(data['concentration'], data['adsorption'])):
            if (relayoutData['xaxis.range[0]'] < concentration < relayoutData['xaxis.range[1]']) and \
                    (relayoutData['yaxis.range[0]'] < adsorption < relayoutData['yaxis.range[1]']):
                data['selected'][idx] = not data['selected'][idx]
    else:
        for idx, _ in enumerate(data['concentration']):
            data['selected'][idx] = not data['selected'][idx]

    return px.scatter(pd.DataFrame(data), x='concentration', y='adsorption', color='selected')


@callback(
    Output('data-visualization', 'figure', allow_duplicate=True),
    Output('fitting-report', 'children'),
    Input('run-fitting', 'n_clicks'),
    State('fitting-model', 'value'),
    prevent_initial_call=True)
def run_fitting(n_clicks, value):
    def langmuir_adsorption(C, Smax, K):
        return (Smax * K * C) / (1 + K * C)

    def freundlich_adsorption(C, KF, n):
        return KF * (C ** n)

    C, S = [], []
    for concentration, adsorption, selected in zip(data['concentration'], data['adsorption'], data['selected']):
        if selected:
            C.append(concentration)
            S.append(adsorption)
    C = np.array(C, dtype=float)
    S = np.array(S, dtype=float)

    if value == 'freundlich-model':
        model = freundlich_adsorption
        (Smax_fit, K_fit), pcov, *rest = curve_fit(model, C, S, p0=[1.0, 1.0])
    elif value == 'langmuir-model':
        model = langmuir_adsorption
        (Smax_fit, K_fit), pcov, *rest = curve_fit(model, C, S, p0=[max(S), 1.0])

    C_smooth = np.linspace(min(C), max(C), 500)
    S_fit = model(C_smooth, Smax_fit, K_fit)

    S_fit_original = model(C, Smax_fit, K_fit)
    r_squared = r2_score(S, S_fit_original)

    fig = px.line(x=C_smooth, y=S_fit, labels={'x': 'Concentration', 'y': 'Adsorption'})
    fig.add_scatter(x=data['concentration'], y=data['adsorption'], mode='markers', name='UnSelected', opacity=0.5)
    fig.add_scatter(x=C, y=S, mode='markers', name='Selected')

    if value == 'freundlich-model':
        report = html.Div([
            html.P(f"Fitted Freundlich Model: S = {Smax_fit:.2f} * C^{K_fit:.2f}"),
        ])
    elif value == 'langmuir-model':
        report = html.Div([
            html.P(f"Fitted Langmuir Model: S = {Smax_fit:.2f} * {K_fit:.2f} * C / (1 + {K_fit:.2f} * C)"),
        ])

    report.children.append(html.P(f"R-squared: {r_squared}"))
    return fig, report


if __name__ == "__main__":
    app.run(host='0.0.0.0')
