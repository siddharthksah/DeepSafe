import pandas as pd
import plotly.express as px

df = pd.read_csv('../state.csv')

fig = px.line(df, x = 'Timestamp', y = 'Number of Inferences', title='Users')
fig.show()

# from dash import Dash, dcc, html, Input, Output
# import plotly.express as px
# import pandas as pd

# app = Dash(__name__)

# app.layout = html.Div([
#     html.H4('Simple stock plot with adjustable axis'),
#     html.Button("Switch Axis", n_clicks=0, 
#                 id='button'),
#     dcc.Graph(id="graph"),
# ])


# @app.callback(
#     Output("graph", "figure"), 
#     Input("button", "n_clicks"))
# def display_graph(n_clicks):
#     df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_apple_stock.csv') # replace with your own data source

#     if n_clicks % 2 == 0:
#         x, y = 'AAPL_x', 'AAPL_y'
#     else:
#         x, y = 'AAPL_y', 'AAPL_x'

#     fig = px.line(df, x=x, y=y)    
#     return fig


# app.run_server(debug=True)