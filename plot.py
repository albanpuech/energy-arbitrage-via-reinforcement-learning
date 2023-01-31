import numpy as np
import pandas as pd
import plotly.express as px
import datetime


import plotly.graph_objects as go
from plotly.subplots import make_subplots

import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def display_profit(df_optim) :
    """
    Displays daily profits 
    """

    days =pd.to_datetime(df_optim["timestamp"].apply(lambda x: datetime.datetime(x.year, x.month, x.day)),utc=True).unique()
    
    daily_profit = []


    for i in range(len(days)) :
        daily_profit.append(   (-df_optim.price.iloc[i*24:(i+1)*24] @ df_optim.schedule.iloc[i*24:(i+1)*24]) / 10**6  )

    fig = make_subplots(specs=[[{"secondary_y": True}]])


    fig.add_trace(go.Scatter(x=df_optim.timestamp, y=df_optim.price,name='Price (EUR/MWh)',line={"shape":"hv"},showlegend=True),
    secondary_y=False)

    fig.add_trace(go.Bar(x=days,y=daily_profit,name="Daily profit (EUR)",offset=2,showlegend=True,opacity=0.5),
    secondary_y=True)

    fig.update_layout(
    title_text="Daily profit<br>Total: {} EUR<br>Mean: {} EUR".format(round(sum(daily_profit)), round(np.mean(daily_profit)))
    )

    fig.update_xaxes(title_text="Hour")

    fig.update_yaxes(title_text="Price (EUR/MWh)", secondary_y=False)
    fig.update_yaxes(title_text="Daily profit (EUR)", secondary_y=True)

    fig.update_layout(bargap=0.)
    fig.write_html("out/profit.html")
    fig.show()
    


    
    



def display_schedule(df_optim,start=None,end=None) :
    """
    Displays charge schedule between start datetime and end datetime 
    """
		
    if start == None :
        start = df_optim.timestamp.iloc[0]
    if end == None :
        end = df_optim.timestamp.iloc[-1]
    mask = (df_optim.timestamp<end) & (df_optim.timestamp>=start)
    df_to_show = df_optim[mask]


    fig = make_subplots(specs=[[{"secondary_y": True}]])


    fig.add_trace(go.Scatter(x=df_to_show.timestamp, y=df_to_show.price,name='Price (EUR/MWh)',line={"shape":"hv"},showlegend=True),
        secondary_y=False)

    fig.add_trace(
        go.Scatter(x=df_to_show.timestamp,y=df_to_show.SOC,name="SOC (%)",showlegend=True),secondary_y=True)



    color = None 
    count = 0

    shapes = []


    for h in df_to_show.index[np.sign(df_to_show.schedule).diff() != 0] : 
        
       
        if (df_to_show.schedule[h] == 0 and not color) or (df_to_show.schedule[h] > 0  and color == "green") or (df_to_show.schedule[h] < 0  and color == "red") :
            continue

        elif df_to_show.schedule[h] == 0 and color :
            shapes.append(dict(type="rect",x0=df_to_show.timestamp[start], y0=1, x1=df_to_show.timestamp[h], y1=100,  yref="y2",fillcolor=color, opacity=0.25, line_width=0))
            color = None

        elif df_to_show.schedule[h] > 0 and color == "red" : 
            shapes.append(dict(type="rect",x0=df_to_show.timestamp[start], y0=1, x1=df_to_show.timestamp[h], y1=100, yref="y2",fillcolor=color, opacity=0.25, line_width=0))
            start = h 
            color = "green"

        elif df_to_show.schedule[h] > 0 and not color :
            start = h
            color = "green"
            
        elif df_to_show.schedule[h] < 0 and color == "green" : 
            shapes.append(dict(type="rect",x0=df_to_show.timestamp[start], y0=1, x1=df_to_show.timestamp[h], y1=100, yref="y2", fillcolor=color, opacity=0.25, line_width=0))
            start = h 
            color = "red"
            
        elif df_to_show.schedule[h] < 0 and not color :
            start = h
            color = "red"






    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(step="all"),
                    dict(count=1,
                        label="1m",
                        step="month",
                        stepmode="backward"),
                    dict(count=2*7,
                        label="2w",
                        step="day",
                        stepmode="backward"),
                    dict(count=1*7,
                        label="1w",
                        step="day",
                        stepmode="backward"),
                    dict(count=2,
                        label="2d",
                        step="day",
                        stepmode="backward"),
                    dict(count=1,
                        label="1d",
                        step="day",
                        stepmode="backward"),
                    
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    fig.update_layout(
        shapes=shapes)


    fig.update_layout(
    title_text="Charge schedule.    Please use the buttons below to set the data range.<br>"
    )

    fig.update_xaxes(title_text="Date")

    fig.update_yaxes(title_text="Price (EUR/MWh)", secondary_y=False)
    fig.update_yaxes(title_text="SOC (%)", secondary_y=True)
        
    fig.show()  
    fig.write_html("out/schedule.html")





    