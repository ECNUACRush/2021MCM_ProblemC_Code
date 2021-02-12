import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objs import *
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import datetime
"""
PtoblemIV use SVR
"""
"""
Get the date n days after a date
"""
def getday(y,m,d,n):
    the_date = datetime.datetime(y,m,d)
    result_date = the_date + datetime.timedelta(days=n)
    d = result_date.strftime('%Y-%m-%d')
    return d
def SVR_logistic2(path,nums):
    ff = pd.read_csv(path, sep=',', index_col=False,
                     encoding="utf-8", low_memory=False)  ##read file
    y_train=[]
    X_train=[]
    count=0
    for item in ff.index:
        if ff.iloc[item - 1]["Lab Status"]==0 or ff.iloc[item - 1]["Lab Status"]==2:
            continue
        time=int(ff.iloc[item - 1]["Detection Date"])
        X_train.append([(_+count)*(2**(_-nums)) for _ in range(nums)])
        count+=1
        y_train.append(time)
    y_train=np.array(y_train)
    X_train=np.array(X_train)
    random_seed=13
    X_train,y_train=shuffle(X_train,y_train,random_state=random_seed)
    parameters={'kernel':['linear'],'gamma':np.logspace(-5,0,num=6,base=2.0),'C':np.logspace(-5,5,num=11,base=2.0)}##Using linear kernel function
    svr=SVR()
    grid_search=GridSearchCV(svr,parameters,cv=5,n_jobs=4,scoring="neg_mean_squared_error")
    grid_search.fit(X_train,y_train)
    y_pred=grid_search.predict(X_train)
    abs_vals=np.mean(np.abs(y_train-y_pred)*np.abs(y_train-y_pred))
    print(abs_vals)
    t = np.linspace(0, y_pred.shape[0], y_pred.shape[0]+ 1).tolist()
    figure1 = go.Scatter(x=t, y=y_pred, mode='lines+markers', name='pred', marker=dict(
        size=2,
        color="red",
        showscale=True
    ), line=dict(
        color='orange',
        width=2
    )
                         )
    figure2 = go.Scatter(x=t, y=y_train, mode='lines+markers', name='target', marker=dict(
        size=2,
        color="blue",
        showscale=True
    ), line=dict(
        color='#00FFFF',
        width=2
    )
                         )

    data = [figure1,figure2]
    x=np.linspace(1,700,11)
    y=[]
    for i in x:
        y.append(str(getday(2019,1,20,i+1)))
    layout = dict(
        title="ProblemIV",
        xaxis_title="X:index",
        yaxis_title="Y:data",
        legend_title="Legend",
        font=dict(
            family="Segoe UI Black",
            size=16,
            color="black"
        ),
        xaxis=dict(showline=True, showgrid=True, side='bottom', linecolor='black', gridcolor='white'),
        yaxis=dict(showline=True, showgrid=True, gridcolor='white', linecolor="black", side="left" ,tickmode='array',
                 tickvals = x,
                 ticktext = y
                   )
        , legend=dict(
            x=0.9,
            y=1,
            bgcolor="white",
            bordercolor="black",
            borderwidth=2,
            font=dict(size=18, color="black")
        )
    )
    fig = Figure(data=data, layout=layout)
    fig.show()

    X_newtrain=np.array([[(_+__)*(2**(_-nums)) for _ in range(nums)] for __ in np.linspace(count,15+count,16)])
    return grid_search.predict(X_newtrain)
if __name__=="__main__":
    SVR_logistic2("../Others/data/Five_data.csv",5)
