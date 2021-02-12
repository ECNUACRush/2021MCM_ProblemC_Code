import pandas as pd
import numpy as np
import torch.nn as nn
import importlib
import plotly.graph_objects as go
from plotly.graph_objs import *
import torch
import plotly.io as pio
"""
ProblemV train and test
"""
"""
Sort by the Detection Date
"""
def allmerge_sort(path):
    ff = pd.read_csv(path, sep=',')
    ff=ff.sort_values(by=['Detection Date'])
    begin_time=str(ff.iloc[0]["Detection Date"])
    end_time=str(ff.iloc[len(ff.index)-1]["Detection Date"])
    print(begin_time,end_time)
    years_begin=int(begin_time[0:4])
    month_begin=int(begin_time[5:7])
    day_begin=int(begin_time[8:10])
    ##days=[0,28,59,89,120,150,181,212,242,273,303,334]
    ##days2=[31,28,31,30,31,30,31,31,30,31,30,31]
    days=[-31,0,28,59,89,120,150,181,212,242,273,303]
    list_all=[]
    for item in range(len(ff.index)):
        if ff.iloc[item]["Lab Status"]=="Positive ID":
            var=1
        else:
            var=0
        temp=str(ff.iloc[item]["Detection Date"])
        years_temp=int(temp[0:4])
        month_temp = int(temp[5:7])
        day_temp = int(temp[8:10])

        time=(years_temp-years_begin)*365+days[month_temp-month_begin]+(32-day_begin)+day_temp
        print(time,temp)
        list_all.append([time,ff.iloc[item]["Latitude"],ff.iloc[item]["Longitude"],ff.iloc[item]["Notes"],var])
    save = pd.DataFrame(list_all, columns=["Detection Date", "Latitude", "Longitude","Notes", "Lab Status"])
    save.to_csv('../Others/data/Five_data.csv', index=False, header=True)
"""
get train data and target data
"""
def imfotmation_get(path,batch_len,nums):
    ff = pd.read_csv(path, sep=',')
    result_list=[]
    submit_dict={}
    positive_dict={}
    for item in ff.index:
        v=int(ff.iloc[item-1]["Detection Date"]/batch_len)
        if v not in submit_dict.keys():
            submit_dict[v]=1
        else:
            submit_dict[v]+=1
        if ff.iloc[item-1]["Lab Status"]==1:
            if v not in positive_dict.keys():
                positive_dict[v] = 1
            else:
                positive_dict[v] += 1
    last_submit=np.array([[_,submit_dict[_]] for _ in submit_dict.keys()])
    last_positive=np.array([[_,positive_dict[_]] for _ in positive_dict.keys()])
    print(last_positive,last_submit)
    Submit_train=np.array([[ (__+_)*2**(_-nums) for _ in range(nums)] for __ in last_submit[:,0]])
    Positive_train=np.array([[(__+_)*2**(_-nums) for _ in range(nums)] for __ in last_positive[:,0]])
    Submit_target=np.array(last_submit[:,1])
    Positive_target=np.array(last_positive[:,1])
    return Submit_train,Submit_target,Positive_train,Positive_target
"""
train processing
"""
def time_SVR(X_train,y_train,o):
    torch.cuda.empty_cache()
    index = importlib.import_module("getmodule")
    print(index)
    fig = index.__dict__["Config"]()#parameter setting
    model = index.__dict__["Model"](config=fig)  ##model
    X_train=torch.tensor(X_train).to(torch.float)
    y_train=torch.tensor(y_train).to(torch.float)
    length = X_train.shape[0]
    trainx_loader = (X_train-X_train.mean(dim=0))/X_train.std(dim=0)
    trainy_loader =(y_train-y_train.mean(dim=0))/y_train.std(dim=0)
    a=y_train.mean(dim=0)
    b=y_train.std(dim=0)
    loss_func = nn.MSELoss()##Mean square error
    model.to(fig.devices)
    optimizer = torch.optim.Adam(model.parameters(), lr=fig.learn_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=10,
                                                gamma=0.5)
    "==================================train==================================="
    for epoch in range(fig.num_epochs):
        data = trainx_loader
        label = trainy_loader
        data = torch.tensor(data).to(fig.devices)
        label = torch.tensor(label).to(fig.devices).to(torch.float)
        optimizer.zero_grad()
        pred = model.forward(data).squeeze(1)
        loss_val = loss_func(pred, label).to(torch.float)
        loss_val.backward()
        optimizer.step()
        scheduler.step()
    index = torch.randint(low=0, high=trainy_loader.shape[0], size=[trainy_loader.shape[0]]).long()
    trainx_loader = trainx_loader.to(fig.devices)
    trainy_loader = trainy_loader.to(fig.devices).to(torch.float)
    optimizer.zero_grad()
    pred = model.forward(trainx_loader).squeeze(1).detach().clone().cpu().numpy()
    trainy_loader=trainy_loader.detach().clone().cpu().numpy()
    print(pred,trainy_loader)
    t = np.linspace(0, pred.shape[0], pred.shape[0] + 1).tolist()
    pio.templates.default = "simple_white"
    figure1 = go.Scatter(x=t, y=pred, mode='lines+markers', name='pred', marker=dict(
        size=2,
        color="red",
        showscale=True
    ), line=dict(
        color='orange',
        width=2
    )
                         )
    figure2 = go.Scatter(x=t, y=trainy_loader, mode='lines+markers', name='target', marker=dict(
        size=2,
        color="blue",
        showscale=True
    ), line=dict(
        color='#00FFFF',
        width=2
    )
                         )

    data = [figure1, figure2]
    layout = dict(
        title="ProblemV",
        xaxis_title="X:index",
        yaxis_title="Y:data",
        legend_title="Legend",
        font=dict(
            family="Segoe UI Black",
            size=16,
            color="black"
        ),
        xaxis=dict(showline=True, showgrid=False, side='bottom', linecolor='black', gridcolor='white'),
        yaxis=dict(showline=True, showgrid=False, gridcolor='white', linecolor="black", side="left"
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
    torch.save(model.state_dict(), "five_model{}.pth".format(o))
    return a,b
"""
call model
"""
def Train_solve(path,len,nums):
    s_x,s_y,p_x,p_y=imfotmation_get(path,len,nums)
    a,b=time_SVR(s_x,s_y,1)
    c,d=time_SVR(p_x,p_y,2)
    last_solve(path,len,nums,a,b,c,d)
"""
According to the prediction, the ProblemV results are obtained
"""
def last_solve(path,len,nums,a,b,c,d):
    s_x, s_y, p_x, p_y = imfotmation_get(path, len, nums)
    torch.cuda.empty_cache()
    index = importlib.import_module("2021MCM_ProblemC_Code.Problem5.getmodule")
    fig = index.__dict__["Config"]()
    model = index.__dict__["Model"](config=fig)
    model.load_state_dict(torch.load("five_model1.pth"))
    with torch.no_grad():
        trainx_loader=torch.tensor([[(__+_)*2**(_-10) for __ in range(10)] for _ in np.linspace(s_x[-1][0],s_x[-1][0]+64,65)])
        pred1=((model.forward(trainx_loader)*b+a).squeeze(1).detach().clone().cpu().numpy())

    model.load_state_dict(torch.load("five_model2.pth"))
    with torch.no_grad():
        trainx_loader=torch.tensor([[(__+_)*2**(_-10) for __ in range(10)] for _ in np.linspace(s_x[-1][0],s_x[-1][0]+64,65)])
        pred2=((model.forward(trainx_loader)*d+c).squeeze(1).detach().clone().cpu().numpy())
    t=np.linspace(0,pred1.shape[0],pred1.shape[0]+1)
    pio.templates.default="simple_white"
    trace1 = go.Scatter(
        name="number of vespa mandarinia's reports",
        x=t, y=pred2,
        marker=dict(
            size=8 + np.random.randn(1000),
            color="orange",
            showscale=True,
            line=dict(
                width=10,  # 线条大小
                color="rgba(1, 170, 118, 0.3)"  # 线条的颜色
            )
        )
        , textposition='bottom right', mode='markers+lines'
    )
    trace2 = go.Scatter(
        name="number of all reports",
        x=t, y=pred1,
        marker=dict(
            size=8 + np.random.randn(1000),
            color="blue",
            showscale=True,
            line=dict(
                width=10,
                color="#00FFFF"
            )
        )
        , textposition='bottom right', mode='markers+lines'
    )
    layout = dict(
        title="times-reports",
        showlegend=True,
        xaxis_title="X:time",
        yaxis_title="Y:number of reports",
        legend_title="Legend",
        legend=dict(
            x=0.9,
            y=1.1
        ),

        xaxis=dict(showline=False, side='bottom', linecolor='white', gridcolor='white'),
        yaxis=dict(showline=False, gridcolor='white', type="linear"
                   )
    )
    data = [trace1,trace2]
    fig = Figure(data=data, layout=layout)
    fig.show()

if __name__=="__main__":
    Train_solve("../Others/data/Five_data.csv", 10, 10)