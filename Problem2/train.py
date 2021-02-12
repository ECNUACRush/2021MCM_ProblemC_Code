import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import importlib
import plotly as py
import chart_studio
import plotly.io as io
import plotly.graph_objects as go
from plotly.graph_objs import *
import torch
import plotly.io as pio
'''=======================================================train============================================================='''
torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description='Weight Experiments')
parser.add_argument('--dataset', dest='dataset', help='training dataset', default='Dataallloader', type=str)
parser.add_argument('--net', dest='net', help='training network', default='Model', type=str)
parser.add_argument('--parameter', dest='parameter',default="Config", type=str)
parser.add_argument('--len_dict', dest='len_dict',default=2102, type=int)
parser.add_argument('--train_batch_size', dest='train_batch_size', help='training batch size', default=50, type=int)
"""=======================================================main function============================================================="""
args = parser.parse_args()
if __name__ == "__main__":
    pio.templates.default = "none"
    pyplt = py.offline.plot
    p = io.renderers['png']
    p.width = 800
    p.height = 600
    chart_studio.tools.set_config_file(world_readable=True, sharing='public')
    All = importlib.import_module("2021MCM_ProblemC_Code.Problem2.myimportlib")
    parameter=All.__dict__[args.parameter](n_vocab=args.len_dict)##参数
    model = All.__dict__[args.net](parameter)  ##网络
    train_loader = All.__dict__[args.dataset](args.train_batch_size)
    model.to(parameter.devices)
    loss_func = torch.nn.CrossEntropyLoss()##Cross entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=parameter.learn_rate)##Optimizer settings
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)##Ways to reduce learning rate
    list_loss=[]
    for epoch in range(parameter.num_epochs):
        loss_avg = 0
        for i,(data,target) in enumerate(train_loader):
            model.len_list =data[:,-1]
            data=data[:,:-1]
            data = torch.tensor(data).to(parameter.devices)
            label = torch.tensor(target, dtype=torch.int64).to(parameter.devices)
            optimizer.zero_grad()
            pred = model.forward(data)##forward
            loss_val = loss_func(pred, label)
            loss_avg += float(loss_val)
            loss_val.backward()
            optimizer.step()
        loss_avg = loss_avg / len(train_loader)
        list_loss.append(loss_avg)
        print("epoch is {}, val is {}".format(epoch, loss_avg))
        scheduler.step()##Adjust learning rate
    t=np.linspace(0,len(list_loss),len(list_loss)+1).tolist()
    figure1 = go.Scatter(x=t,y=list_loss,mode = 'lines+markers',name = 'label',  marker = dict(
        size = 2,
        color="red",
        showscale = True
    ),  line = dict(
        color = 'orange',
        width = 1
    )
)

    data = [figure1]
    layout = dict(
    title="len_dict:2100",
    xaxis_title="X:epoch",
    yaxis_title="Y:loss",
    legend_title="Legend",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    ),
        xaxis=dict(showline=True, showgrid=True,side='bottom', linecolor='black', gridcolor='white'),
        yaxis=dict(showline=False, showgrid=True,gridcolor='white',linecolor="black",side="left")
    )
    fig = Figure(data=data,layout=layout)
    fig.show()

    torch.save(model.state_dict(), "model{}.pth".format(parameter.num_epochs))