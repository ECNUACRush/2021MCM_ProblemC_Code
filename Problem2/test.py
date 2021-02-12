import numpy as np
import plotly as py
import chart_studio
import plotly.io as io
import plotly.graph_objects as go
from plotly.graph_objs import *
import torch
import argparse
import importlib
import plotly.io as pio
"""
ProblemII test processing
"""
'''=======================================================Train============================================================='''
torch.cuda.empty_cache()##cuda clear
parser = argparse.ArgumentParser(description='Weight Experiments')##Linux command line
parser.add_argument('--dataset', dest='dataset', help='training dataset', default='Dataallloader', type=str)
parser.add_argument('--net', dest='net', help='training network', default='Model', type=str)
parser.add_argument('--parameter', dest='parameter',default="Config", type=str)
parser.add_argument('--len_dict', dest='len_dict',default=2502, type=int)
parser.add_argument('--train_batch_size', dest='train_batch_size', help='training batch size', default=100, type=int)
args = parser.parse_args()
'''=============================================Test model accuracy==========================================================='''
if __name__=="__main__":
    pio.templates.default = "simple_white"
    All = importlib.import_module("2021MCM_ProblemC_Code.Problem2.myimportlib")
    parameter=All.__dict__[args.parameter](n_vocab=args.len_dict)##parameter
    model = All.__dict__[args.net](parameter)  ##model
    train_loader = All.__dict__[args.dataset](args.train_batch_size)
    model.to(parameter.devices)
    pyplt = py.offline.plot
    p=io.renderers['png']
    p.width=800
    p.height=600
    chart_studio.tools.set_config_file(world_readable=True,sharing='public')
    list_dist=[]
    model.load_state_dict(torch.load("model100.pth"))##import model100.pth
    print(train_loader)
    with torch.no_grad():
        for i, (data,target) in enumerate(train_loader):
            if i>5:
                break
            model.len_list = data[:, -1]
            data = data[:, :-1]
            data = torch.tensor(data).to(parameter.devices)
            label = torch.tensor(target, dtype=torch.int64).to(parameter.devices)
            pred_softmax = model.forward(data)
            pred = torch.argmax(pred_softmax, dim=1)
            nplist=torch.abs(pred-label).detach().cpu().tolist()
            for i,o in enumerate(nplist):
                if o!=0:
                    nplist[i]=float(torch.max(pred_softmax[i]))
            list_dist+=nplist
    N=len(list_dist)
    count=0
    for item in list_dist:
        if item ==0:
            count+=1
    print("pre_accuracy:{}".format(count/N))
    print("fc_weight:",model.fc2.weight.data)
    t=np.linspace(0,40,N).tolist()
    figure1 = go.Scatter(x=t,y=list_dist,mode = 'markers',name = 'label',
    marker = dict(
        size = 8,
        color=np.random.randn(N),
        colorscale = 'blues',
        showscale = True
    ))

    data = [figure1]
    layout = dict(
    title="Pred——dist",
    xaxis_title="X:batch",
    yaxis_title="Y:target",
    legend_title="Legend Title",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    ),
    xaxis_title_font_color='black',
    yaxis_title_font_color = 'black',
    )
    fig = Figure(data=data,layout=layout)
    fig.show()


