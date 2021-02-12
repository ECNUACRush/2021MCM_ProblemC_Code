import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from math import radians, cos, sin, asin, sqrt
import json
import plotly.graph_objects as go
import plotly as py
import chart_studio
import plotly.io as io
import statsmodels.stats.weightstats as Z
pyplt = py.offline.plot
p=io.renderers['png']
p.width=800
p.height=600
from plotly.graph_objs import *
chart_studio.tools.set_config_file(world_readable=True,
                             sharing='public')
colors = ['navy', 'turquoise', 'darkorange']
target_names=["Negative","Positive","Unverified"]
list=[]
'''
hypothesis test
'''
def Test(array,value):
    weightstatsz, pval = Z.ztest(array, value=value)
    if abs(weightstatsz)<=0.05:
        print("{}<5%,accept".format(abs(weightstatsz)))
        return 1
    else:
        print("{}>=5%,refuse".format(abs(weightstatsz)))
        return 0
'''
Longitude latitude conversion kilometer
'''
def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # Average radius of the earth, 6371km
    distance=round(distance/1000,3)
    return distance
"""
Draw ellipse
"""
def make_ellipses(gmm,nums):
    covariances = gmm.covariances_[0][:2, :2]
    v, w = np.linalg.eigh(covariances)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    v[0]=1 if v[0]<1 else v[0]
    v[0] =5 if v[0]>5 else v[0]
    v[1]=1 if v[1]<1 else v[1]
    v[1] =5 if v[1]>5 else v[1]
    ell = mpl.patches.Ellipse(gmm.means_[0, :2], v[0], v[1],
                                  180 + angle, color=colors[nums])
    ell.set_alpha(0.5)
    return ell
"""
Gaussian mixture model
"""
def MyGaussianMixture(begin,end,iinput,nums,mystr):
    dir_path= "../Others/data/First_data.csv"
    ff = pd.read_csv(dir_path, sep=',', index_col=False,
                     encoding="utf-8", low_memory=False) ##Read file
    list_train=[]
    list_target=[]
    max_x=-200
    min_x=200
    max_y=-200
    min_y=200
    for item in ff.index:
        if begin>ff.iloc[item]["Detection Date"] or end<ff.iloc[item]["Detection Date"]:
            continue
        else:
            if float(ff.iloc[item]["Longitude"])>max_x:
                max_x=float(ff.iloc[item]["Longitude"])
            if float(ff.iloc[item]["Latitude"]) > max_y:
                max_y = float(ff.iloc[item]["Latitude"])
            if float(ff.iloc[item]["Longitude"])<min_x:
                min_x=float(ff.iloc[item]["Longitude"])
            if float(ff.iloc[item]["Latitude"]) <min_y:
                min_y = float(ff.iloc[item]["Latitude"])
            list_train.append([float(ff.iloc[item]["Longitude"]),float(ff.iloc[item]["Latitude"])])
            list_target.append(int(ff.iloc[item]["Lab Status"]))
    skf = StratifiedKFold(n_splits=2,random_state=0,shuffle=True)
    train=np.array(list_train)
    target=np.array(list_target)
    my_x_ticks = np.linspace(min_x,max_x,5)
    my_y_ticks = np.arange(min_y,max_y,5)
    train_index, test_index = next(iter(skf.split(train,target)))
    X_train = train[train_index]
    y_train = target[train_index]
    X_test = train[test_index]
    y_test =target[test_index]
    n_classes =np.unique(y_train)
    # Try GMMs using different types of covariances.
    fig=plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    list_return=[]
    for index,label in enumerate(n_classes):
        estimators = GaussianMixture(n_components=1,
                                     covariance_type="full", max_iter=100, random_state=0)
        estimators.means_init = np.array([X_train[y_train ==label].mean(axis=0)])
        m=X_train[y_train ==label]
        if m.shape[0]==1:
            m=np.concatenate((m, m), axis=0)
        estimators.fit(m)
        ax.add_patch(make_ellipses(estimators,label))
        if iinput==label:
            list_return.append(nums)
            list_return.append(estimators.means_[0][0])
            list_return.append(estimators.means_[0][1])
            list_return.append(estimators.covariances_[0][0][0])
            list_return.append(estimators.covariances_[0][1][1])
            Test(m[:,0],estimators.means_[0][0])
            Test(m[:,1], estimators.means_[0][1])
        data = train[target == label]
        plt.scatter(data[:, 0], data[:, 1], s=0.8, color=colors[label],
                    label=target_names[label])
        data = X_test[y_test ==label]
        plt.scatter(data[:, 0], data[:, 1], marker='x', color=colors[label])
    plt.title(mystr)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.xlim((-124.665014-0.5,-116.87368700000002+0.5))
    plt.ylim((45.488689-0.5, 49.548004+0.5))
    print(min_x,max_x,min_y,max_y)
    plt.legend(scatterpoints=1, loc='best', prop=dict(size=12))
    plt.show()
    return list_return
"""
Call Gaussian mixture model
"""
def train_timepred(begin,end,nums):
    leng=(end-begin)/nums
    list_last=[]
    time=["2019/01/20","2019/06/08","2019/10/26","2020/03/15","2020/07/23","2020/12/20"]
    for i in range(nums):
        list_last.append(MyGaussianMixture(begin+i*leng,begin+(i+1)*leng,1,i,"begin:"+time[2]+" ~ "+"end:"+time[5]))
    save = pd.DataFrame(list_last, columns=["day","x_mean", "y_mean","x_std","y_std"])
    save.to_csv('../Others/data/positive.csv', index=False, header=True)
"""
SVR model
"""
def SVR_logistic(path,nums,choose):
    ff = pd.read_csv(path, sep=',', index_col=False,
                     encoding="utf-8", low_memory=False) ##Read file
    y_train=[]
    X_train=[]
    for item in ff.index:
        time=int(ff.iloc[item - 1]["day"])
        y_train.append([ff.iloc[item - 1]["x_mean"],ff.iloc[item - 1]["y_mean"],ff.iloc[item - 1]["x_std"],ff.iloc[item - 1]["y_std"]])
        X_train.append([_+time for _ in range(nums)])
    y_train=np.array(y_train)
    X_train=np.array(X_train)
    random_seed=13
    X_train,y_train=shuffle(X_train,y_train,random_state=random_seed)
    parameters={'kernel':['rbf'],'gamma':np.logspace(-5,0,num=6,base=2.0),'C':np.logspace(-5,5,num=11,base=2.0)}
    svr=SVR()
    grid_search=GridSearchCV(svr,parameters,cv=5,n_jobs=4,scoring="neg_mean_squared_error")
    y_train=y_train[:,choose]
    grid_search.fit(X_train,y_train)
    y_pred=grid_search.predict(X_train)
    sum=0
    Test(y_train,np.mean(y_pred))
    abs_vals=np.mean(np.abs(y_train-y_pred)*np.abs(y_train-y_pred))
    print(abs_vals)
    X_newtrain=np.array([[__+_ for _ in range(nums)] for __ in np.linspace(0,15,16)])
    return X_train[:,0],X_newtrain[:,0],y_train,grid_search.predict(X_newtrain)
"""
Time series visualization scatter diagram
"""
def Paint(path):
    x_index,y_index2,x_mean,x_mean2=SVR_logistic(path,50,0)
    y_index,y_index2,y_mean,y_mean2 = SVR_logistic(path,50,1)
    _,__,x_std,x_std2=SVR_logistic(path, 50,2)
    _,__,y_std,y_std2=SVR_logistic(path, 50,3)
    ff=open(r"../Others/data/mean", "w")
    print(x_mean2,y_mean2)
    ff.writelines("max{},{}min{},{}\n".format(np.max(x_mean2[10:]),np.max(y_mean2[10:]),np.min(x_mean2[10:]),np.min(y_mean2[10:])))
    ff.writelines("mean{},{}\n".format(np.mean(x_mean2[10:]),np.mean(y_mean2[10:])))
    ff.close()
    trace1=go.Scattermapbox(
        name='Canada-Distributuin of wasps-train',
        lat=y_mean,
        lon=x_mean,
        mode='markers',
        showlegend=True,
        marker=go.scattermapbox.Marker(
            size=5,
            opacity=0.8,
            symbol='circle',
            color="#ADFF2F",
        ),  text=x_index,
        textfont=dict(size=18),
    )
    trace2 = go.Scattermapbox(
        name='Canada-Distributuin of wasps-pred',
        lat=y_mean2,
        lon=x_mean2,
        mode='lines+markers',
        showlegend=True,
        line=go.scattermapbox.Line(
            width=2,
            color="red"
        ),
        textfont=dict(size=18),
    )
    layout=go.Layout(
        hovermode='closest',
        mapbox=dict(
            accesstoken="pk.eyJ1Ijoic3N0MTIzNDU2IiwiYSI6ImNra3Q1cHRibzE3dDAycHFrd2h0Z29ibHcifQ.DD0ly4kXGknJMTDbdw5cUg",
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=np.mean(np.append(y_mean,y_mean2)),
                lon=np.mean(np.append(x_mean,x_mean2))
            ),
            pitch=0,
            zoom=3,
            style='satellite'  #The map types displayed include remote sensing map, street map, etc
        ))
    data = [trace2,trace1]
    fig = Figure(data=data, layout=layout)
    fig.show()
"""
Drawing scatter map with lab status as label
"""
def Paint2():
    ff = pd.read_csv("../Others/data/DataSet.csv", sep=',', index_col=False,
                     encoding="utf-8", low_memory=False)  ##Read file
    with open(r'../Others/data/CAN.geo.json', encoding='utf-8')as f:
        canada_geo = json.load(f)
    Negative=[]
    Positive=[]
    Univerified=[]
    for item in ff.index:
        Negative.append([ff.iloc[item - 1]["Latitude"],ff.iloc[item - 1]["Longitude"]]) if ff.iloc[item-1]["Lab Status"]=="Negative ID" else Negative
        Positive.append([ff.iloc[item - 1]["Latitude"], ff.iloc[item - 1]["Longitude"]]) if ff.iloc[item - 1][
                                                                                                "Lab Status"] == "Positive ID" else Positive
        Univerified.append([ff.iloc[item - 1]["Latitude"], ff.iloc[item - 1]["Longitude"]]) if ff.iloc[item - 1][
                                                                                            "Lab Status"] == "Unverified" else Univerified
    Negative=np.array(Negative)
    Positive=np.array(Positive)
    Univerified=np.array(Univerified)
    trace1=go.Scattermapbox(
        name='Negative',
        lat=Negative[:,0],
        lon=Negative[:,1],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=5,
            opacity=0.8,
            symbol='circle',
            color="#ADFF2F",
        ),
        textfont=dict(size=18),
    )
    trace2 = go.Scattermapbox(
        name='Positive',
        lat=Positive[:, 0],
        lon=Positive[:, 1],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=8,
            opacity=0.8,
            symbol='circle',
            color="red",
        ),
        textfont=dict(size=18),
    )
    trace3 = go.Scattermapbox(
        name='Unverified',
        lat=Univerified[:, 0],
        lon=Univerified[:, 1],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=5,
            opacity=0.8,
            symbol='circle',
            color="#0099FF",
        ),
        textfont=dict(size=18),
    )
    layout=go.Layout(
        hovermode='closest',
        mapbox=dict(
            accesstoken="pk.eyJ1Ijoic3N0MTIzNDU2IiwiYSI6ImNra3Q1cHRibzE3dDAycHFrd2h0Z29ibHcifQ.DD0ly4kXGknJMTDbdw5cUg",
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=np.mean(np.r_[np.r_[Positive[:,0],Univerified[:,0]],Negative[:,0]]),
                lon=np.mean(np.r_[np.r_[Positive[:,1],Univerified[:,1]],Negative[:,1]])),
            pitch=0,
            zoom=3,
            style='open-street-map'

        )
        , legend=dict(
            x=0.1,
            y=0.9,
            bgcolor="white",
            bordercolor="black",
            borderwidth=2,
            font=dict(size=18, color="black"))
    )
    data = [trace1,trace2,trace3]
    fig = Figure(data=data,layout=layout)
    fig.show()
"""
ProblemI result visualization
"""
def Paint3(batch):
    max_p=[-122.41035789222359, 48.91263276070714]##positive
    min_p=[-122.56763755431301, 48.823154396376424]
    p=[-122.4889977232718, 48.867893578572634]
    max_n =[- 121.88532010065887, 47.56590564750796]
    min_n=[ - 121.88532010065887, 47.55280492757116]
    n= [- 121.88532010065887, 47.56370207715077]
    max_u =[- 121.86298380226917, 47.62707605887937]
    min_u=[ - 121.87697661910784, 47.62707605887937]
    u=[ - 121.8746222877882, 47.62707605887937]
    min_o=[-123.9431,48.7775]
    max_o=[-122.4186,49.1494]
    last_mean=[-122.58246499999998,48.983375]
    t = np.linspace(0, 2*np.pi, 240)
    lat_l=[]
    lon_l=[]
    for i in t:
        lat_l.append(last_mean[1]+30*sin(i)/111)
        lon_l.append(last_mean[0]+30*cos(i)/111)

    lat_p=np.linspace(min_p[0],max_p[0],batch)
    lon_p=np.linspace(min_p[1],max_p[1],batch)
    lat_pp=np.zeros(batch*batch)
    lon_pp=np.zeros(batch*batch)
    for i in range(batch):
        for j in range(batch):
            lon_pp[i*batch+j]=lat_p[i]
            lat_pp[i*batch+j]=lon_p[j]
    trace1=go.Scattermapbox(
        name='Positive_pre',
        lat=lat_pp,
        lon=lon_pp,
        mode='markers',
        textposition ="top center",
        text=11,
        marker=go.scattermapbox.Marker(
            size=1,
            opacity=1,
            symbol='circle',
            color="red",
        ),
        textfont=dict(size=18),
    )
    lat_u = np.linspace(min_u[0], max_u[0], batch)
    lon_u = np.linspace(min_u[1], max_u[1], batch)
    lat_uu = np.zeros(batch * batch)
    lon_uu = np.zeros(batch * batch)
    for i in range(batch):
        for j in range(batch):
            lon_uu[i * batch + j] = lat_u[i]
            lat_uu[i * batch + j] = lon_u[j]
    trace2 = go.Scattermapbox(
        name='Unverified_pre',
        lat=lat_uu,
        lon=lon_uu,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=5,
            opacity=1,
            symbol='circle',
            color="#0099FF",
        ),
        textfont=dict(size=18),
    )
    lat_n = np.linspace(min_n[0], max_n[0], batch)
    lon_n = np.linspace(min_n[1], max_n[1], batch)
    lat_nn = np.zeros(batch * batch)
    lon_nn = np.zeros(batch * batch)
    for i in range(batch):
        for j in range(batch):
            lon_nn[i * batch + j] = lat_n[i]
            lat_nn[i * batch + j] = lon_n[j]

    trace3 = go.Scattermapbox(
        name='Negative_pre',
        lat=lat_nn,
        lon=lon_nn,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=5,
            opacity=1,
            symbol='circle',
            color="yellow",
        ),
        textfont=dict(size=18),
    )
    lat_o = np.linspace(min_o[0], max_o[0], batch)
    lon_o = np.linspace(min_o[1], max_o[1], batch)
    lat_oo = np.zeros(batch *4)
    lon_oo = np.zeros(batch *4)
    for i in range(batch):
        lat_oo[i]=min_o[1]
        lon_oo[i]=lat_o[i]
    for i in range(batch):
        lat_oo[i+batch]=max_o[1]
        lon_oo[i+batch]=lat_o[i]
    for i in range(batch):
        lat_oo[i+batch*2]=lon_o[i]
        lon_oo[i+batch*2]=min_o[0]
    for i in range(batch):
        lat_oo[i+batch*3]=lon_o[i]
        lon_oo[i+batch*3]=max_o[0]
    trace4 = go.Scattermapbox(
        name='Positive_old',
        lat=lat_oo,
        lon=lon_oo,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=3,
            opacity=1,
            symbol='circle',
            color="#708090",
        ),
        textfont=dict(size=18),
    )
    trace5 = go.Scattermapbox(
        name='limit',
        lat=lat_l,
        lon=lon_l,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=3,
            opacity=1,
            symbol='circle',
            color="#00FFFF",
        ),
        textfont=dict(size=18),
    )
    data = [trace4,trace1,trace2,trace3,trace5]
    layout2=go.Layout(
        hovermode='closest',
        mapbox=dict(
            accesstoken="pk.eyJ1Ijoic3N0MTIzNDU2IiwiYSI6ImNra3Q1cHRibzE3dDAycHFrd2h0Z29ibHcifQ.DD0ly4kXGknJMTDbdw5cUg",
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=(n[1]+p[1]+u[1])/3,
                lon=(n[0]+p[0]+u[0])/3),
            pitch=0,
            zoom=3,
            style='open-street-map'

        )
        , legend=dict(
            x=0.1,
            y=0.9,
            bgcolor="white",
            bordercolor="black",
            borderwidth=2,
    font = dict(size=18,color="black")
        )
    )
    fig = Figure(data=data, layout=layout2)
    fig.show()
if __name__=="__main__":
    Paint2()