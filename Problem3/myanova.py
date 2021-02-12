import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
"""
ProblemIII file variance analysis
"""
def myanova(pdinput):
    model = ols('value~C(symbol) + C(count)', data=pdinput[["symbol","count","value"]]).fit()
    '''
    There is no interaction between a and B
    '''
    anovat = anova_lm(model)
    model2 = ols('value~C(symbol) + C(count)+C(symbol):C(count)', data=pdinput[["symbol","count","value"]]).fit()
    '''
  There is interaction between a and B
    '''
    anovat2 = anova_lm(model2)
    print(anovat)
    print(anovat2)
def pd_dataprocessing():
    ff = pd.read_excel("../Others/data/merge.xlsx", engine='openpyxl',header=0, index_col=False,
                         encoding="utf-8")  ##read file
    list_tap=[]
    count={}
    for item in ff.index:
        if str(ff.iloc[item - 1]["GlobalID"]) not in count.keys():
            count[str(ff.iloc[item - 1]["GlobalID"])]=1
        else:
            count[str(ff.iloc[item - 1]["GlobalID"])]+=1
    for item in ff.index:
        v=count[str(ff.iloc[item - 1]["GlobalID"])]
        m=str(ff.iloc[item - 1]["FileName"])[-3:]
        if m=="pdf":
            k=0
        elif m=="jpg" or m=="png":
            k=1
        elif m=="mov" or m=="mp4":
            k=2
        else:
            k=3
        if v!=1:
            k=3
        if str(ff.iloc[item - 1]["Lab Status"])=="Positive ID":
            a=1
        elif str(ff.iloc[item - 1]["Lab Status"])=="Negative ID":
            a=0
        else:
            a=2
        list_tap.append([k,v,a])
    return pd.DataFrame(list_tap
        ,columns=["symbol","count","value"])
if __name__=="__main__":
    myanova(pd_dataprocessing())