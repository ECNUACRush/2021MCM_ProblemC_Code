import pandas as pd

import os
def Alignment():
    list_name=[]
    file1 = '../Others/data/merge.csv'
    for root,dirs,files in os.walk("../Others/data/image"):#(use os.walk This method returns a ternary tuple (dirpath (string), dirnames (list), filenames (list)), where the first is the starting path, the second is the folder under the starting path, and the third is the file under the starting path.)
        for name in files:
            list_name.append(name)
    ff = pd.read_csv(file1, sep=',',
                      encoding="utf-8")
    dict_nametovar={}
    for item in ff.index:
        if ff.iloc[item - 1]["Lab Status"]=="Positive ID":
            var=1
        else:
            var=0
        dict_nametovar[ff.iloc[item - 1]["FileName"]]=var
    list_var=[]
    for name in list_name:
        list_var.append(dict_nametovar[name])
    ff = open("../Others/data/var", "w")
    for item in list_var:
        ff.writelines("{}\n".format(item))
    ff.close()


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
    days=[-31,0,28,59,89,120,150,181,212,242,273,303]##Get the day difference from the month difference
    list_all=[]
    for item in range(len(ff.index)):
        if ff.iloc[item]["Lab Status"]=="Positive ID":
            var=1
        elif ff.iloc[item]["Lab Status"]=="Negative ID":
            var=0
        elif ff.iloc[item]["Lab Status"]=="Unverified":
            var=2
        else:
            continue
        temp=str(ff.iloc[item]["Detection Date"])
        years_temp=int(temp[0:4])
        month_temp = int(temp[5:7])
        day_temp = int(temp[8:10])

        time=(years_temp-years_begin)*365+days[month_temp-month_begin]+(32-day_begin)+day_temp
        print(time,temp)
        list_all.append([time,ff.iloc[item]["Latitude"],ff.iloc[item]["Longitude"],ff.iloc[item]["Notes"],var])
    save = pd.DataFrame(list_all, columns=["Detection Date", "Latitude", "Longitude","Notes", "Lab Status"])
    save.to_csv('../Others/data/Second_data.csv', index=False, header=True)
if __name__=="__main__":
    allmerge_sort("../Others/data/DataSet.csv")