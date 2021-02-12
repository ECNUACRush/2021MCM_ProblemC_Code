import pandas as pd
import os
import openpyxl
def Data_filer():
    data_path1 ="../Others/data/DataSet.xlsx"
    list_set=[]
    ff = pd.read_excel(data_path1, engine='openpyxl',header=0, index_col=False,
                         encoding="utf-8")
    for item in ff.index:
        if ff.iloc[item - 1]["Lab Status"]=="Unprocessed":
            continue
        if str(ff.iloc[item - 1]["Detection Date"])[0:4] != "2019" and str(ff.iloc[item - 1]["Detection Date"])[0:4] != "2020":
            print(str(ff.iloc[item - 1]["Detection Date"])[0:4])
            continue
        list_set.append([ff.iloc[item-1]["GlobalID"],ff.iloc[item-1]["Detection Date"],ff.iloc[item-1]["Lab Status"]
                                 ,ff.iloc[item-1]["Submission Date"],ff.iloc[item-1]["Latitude"],ff.iloc[item-1]["Longitude"]
                              ,ff.iloc[item-1]["Notes"]])
    save = pd.DataFrame(list_set, columns=["GlobalID", "Detection Date", "Lab Status", "Submission Date", "Latitude",
                                           "Longitude","Notes"])
    save.to_csv('../Others/data/DataSet.csv', index=False, header=True)
def Data_IDchoose():
    data_path1 = "../Others/data/DataSet.csv"
    data_path2 = "../Others/data/2021MCM_ProblemC_ Images_by_GlobalID.xlsx"
    list_set = []
    ff = pd.read_csv(data_path1, sep=',', index_col=False,
                     encoding="utf-8", low_memory=False)
    for item in ff.index:
        list_set.append(ff.iloc[item - 1]["GlobalID"])
    gg=pd.read_excel(data_path2, engine='openpyxl',header=0, index_col=False,
                         encoding="utf-8")
    list_globalID=[]
    for item in gg.index:
        if gg.iloc[item-1]["GlobalID"] in list_set:
            list_globalID.append([gg.iloc[item-1]["GlobalID"],gg.iloc[item-1]["FileName"]])
    save = pd.DataFrame(list_globalID, columns=["GlobalID", "FileName"])
    save.to_csv('../Others/data/Dataglobalid.csv', index=False, header=True)

def del_files(path):
    data_path1 = "../Others/data/Dataglobalid.csv"
    ff = pd.read_csv(data_path1, sep=',', index_col=False,
                     encoding="utf-8", low_memory=False)
    list_filename=[]
    for item in ff.index:
        list_filename.append(ff.iloc[item - 1]["FileName"])
    for root,dirs,files in os.walk(path):
        for name in files:
            if name not in list_filename:
                os.remove(os.path.join(root,name))
                print('Delete files:',os.path.join(root,name))

def merge():
  file1 = '../Others/data/DataSet.csv'
  file2 = '../Others/data/Dataglobalid.csv'
  df1 = pd.read_csv(file1,sep=',',
                     encoding="utf-8")
  df2 = pd.read_csv(file2, sep=',',
                     encoding="utf-8")
  print(df1,df2)
  df_Merge=pd.merge(df1, df2)
  df_Merge.to_csv('../Others/data/merge.csv')

if __name__=="__main__":
    merge()