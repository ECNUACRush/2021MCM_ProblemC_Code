import torchvision
import pandas as pd
from torchvision import datasets, transforms
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from ..Problem2 import dataprocessing.getdata
"""
Complete the encapsulation of datloader
"""
class Mydataset(Dataset):
    def __init__(self):
        super().__init__()
        data,var ,len_seq=getdata()
        list_data=[]
        ff = pd.read_csv("../Others/data/Second_data.csv", sep=',')
        for item in ff.index:
            list_data.append([ff.iloc[item-1]["Detection Date"]
                                 ,ff.iloc[item-1]["Latitude"]
                              ,ff.iloc[item-1]["Longitude"]])
        list_data=torch.tensor(list_data)
        self.data=torch.cat((torch.cat((list_data,data),dim=1),len_seq),dim=1)
        self.var=var.squeeze(1)
        self.len_seq=len_seq
    def __len__(self):
        return self.var.shape[0]
    def __getitem__(self,idex):
        label=self.var[idex]
        data=self.data[idex]
        return data,label
def Dataallloader(batchsize):
    nlp_datasets = Mydataset()
    dataloaders =DataLoader(nlp_datasets, batch_size=batchsize,
                                               shuffle=True, num_workers=2)
    return dataloaders
