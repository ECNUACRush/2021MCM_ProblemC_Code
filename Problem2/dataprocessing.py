import pandas as pd
import nltk
import numpy as np
import torch
"""
NLP data processing
"""
def getdata():
    '''==================================stopwords==================================='''
    data_stop_path= "../Others/stop_words_eng.txt"
    stops_word = open(data_stop_path,encoding="utf-8").readlines()
    stops_word = [line.strip() for line in stops_word]
    stops_word.append(" ")
    '''add" "和"/n"stopwords'''
    stops_word.append("\n")
    voc_dict = {}
    min_seq = 1
    top_n = 2100
    UNK = "<UNK>"
    PAD = "<PAD>"
    N="N"
    Y="Y"
    '''==================================process data==================================='''
    data_path = ["../Others/data/Second_data.csv"]##File list
    pddata=[]
    NPtransform=[]
    target=[]
    max_len_seq=0#Record the longest sentence length
    for item in data_path:
        ff = pd.read_csv(item, sep=',', header=0, index_col=False,
                     encoding="utf-8", low_memory=False)##Read file
        pddata.append(ff)
    for items in pddata:
        for item in items.index:
            j=int(items.iloc[item-1]["Lab Status"])
            target.append(j)
            seg_list=[]
            if(items.iloc[item - 1]["Notes"]==items.iloc[item - 1]["Notes"]):##Determine whether it is Nan data
                A=nltk.word_tokenize(items.iloc[item - 1]["Notes"])## use nltk
            else:
                A=[]
            seg_list+=A
            now=seg_list##The final eigenvector, which has not yet been converted to a number
            NPtransform.append(now)
            if len(seg_list) > max_len_seq:
                max_len_seq = len(seg_list)
            for seg_item in seg_list:
                if seg_item in stops_word:##Remove stop words
                    continue
                if seg_item in voc_dict.keys():
                    voc_dict[seg_item]=voc_dict[seg_item]+1
                else:
                    voc_dict[seg_item]=1##add to dict
    voc_list=sorted([_ for _ in voc_dict.items() if _[1]>min_seq],key=lambda x:x[1],reverse=True)[:top_n]##sort
    voc_dict={word_count[0]:idx+2 for idx,word_count in enumerate(voc_list)}##become dict
    voc_dict.update({UNK:1,PAD:0})
    print(voc_dict)
    ff = open("../Others/dict", "w")
    for item in voc_dict.keys():
        ff.writelines("{},{}\n".format(item, voc_dict[item]))
    ff.close()
    NPtrain=[]
    len_list=[]
    for words in NPtransform:
        i=0
        input_idx = []
        for word in words:
            if word in voc_dict.keys():
                input_idx.append(voc_dict[word])
                i += 1
        if len(input_idx)<max_len_seq:
            input_idx += [voc_dict["<PAD>"]
                          for _ in range(max_len_seq - len(input_idx))]##Convert character string into number in feature vector
        if i>32:
            i=32
        if i<=0:
            i=1
        NPtrain.append(input_idx)
        len_list.append(i)
    len_list = torch.tensor(len_list, dtype=torch.int).unsqueeze(1)
    data= torch.tensor(NPtrain,dtype=torch.float)[:,:32]
    index = torch.randint(low=0, high=data.shape[1], size=[32]).long()
    target=torch.tensor(target,dtype=torch.float).unsqueeze(1)##become tensor
    print(len(len_list))
    '''=================================分别返回训练特征，训练标签，句子有效长度=============================================='''
    return data,target,len_list
