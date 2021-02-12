import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
"""
ProblemII Model
"""
'''==========================================Model====================================================='''
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embeding = nn.Embedding(config.n_vocab,
                                      config.embed_size,
                                     padding_idx=config.n_vocab - 1)
        self.bn1 = nn.BatchNorm1d(config.other_feature,affine=False)
        self.lstm = nn.LSTM(config.embed_size,
                            config.hidden_size,
                            config.num_layers,
                            bidirectional=True, batch_first=True,
                            dropout=config.dropout)
        self.len_list=[]
        self.minpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed_size,
                            1)
        self.fc2=nn.Linear(1+config.other_feature,config.num_classes,bias=True)
        init.kaiming_normal_(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.softmax = nn.Softmax(dim=1)
        self.other_feature=config.other_feature

    def forward(self, x):
        m=x[:,:self.other_feature].to(torch.float)
        x = x[:,self.other_feature:].to(torch.int64)
        x = self.embeding(x)# [batchsize, seqlen, embed_size]
        embed=nn.utils.rnn.pack_padded_sequence(input=x, lengths=self.len_list, batch_first=True,enforce_sorted=False)
        out, _ = self.lstm(embed)
        out,len=nn.utils.rnn.pad_packed_sequence(out, batch_first=True,total_length=32)
        out = torch.cat((x, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.minpool(out).reshape(out.size()[0], -1)
        out = self.fc(out)
        out=torch.cat((self.bn1(m),out),dim=1)## m ->Latitude,Longitude,Detection Date
        out = F.relu(out)
        out=self.fc2(out)
        out = self.softmax(out)
        return out





