import torch.nn as nn
import torch
import torch.nn.functional as F
"""
model Lstm->Linear->Lstm->Linear
"""
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(config.embed_size,
                            config.hidden_size,
                            config.num_layers-2,
                            bidirectional=True, batch_first=True,
                            dropout=config.dropout)
        self.lstm2 = nn.LSTM(config.embed_size,
                            config.hidden_size,
                            config.num_layers-2,
                            bidirectional=True, batch_first=True,
                            dropout=config.dropout)
        self.fc1 = nn.Linear(config.hidden_size*2,
                            config.embed_size)
        self.fc2=nn.Linear(config.hidden_size*2,config.num_classes)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.fill_(1)
        self.bn1=nn.BatchNorm1d(config.embed_size,affine=True)
    def forward(self, x):
        x=x.to(torch.float32)
        out=x.unsqueeze(1)
        out, _ = self.lstm(out)
        out=out.view(x.size(0), -1)
        out = self.fc1(out)
        out=x.unsqueeze(1)
        out,_=self.lstm2(out)
        out=out.view(x.size(0),-1)
        out=self.fc2(out)
        return out
