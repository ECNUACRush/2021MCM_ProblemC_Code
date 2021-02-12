import torch
"""
Super parameter setting
"""
class Config():
    def __init__(self,n_vocab):
        self.n_vocab =n_vocab
        self.embed_size = 256
        self.hidden_size = 256
        self.num_layers = 5
        self.dropout = 0.8
        self.other_feature=3
        self.num_classes = 3
        self.pad_size =32
        self.batch_size = 256
        self.is_shuffle = True
        self.learn_rate = 0.01
        self.num_epochs = 50
        self.devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


