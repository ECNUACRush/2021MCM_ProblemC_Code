import torch
"""
Super parameter setting
"""
class Config():
    def __init__(self):
        self.embed_size = 10
        self.hidden_size = 5
        self.num_layers = 5
        self.dropout = 0.8
        self.num_classes = 1
        self.batch_size = 64
        self.is_shuffle = True
        self.learn_rate = 0.5
        self.num_epochs =5000
        self.devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



