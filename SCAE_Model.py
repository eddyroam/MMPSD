import torch
import torch.nn as nn
import torch.nn.functional as F

class SCAE(nn.Module):
    def __init__(self):
        super(SCAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Linear(in_features=4000, out_features=500)
        self.enc2 = nn.Linear(in_features=500, out_features=500)
        self.enc3 = nn.Linear(in_features=500, out_features=100)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=100, out_features=500)
        self.dec2 = nn.Linear(in_features=500, out_features=500)
        self.dec3 = nn.Linear(in_features=500, out_features=4000)
    
    def encoder(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        return x
    
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = torch.tanh(self.dec3(x))
        return x
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x