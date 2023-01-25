import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from Preprocessing import DataProcess, MakeDataset, MakePolymerDataset
from VAE_Models import ConditionalVAE, IndividualVAE
from Generate_Synthetic_Data import GenerateVAEData
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def FinalLoss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def WeightsInit(m): 
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight.data)
        
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on a GPU")
else:
    device = torch.device("cpu")
    print("Running on a CPU")

#Hyperparameters
batch_size = 24
epochs = 100
lr = 0.0001
loss_history = []
embeddings = nn.Embedding(5, 10)

#Loading Datasets
polymers =["HDPE", "LDPE", "PET", "PP"]
Datatype = "FTIR"
x, y = DataProcess(Datatype).BuildTrainingData()
dataset = MakeDataset(x, y)
HDPE_dataset = MakePolymerDataset(x, y, 0)
LDPE_dataset = MakePolymerDataset(x, y, 1)
PET_dataset = MakePolymerDataset(x, y, 2)
PP_dataset = MakePolymerDataset(x, y, 3)
datasets = [HDPE_dataset, LDPE_dataset, PET_dataset, PP_dataset]

'''Code for Training Separate VAE Models'''

#Model Initialisation
for i in range(len(datasets)):
    SVAE = IndividualVAE().to(device)
    SVAE.apply(WeightsInit)  
    optimizer = optim.Adam(SVAE.parameters(), lr=lr)
    dataloader = DataLoader(datasets[i], batch_size)
    criterion = nn.BCELoss()
  
    #Training Loop    
    for epoch in range(epochs):
        batch = tqdm(enumerate(dataloader), desc = polymers[i] + " Epoch " + str(epoch + 1), total = len(dataloader.dataset)//dataloader.batch_size)
    
        for j, data in batch:
            features = data['features'].to(device)
            optimizer.zero_grad()
            reconstruction, mu, logvar = SVAE(features)
            bce_loss = criterion(reconstruction, features)
            loss = FinalLoss(bce_loss, mu, logvar)
            loss.backward()
            optimizer.step()
    
    #torch.save(SVAE.state_dict(), "SVAE" + Datatype + " " + polymers[i] + " .pt")   
    
'''Code for Conditional VAE Models'''

#Model Inititalisation 
CVAE = ConditionalVAE().to(device)
CVAE.apply(WeightsInit)  
optimizer = optim.Adam(CVAE.parameters(), lr=lr)
dataloader = DataLoader(dataset, batch_size)
criterion = nn.BCELoss()
  
#Training Loop
for epoch in range(epochs):
    batch = tqdm(enumerate(dataloader), desc = "Epoch " + str(epoch + 1), total = len(dataloader.dataset)//dataloader.batch_size)

    for i, data in batch:
        features = data['features'].to(device)
        labels = torch.argmax(data['labels'].to(device), dim=1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = CVAE(features, embeddings(labels))
        bce_loss = criterion(reconstruction, features)
        loss = FinalLoss(bce_loss, mu, logvar)
        loss.backward()
        optimizer.step()

#torch.save(CVAE.state_dict(), "CVAE" + Datatype + ".pt")   