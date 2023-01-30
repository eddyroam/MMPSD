import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from Preprocessing import DataProcess, MakeDataset
from SCAE_Model import SCAE
from Generate_Synthetic_Data import GenerateSCAEData
from Evaluate_FID import ReturnFID
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

InputData = "FTIR"      #Change between "FTIR", "Raman" or "LIBS"
OutputData = "Raman"    #Change between "FTIR", "Raman" or "LIBS"


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
kfold = 5
batch_size = 128
epochs = 100
lr = 0.001
num_runs = 10
DataType = {"FTIR": 0, "Raman": 1, "LIBS":2}

#Loading Datasets
ftir_x, ftir_y = DataProcess("FTIR").BuildTrainingData()
raman_x, raman_y = DataProcess("Raman").BuildTrainingData()
libs_x, libs_y = DataProcess("LIBS").BuildTrainingData()
x = np.vstack([ftir_x, raman_x, libs_x]).reshape([122,3,4000])
dataset = MakeDataset(x, ftir_y) #ftir_y, raman_y and libs_y are the same

#Model Initialisation
fid_scores = []
SCAE = SCAE().to(device)
optimizer = optim.Adam(SCAE.parameters(), lr=lr)
criterion = nn.MSELoss()
kfold = KFold(n_splits=kfold, shuffle=True, random_state=42)
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    train_subsampler = SubsetRandomSampler(train_ids)
    test_subsampler = SubsetRandomSampler(test_ids)

    train_dataloader = DataLoader(dataset, batch_size, sampler=train_subsampler)
    test_dataloader = DataLoader(dataset, batch_size, sampler=test_subsampler)
 
    #Training Loop  
    SCAE.apply(WeightsInit)   
    for epoch in range(epochs):
        batch = tqdm(enumerate(train_dataloader), desc = " Epoch " + str(epoch + 1), total = len(train_dataloader.dataset)//train_dataloader.batch_size)
        
        for i, data in batch:
            features = data['features'][:,DataType[InputData]].to(device)
            groundtruth = data['features'][:,DataType[OutputData]].to(device)
            labels = data['labels'].to(device)
            optimizer.zero_grad()
            output = SCAE(features)
            loss = criterion(output, groundtruth)
            loss.backward()
            optimizer.step()
    
    #Evaluation
    use_pretrained = False
    synthetic_data, groundtruth = GenerateSCAEData(use_pretrained, SCAE, test_dataloader, InputData, OutputData)
    fid = ReturnFID(groundtruth, synthetic_data, device)
    fid_scores.append(fid)

fid_average = sum(fid_scores)/len(fid_scores)
print("FID score: " + "%.3f" % fid_average)
#torch.save(SCAE.state_dict(), "SCAE " + InputData + " to " + OutputData + ".pt")    