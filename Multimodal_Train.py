import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from Preprocessing import DataProcess, MakeDataset
from Multimodal_Models import DataFusion, FeatureFusion, DecisionFusion
from Generate_Synthetic_Data import GenerateSCAEData
from Confusion_Matrix_Metrics import MetricEvaluation
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
epochs = 10
lr = 0.001
num_runs = 10
num_classes = 5
num_data = 2

#Loading Datasets
ftir_x, ftir_y = DataProcess("FTIR").BuildTrainingData()
raman_x, raman_y = DataProcess("Raman").BuildTrainingData()
libs_x, libs_y = DataProcess("LIBS").BuildTrainingData()

'''Generate Synthetic Data to feed into model, if desired'''
#fake_raman_x = GenerateSCAEData(input_data=ftir_x, input_type = "FTIR", output_type = "Raman").create().cpu().numpy()

x = np.dstack([ftir_x, raman_x, libs_x]).reshape([122,3,4000]) 
dataset = MakeDataset(x, ftir_y) #ftir_y, raman_y and libs_y are the same

#Training with kfold cross-validation
kfold = KFold(n_splits=kfold, shuffle=True, random_state=42)
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    train_subsampler = SubsetRandomSampler(train_ids)
    test_subsampler = SubsetRandomSampler(test_ids)

    train_dataloader = DataLoader(dataset, batch_size, sampler=train_subsampler)
    test_dataloader = DataLoader(dataset, batch_size, sampler=test_subsampler)
    
    #Model Initialisation
    model = DataFusion(num_classes, num_data).to(device) #Choose between DataFusion, FeatureFusion or DecisionFusion
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.apply(WeightsInit)
    criterion = nn.MSELoss()

    #Training Loop  
    for epoch in range(epochs):
        batch = tqdm(enumerate(train_dataloader), desc = " Epoch " + str(epoch + 1), total = len(train_dataloader.dataset)//train_dataloader.batch_size)
        
        for i, data in batch:
            ftir = data['features'][:,0].view(-1,1,4000).to(device)
            raman = data['features'][:,1].view(-1,1,4000).to(device)
            libs = data['features'][:,2].view(-1,1,4000).to(device)
            labels = data['labels'].to(device)
            optimizer.zero_grad()
            output = model(ftir, raman, libs) #ensure that the number of inputs into model is same as 'num_data'
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
    #Evaluate
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            ftir = data['features'][:,0].view(-1,1,4000).to(device)
            raman = data['features'][:,1].view(-1,1,4000).to(device)
            #libs = data['features'][:,2].view(-1,1,4000).to(device)
            labels = data['labels'].to(device)
            outputs = model(ftir, raman)
            
            y_true = torch.argmax(labels, dim=1).cpu().numpy()
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
            cm = confusion_matrix(y_true, y_pred)
            accuracy, precision, recall = MetricEvaluation(cm).calculate()
            print('Accuracy: ' + '%.3f' % accuracy + ' Precision: ' + '%.3f' % precision + ' Recall: ' + '%.3f' % recall)
            
#torch.save(model.state_dict(), type(model).__name__ + ".pt")    



