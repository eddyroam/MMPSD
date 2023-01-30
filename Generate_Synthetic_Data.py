import random
import numpy as np
import torch

def GenerateSCAEData(use_pretrained, model, dataloader, input_type, output_type):
    """
    :param input_data: numpy array for data of size [a, 4000], where a is the number of data
    :param input_type: string of either "FTIR". "Raman" or "LIBS"
    :param output_type: string of either "FTIR". "Raman" or "LIBS"
    :param device: torch.device("cuda:0") or torch.device("cpu")
    """
    if use_pretrained:
        model.load_state_dict(torch.load("Model/SCAE " + input_type + " to " + output_type + ".pt", map_location=torch.device('cpu')))
        
    DataType = {"FTIR": 0, "Raman": 1, "LIBS": 2}
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            features = data['features'][:,DataType[input_type]]
            groundtruth = data['features'][:,DataType[output_type]]
            synthetic_data = model(features)
         
    return synthetic_data, groundtruth

def GenerateVAEData(use_pretrained, models, output_type, y_label, device):
    """
    :param output_type: string of either "FTIR". "Raman" or "LIBS"
    :param y_label: numpy array for one-hot y-label of size [a, 5], where a is the number of data
    :param device: torch.device("cuda:0") or torch.device("cpu")
    """
    random.seed(42)
    size = y_label.shape[0]
    synthetic_data = torch.zeros([size, 4000])
    
    HDPE_model = models[0]
    LDPE_model = models[1]
    PET_model = models[2]
    PP_model = models[3]
    models = [HDPE_model, LDPE_model, PET_model, PP_model]
    
    if use_pretrained:
        HDPE_model.load_state_dict(torch.load("Model/SVAE " + output_type + " HDPE.pt", map_location=torch.device('cpu')))
        LDPE_model.load_state_dict(torch.load("Model/SVAE " + output_type + " LDPE.pt", map_location=torch.device('cpu')))
        PET_model.load_state_dict(torch.load("Model/SVAE " + output_type + " PET.pt", map_location=torch.device('cpu')))
        PP_model.load_state_dict(torch.load("Model/SVAE " + output_type + " PP.pt", map_location=torch.device('cpu')))
    
    with torch.no_grad():
        for i in range(size):
            noise_label = int(np.argmax(y_label[i]))
            rinpt = torch.randn(1, 50).to(device)
            synthetic_data[i] = models[noise_label].decoder(rinpt)
         
    return synthetic_data
         