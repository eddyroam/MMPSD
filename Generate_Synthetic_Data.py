import random
import numpy as np
import torch
from SCAE_Model import SCAE
from VAE_Models import IndividualVAE

def GenerateSCAEData(input_data, input_type, output_type, device):
    """
    :param input_data: numpy array for data of size [a, 4000], where a is the number of data
    :param input_type: string of either "FTIR". "Raman" or "LIBS"
    :param output_type: string of either "FTIR". "Raman" or "LIBS"
    :param device: torch.device("cuda:0") or torch.device("cpu")
    """
    input_data = torch.tensor(input_data, dtype=torch.float32)
    model = SCAE().to(device)
    if device == torch.device("cpu"):
        model.load_state_dict(torch.load("Model/SCAE " + input_type + " to " + output_type + ".pt", map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load("Model/SCAE " + input_type + " to " + output_type + ".pt"))
    
    with torch.no_grad():
        synthetic_data = model(input_data)
         
    return synthetic_data

def GenerateVAEData(output_type, y_label, device):
    """
    :param output_type: string of either "FTIR". "Raman" or "LIBS"
    :param y_label: numpy array for one-hot y-label of size [a, 5], where a is the number of data
    :param device: torch.device("cuda:0") or torch.device("cpu")
    """
    random.seed(42)
    size = y_label.shape[0]
    synthetic_data = torch.zeros([size, 4000])
    
    HDPE_model = IndividualVAE().to(device)
    LDPE_model = IndividualVAE().to(device)
    PET_model = IndividualVAE().to(device)
    PP_model = IndividualVAE().to(device)
    models = [HDPE_model, LDPE_model, PET_model, PP_model]
    
    if device == torch.device("cpu"):
        HDPE_model.load_state_dict(torch.load("Model/SVAE " + output_type + " HDPE.pt", map_location=torch.device('cpu')))
        LDPE_model.load_state_dict(torch.load("Model/SVAE " + output_type + " LDPE.pt", map_location=torch.device('cpu')))
        PET_model.load_state_dict(torch.load("Model/SVAE " + output_type + " PET.pt", map_location=torch.device('cpu')))
        PP_model.load_state_dict(torch.load("Model/SVAE " + output_type + " PP.pt", map_location=torch.device('cpu')))
    else:
        HDPE_model.load_state_dict(torch.load("Model/SVAE " + output_type + " HDPE.pt"))
        LDPE_model.load_state_dict(torch.load("Model/SVAE " + output_type + " LDPE.pt"))
        PET_model.load_state_dict(torch.load("Model/SVAE " + output_type + " PET.pt"))
        PP_model.load_state_dict(torch.load("Model/SVAE " + output_type + " PP.pt"))
    
    with torch.no_grad():
        for i in range(size):
            noise_label = int(np.argmax(y_label[i]))
            rinpt = torch.randn(1, 50).to(device)
            synthetic_data[i] = models[noise_label].decoder(rinpt)
         
    return synthetic_data
         