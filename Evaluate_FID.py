import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from Multimodal_Models import InceptionNetwork
import torch
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# calculate frechet inception distance
def CalculateFID(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def ReturnFID(real_data, synthetic_data, device):
    """
    :param real_data: Tensor for real data of size [a, 4000], where a is the number of data
    :param synthetic_data: Tensor for synthetic data of size [a, 4000], where a is the number of data
    :param device: torch.device("cuda:0") or torch.device("cpu")
    """
    size = real_data.shape[0]
    
    model = InceptionNetwork(5).to(device)
    model.load_state_dict(torch.load("Model/Pretrained Inception Net.pt", map_location=torch.device('cpu')))
    model.fc2 = Identity()
    
    model.eval()
    with torch.no_grad():
        real_output = model(real_data.view(size,1,4000))
        fake_output = model(synthetic_data.view(size,1,4000))
        fid = CalculateFID(real_output.cpu().numpy(), fake_output.cpu().numpy())
                      
    return fid