import os
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import pybaselines
import torch
from torch.utils.data import Dataset


class DataProcess():
    def __init__(self, Datatype):
        self.Datatype = Datatype
        self.x, self.y = self.BuildTrainingData()

    def Preprocess(self, array, scaling=True, baseline=True, smoothing=True):
        if scaling == True:
            if self.Datatype == "LIBS":
                array = (array-np.min(array))/(array[1806]) #Normalise to carbon emission line
            else:
                array = (array-np.min(array))/(np.max(array) - np.min(array)) #Normalise to entire spectrum   
        if smoothing == True:
            array = savgol_filter(array, 9, 3)
        if baseline == True:
            array = array - pybaselines.whittaker.asls(array)[0] 
        return array
    
    def VariableMapping(self, Wavelength, Intensity):
        MappedX = np.zeros([4000])
        WavelengthStart = int(Wavelength[0])
        WavelengthEnd = int(Wavelength[-1])
        if WavelengthEnd > 3999:
            WavelengthEnd = 3999
        if self.Datatype == "LIBS":
            TempX = np.linspace(WavelengthStart, WavelengthEnd, 4000)
        else:
            TempX = np.linspace(WavelengthStart, WavelengthEnd, WavelengthEnd - WavelengthStart+1)
        interp = interp1d(Wavelength, Intensity, kind = "slinear", fill_value="extrapolate")
        ProcessedX = interp(TempX)
        idx = 0
        for i in range(WavelengthStart, WavelengthEnd + 1):
            MappedX[i] = ProcessedX[idx]
            idx += 1
        if self.Datatype == "LIBS":
            return ProcessedX
        else:
            return MappedX    
    
    def BuildTrainingData(self):
        HDPE = "./" + self.Datatype + "/HDPE"
        LDPE = "./" + self.Datatype + "/LDPE"
        PET = "./" + self.Datatype + "/PET"
        PP = "./" + self.Datatype + "/PP"
        LABELS = {HDPE: 0, LDPE: 1, PET:2, PP: 3}
        TrainingDataX = np.zeros([122, 4000])
        TrainingDataY = np.zeros([122, 5])
        
        counter = 0
        for label in LABELS:
            for f in os.listdir(label):
                path = os.path.join(label, f)
                if self.Datatype == "FTIR":
                    wavelength = np.genfromtxt(path, delimiter =',')[:,0]
                    intensity = np.genfromtxt(path, delimiter =',')[:,1]
                elif self.Datatype == "Raman":
                    wavelength = np.flip(np.genfromtxt(path)[:,0])
                    intensity = np.flip(np.genfromtxt(path)[:,1]) 
                elif self.Datatype == "LIBS":
                    wavelength = np.genfromtxt(path, delimiter =',')[:,0][230:15601]
                    intensity = np.genfromtxt(path, delimiter =',')[:,1][230:15601]
                TrainingDataX[counter] = self.VariableMapping(wavelength, self.Preprocess(intensity))
                TrainingDataY[counter] = np.eye(5)[LABELS[label]]
                counter+=1

        return TrainingDataX, TrainingDataY


class MakeDataset(Dataset):
    '''
    Build dataset from entire database
    '''
    def __init__(self, X, Y): 
        self.x = X
        self.y = Y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        feature = self.x[index, :]
        label = self.y[index, :]
        
        feature = torch.tensor(feature, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return {
            'features': feature,
            'labels' : label
        }

class MakePolymerDataset(Dataset):
    '''
    Build dataset from data of one class
    '''
    def __init__(self, X, Y, PolymerIdx): 
        self.x = X
        self.y = Y
        self.PolymerIdx = PolymerIdx
        self.FilteredDataset = self.Filter()
    
    def __len__(self):
        return len(self.FilteredDataset)
    
    def __getitem__(self, index):
        feature = self.FilteredDataset[index, :]
        
        feature = torch.tensor(feature, dtype=torch.float32)
        return {
            'features': feature
        }
    
    def Filter(self):
        FilteredDataset = None
        for i in range(self.x.shape[0]-1, -1, -1):
            if np.argmax(self.y[i]) == self.PolymerIdx:
                if FilteredDataset is None:
                    FilteredDataset = self.x[i]
                else:
                    FilteredDataset = np.vstack([FilteredDataset, self.x[i]])
            
        return FilteredDataset