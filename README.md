# Multi-Modal Plastic Spectral Database
The Multi-Modal Plastic Spectral Database (MMPSD) contains FTIR, Raman and LIBS data for all samples in the database. This repository contains the PyTorch implementation of novel cross-modal generative models and multi-modal deep learning using the MMPSD.

## Cross-modal Generative Models
Cross-generative models are used to generate synthetic data from another data modality of the same sample. For instance, FTIR data could be freely converted to its equivalent Raman and LIBS data. This is achieved using a Spectral Conversion Autoencoder (SCAE). To train the SCAE to convert FTIR data to Raman data, the encoder would take in the FTIR data as input, while the output of the decoder would be optimized using a loss function with respect to the equivalent Raman data. 

![image](https://user-images.githubusercontent.com/97589250/215505206-9c14c2bd-22cf-422a-9425-2847065f876a.png)

## Multi-modal Deep Learning
Multi-modal deep learning model is achieved by fusing the data from different modalities together. There are three different levels of multi-modal data fusion algorithms – 1) data fusion, 2) feature fusion and 3) decision fusion. An Inception network was adopted as the base network for the multi-modal deep learning. There are three different types of data fusion algorithm – data fusion, feature fusion and decision fusion. In the data fusion method, the inputs are concatenated as separate channels, while feature fusion is done through concatenation at the inception block level, when is then flattened and fed into a fully-connected layer before the decision. Decision fusion employs a soft voting ensemble method, where the output probabilities of each of the deep learning model is averaged, and the class with the highest average probability will be the final decision. 

![image](https://user-images.githubusercontent.com/97589250/215505545-803d7751-8aeb-4400-b43a-a25148f50617.png)

# Usage
Run the files ending with '_train.py'. 

For VAE_train, change variable _Datatype_ between "FTIR", "Raman" or "LIBS" to train the model to generate respective synthetic data.

    Datatype = "FTIR"   #Change between "FTIR", "Raman" or "LIBS"
   
For SCAE_train, change variables _InputData_ and _OutputData_ between "FTIR", "Raman" or "LIBS" to train the model to convert between indicated data modalities.

    InputData = "FTIR"      #Change between "FTIR", "Raman" or "LIBS"
    OutputData = "Raman"    #Change between "FTIR", "Raman" or "LIBS"
    
For Multimodal_train, change variable _model_ between DataFusion, FeatureFusion or DecisionFusion to indicate the type of data fusion algorithm.

    model = DataFusion(num_classes, num_data).to(device) #Choose between DataFusion, FeatureFusion or DecisionFusion
    
# Citations
    
