import torch
import torch.nn as nn
import torch.nn.functional as F

class DataFusion(nn.Module):
    def __init__(self, num_classes, num_data):
        super(DataFusion, self).__init__()
        self.num_classes = num_classes
        self.num_data = num_data
        self.conv1 = conv_block(in_channels=self.num_data, out_channels=8, kernel_size=10, stride=2, padding=4)
        self.conv2 = conv_block(in_channels=8, out_channels=16, kernel_size=10, stride=2, padding=4)
        self.inception1 = Naive_inception_block(16, 8, 8, 8, 8)
        self.fc1 = nn.Linear(250*32, 128)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, self.num_classes)
        self.bn1 = nn.BatchNorm1d(128)

    def forward(self, x = None, y = None, z = None):
        if z == None:
            x = torch.cat([x, y], dim = 1)
        else:
            x = torch.cat([x, y, z], dim = 1)
        x = self.conv1(x)
        x = F.max_pool1d(x, 2)
        x = self.conv2(x)
        x = F.max_pool1d(x, 2)
        x = self.inception1(x)
        x = x.view(-1, 250*32)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x)
    
class FeatureFusion(nn.Module):
    def __init__(self, num_classes, num_data):
        super(FeatureFusion, self).__init__()
        self.num_classes = num_classes
        self.num_data = num_data
        self.conv1a = conv_block(in_channels=1, out_channels=8, kernel_size=10, stride=2, padding=4)
        self.conv1b = conv_block(in_channels=1, out_channels=8, kernel_size=10, stride=2, padding=4)
        self.conv1c = conv_block(in_channels=1, out_channels=8, kernel_size=10, stride=2, padding=4)
        
        self.conv2a = conv_block(in_channels=8, out_channels=16, kernel_size=10, stride=2, padding=4)
        self.conv2b = conv_block(in_channels=8, out_channels=16, kernel_size=10, stride=2, padding=4)
        self.conv2c = conv_block(in_channels=8, out_channels=16, kernel_size=10, stride=2, padding=4)
        
        self.inception1a = Naive_inception_block(16, 8, 8, 8, 8)
        self.inception1b = Naive_inception_block(16, 8, 8, 8, 8)
        self.inception1c = Naive_inception_block(16, 8, 8, 8, 8)
        
        self.fc1 = nn.Linear(250*32*self.num_data, 128)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, self.num_classes)
        self.bn1 = nn.BatchNorm1d(128)

    def forward(self, x = None, y = None, z = None):
        x = self.conv1a(x)
        x = F.max_pool1d(x, 2)
        x = self.conv2a(x)
        x = F.max_pool1d(x, 2)
        x = self.inception1a(x)
        
        y = self.conv1b(y)
        y = F.max_pool1d(y, 2)
        y = self.conv2b(y)
        y = F.max_pool1d(y, 2)
        y = self.inception1b(y)
        
        if z != None:
            z = self.conv1c(z)
            z = F.max_pool1d(z, 2)
            z = self.conv2c(z)
            z = F.max_pool1d(z, 2)
            z = self.inception1c(z)
        
        if z == None:
            x = torch.cat([x, y], dim = 1)
        else:
            x = torch.cat([x, y, z], dim = 1)
        x = x.view(-1, 250*32*self.num_data)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x)

class DecisionFusion(nn.Module):
    def __init__(self, num_classes, num_data):
        super(DecisionFusion, self).__init__()
        self.num_classes = num_classes
        self.num_data = num_data
        self.network1 = InceptionNetwork(num_classes)
        self.network2 = InceptionNetwork(num_classes)
        self.network3 = InceptionNetwork(num_classes)
    
    def forward(self, x = None, y = None, z = None):
        x = self.network1(x)
        y = self.network2(y)
        if z!= None:
            z = self.network3(z)
            
        if z == None:
            return (x+y)/self.num_classes
        else:
            return (x+y+z)/self.num_classes
        
class InceptionNetwork(nn.Module):
    def __init__(self, num_classes):
        super(InceptionNetwork, self).__init__()
        self.num_classes = num_classes
        self.conv1 = conv_block(in_channels=1, out_channels=8, kernel_size=10, stride=2)
        self.conv2 = nn.Conv1d(8, 16, 10, 2)
        self.inception1 = Naive_inception_block(16, 8, 8, 8, 8)
        self.fc1 = nn.Linear(247*32, 128)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, self.num_classes)
        self.bn1 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = self.inception1(x)
        x = x.view(-1, 247*32)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x)   
   
    
class Naive_inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3, out_5x5, out_1x1pool):
        super(Naive_inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)
        self.branch2 = conv_block(in_channels, out_3x3, kernel_size=3, padding=1)
        self.branch3 = conv_block(in_channels, out_5x5, kernel_size=5, padding=2)     
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1),
        )
        
    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )
    
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.relu(self.conv(x))