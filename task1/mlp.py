import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Simple mlp model
class BaseNN(nn.Module):
    def __init__(self, num_features = 8):
        super(BaseNN, self).__init__()
        self.feature_num = num_features
        self.fc1   = nn.Linear(self.feature_num,128)
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, 16)
        self.dp = nn.Dropout(p = 0.2)
        self.fc4   = nn.Linear(16, 1)

    def forward(self, x):  
        out = F.relu(self.fc1(x))
        out = self.dp(out)
        out = F.relu(self.fc2(out))
        out = self.dp(out)
        out = F.relu(self.fc3(out))
        out = self.dp(out)
        out = self.fc4(out)
        return out

class houseDataset(Dataset):
    def __init__(self,house_data):

        x = house_data[:,:-1]
        y = house_data[:,-1]

        self.x_train = torch.tensor(x,dtype=torch.float32)
        self.y_train = torch.tensor(y,dtype=torch.float32)
        self.y_train = self.y_train.resize_(house_data.shape[0], 1)

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]

class houseTestDataset(Dataset):
    def __init__(self,house_data):
        x = house_data
        self.x_train = torch.tensor(x,dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self,idx):
        return self.x_train[idx]