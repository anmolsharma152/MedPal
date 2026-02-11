import torch
import torch.nn as nn

class TabularResNet(nn.Module):
    def __init__(self, input_dim):
        super(TabularResNet, self).__init__()
        # Encoder
        self.entry = nn.Linear(input_dim, 128)
        self.bn_entry = nn.BatchNorm1d(128)
        
        # Residual Block 1
        self.lin1 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Residual Block 2
        self.lin2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Output
        self.head = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.bn_entry(self.entry(x)))
        identity = x 
        
        out = self.dropout(self.relu(self.bn1(self.lin1(x))))
        out = self.bn2(self.lin2(out))
        
        out += identity # Skip Connection
        out = self.relu(out)
        
        return self.sigmoid(self.head(out))