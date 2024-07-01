from torch import nn
import torch
import torch.nn.functional as F
class BasicCNN(nn.Module):
    def __init__(self, output_size):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(1600, 50)
        self.fc2 = nn.Linear(50, output_size)
        
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def predict(self, output):
        return torch.argmax(F.log_softmax(output, dim=1), dim=1)