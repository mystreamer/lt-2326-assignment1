from torch import nn
import torch.nn.functional as F

class ThaiOCRNN(nn.Module):
    def __init__(self, num_features, num_labels):
        super(ThaiOCRNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(num_features, num_labels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
