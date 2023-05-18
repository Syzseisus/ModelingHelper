import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 3, 3, 1, 1)
        self.layer2 = nn.Linear(3 * 28 * 28, 30)
        self.layer3 = nn.Linear(30, 10)
        self.lrelu = nn.LeakyReLU()
        nn.init.xavier_uniform_(self.layer1.weight.data)
        nn.init.zeros_(self.layer1.bias.data)
        nn.init.kaiming_normal_(self.layer2.weight.data)
        nn.init.zeros_(self.layer2.bias.data)

    def forward(self, x):
        x = self.lrelu(self.layer1(x))
        x = self.layer2(x.flatten(1))
        z = self.layer3(x)

        return F.log_softmax(F.relu(z), dim=1)
