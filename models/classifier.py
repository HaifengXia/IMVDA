from torch import nn
import torch
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.functional as F

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class feat_classifier(nn.Module):
    def __init__(self, num_class, bottleneck_dim=2048, type='linear'):
        super(feat_classifier, self).__init__()
        self.type = type
        self.num_parallel = 2
        self.alpha = nn.Parameter(torch.ones(self.num_parallel, requires_grad=True))
        self.fc_multi = nn.Linear(bottleneck_dim, 256)
        self.fc_multi.apply(init_weights)
        self.fc_depth = nn.Linear(bottleneck_dim, 256)
        self.fc_depth.apply(init_weights)
        self.fc_uni = nn.Linear(bottleneck_dim, 256)
        self.fc_uni.apply(init_weights)
        self.relu = nn.ReLU(inplace=True)
        self.fc_add = nn.Linear(256, 256)
        self.fc_add.apply(init_weights)
        self.fc = nn.Linear(256, num_class)
        self.fc.apply(init_weights)
        self.fc1 = nn.Linear(256, num_class)
        self.fc1.apply(init_weights)
        if self.type == 'wn':
            self.fc = weightNorm(nn.Linear(256, num_class), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(256, num_class)
            self.fc.apply(init_weights)

    def forward(self, x, y=None):
        if y is not None:
            x = self.fc_multi(x)
            y = self.fc_depth(y)
            alpha_soft = F.softmax(self.alpha)
            fusion = alpha_soft[0] * x + alpha_soft[1] * y
            out = self.fc(fusion)
            return x, fusion, out
        else:
            x1 = self.fc_uni(x)
            x2 = self.relu(x1)
            fusion = self.fc_add(x2)
            out = self.fc(fusion)
            return x1, fusion, out