import torch.nn as nn
from torchvision import models
import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.05)  # bottleneck PADA-0.005;fc PADA-0.01; office-0.05
        m.bias.data.fill_(0)


class Attention(nn.Module):
    """ attention Layer"""

    def __init__(self, in_dim=31, out_dim=31):
        super(Attention, self).__init__()

        self.query_fc = nn.Linear(in_dim, out_dim)
        self.query_fc.apply(weights_init)
        self.key_fc = nn.Linear(in_dim, out_dim)
        self.key_fc.apply(weights_init)
        #        self.sigm = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):

        proj_query = self.query_fc(x)  # B * F
        proj_key = self.key_fc(y)  # B * F
        energy = torch.matmul(proj_query, proj_key.transpose(1, 0)) # B * B
        attention = self.softmax(energy) # B * B

        return attention

def AligenLoss(src_feat, tgt_feat, att_net):
    batch_size = src_feat.size(0)
    S = att_net(src_feat, tgt_feat)
    X_ts = torch.mm(S, tgt_feat)
    loss = torch.norm((src_feat - X_ts).abs(), 2, 1).sum() / float(batch_size)

    return loss
