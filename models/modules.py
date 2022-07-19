import torch.nn as nn
import torch

class Exchange(nn.Module):
	def __init__(self):
		super(Exchange, self).__init__()

	def forward(self, x, bn, bn_threshold):
		bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
		x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])

		num_params = int(len(bn1)/2)
		p1 = num_params
		x11, x12 = x[0][:, 0:p1], x[0][:, p1:]
		x21, x22 = x[1][:, 0:p1], x[1][:, p1:]
		bn11, bn12 = bn1[0:p1], bn1[p1:]
		bn21, bn22 = bn2[0:p1], bn2[p1:]

		y12, y22 = torch.zeros_like(x12), torch.zeros_like(x22)
		y12[:, bn12>bn_threshold] = x12[:, bn12>bn_threshold]
		y12[:, bn12<bn_threshold] = x22[:, bn12<bn_threshold]
		y22[:, bn22>bn_threshold] = x22[:, bn22>bn_threshold]
		y22[:, bn22<bn_threshold] = x12[:, bn22<bn_threshold]

		x1[:, 0:p1], x1[:, p1:] = x11, y12
		x2[:, 0:p1], x2[:, p1:] = x21, y22

		return [x1, x2]

class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]
