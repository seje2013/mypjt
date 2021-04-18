import torch
import torch.nn.functional as F

input = torch.load('./input')
weight = torch.load('./weight')

output = F.conv2d(input, weight)

print(output.shape)
input = torch.randint_like(input,15)

print('finish')