from torchsummary import summary
import nets
import os
import torch
from thop import profile

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

"""device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = nets.ustereo10_1(192).to(device)
summary(net, [(3, 576, 960), (3, 576, 960)], device='cuda')  # or cuda"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = nets.ustereo10_2(192).to(device)

input1 = torch.randn(1, 3, 576, 960).cuda()
input2 = torch.randn(1, 3, 576, 960).cuda()

flop, para = profile(net, inputs=(input1, input2, ))

print(flop, type(flop))
print(para, type(para))
total = sum([param.nelement() for param in net.parameters()])
print('Number of parameter: %.2fM' % (total/1e6))