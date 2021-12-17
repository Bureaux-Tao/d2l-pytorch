##
import torch
from torch import nn
from torch.nn import functional as F

##
x = torch.arange(4)
torch.save(x, 'weights/x-file')

##
x2 = torch.load('weights/x-file')
x2
# Out[42]: tensor([0, 1, 2, 3])

##
y = torch.zeros(4)
torch.save([x, y], 'weights/x-files')
x2, y2 = torch.load('weights/x-files')
(x2, y2)
# (tensor([0, 1, 2, 3]), tensor([0., 0., 0., 0.]))

##
mydict = {'x': x, 'y': y}
torch.save(mydict, 'weights/mydict')
mydict2 = torch.load('weights/mydict')
mydict2


# {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}

##
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
    
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()
X = torch.randn(size = (2, 20))
Y = net(X)

##
torch.save(net.state_dict(), 'weights/mlp.params')

##
clone = MLP()
clone.load_state_dict(torch.load('weights/mlp.params'))
clone.eval()

##
Y_clone = clone(X)
Y_clone == Y
# tensor([[True, True, True, True, True, True, True, True, True, True],
#         [True, True, True, True, True, True, True, True, True, True]])