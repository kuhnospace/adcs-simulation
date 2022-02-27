import torch
from torch import nn
import torchsde

# Guidance from https://github.com/google-research/torchsde/blob/master/examples/demo.ipynb
class SDE(nn.Module):
    def __init__(self, f_and_g_function, *args):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(0.1), requires_grad=True) # Scalar parameter
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.f_and_g_function = f_and_g_function
        self.args = args

    def f_and_g(self, t, y):
        return self.f_and_g_function(t, y, *self.args)

def integrate(sde, y0, ts):
    y = torch.tensor([y0], dtype=torch.float64)
    with torch.no_grad():
        ys = torchsde.sdeint(sde, y, torch.tensor(ts), method='euler', adaptive=True)
    return ys
