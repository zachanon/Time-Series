import torch
import torch.nn as nn
import numpy as np

class ARIMAModelLinear(nn.Module):
    
    def __init__(self,dimin, dimout):
        super(AutoregressionModelLinear, self).__init__()
        
        self.f = nn.Linear(dimin, dimout)
        self.g = nn.Linear(dimout, dimout)
        
    def forward(self, x):
        
        fout = self.f(x)
        gout = self.g(fout)
        
        return fout, gout
        
    
def xtildeloss(xtilde, xtilde_covariance):
    
    meanloss = torch.sum((xtilde - 0)**2)
    covloss = (torch.det(xtilde_covariance) - 1)**2
    
    return meanloss + covloss

def xhatloss(xhat, xt):
    
    return torch.sum((xhat-xt)**2)