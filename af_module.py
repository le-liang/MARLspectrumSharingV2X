import torch
import torch.nn as nn

class AF(nn.Module):
    def __init__(self,shape,index):
        super(AF,self).__init__()
        self.factor_prediction = nn.Sequential(
            nn.Conv1d(shape+index,shape,1),
            nn.ReLU(),
            nn.Conv1d(shape,shape,1),
            nn.Sigmoid()
        )

    def forward(self,x,SNR):
        temp = nn.functional.adaptive_avg_pool2d(x,(1,1))
        temp = torch.concat([temp,SNR*torch.ones(x.shape[0],1,1,1)],dim=1)
        temp = self.factor_prediction(torch.squeeze(temp,dim=3))
        temp = torch.unsqueeze(temp,dim=2)
        x *= temp
        return x

    def forward2(self,x,SNR,SNR2):
        temp = nn.functional.adaptive_avg_pool2d(x,(1,1))
        temp = torch.concat([temp,SNR*torch.ones(x.shape[0],1,1,1)],dim=1)
        temp = torch.concat([temp,SNR2*torch.ones(x.shape[0],1,1,1)],dim=1)
        temp = self.factor_prediction(torch.squeeze(temp,dim=3))
        temp = torch.unsqueeze(temp,dim=2)
        x *= temp
        return x