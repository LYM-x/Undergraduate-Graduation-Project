import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


class time_series_decoder_paper(Dataset):
    """synthetic time series dataset from section 5.1"""
    
    def __init__(self,data,t0,listx,N=4500,preN=18,transform=None):
        """
        Args:
            data:tensor类型，[数目，每条要预测的时间戳数目]
            t0: previous t0 data points to predict from
            N: number of data points
            transform: any transformations to be applied to time series
        """
        
        self.t0 = t0
        self.N = N
        self.transform = None
        temp = []
        
        # time points
  
        for i in range(len(listx)):
            m = listx[i]
            x = torch.arange(m,m+t0+preN).type(torch.float).unsqueeze(0)
            if(i==0):
                temp = x
            else:
                temp = torch.cat([temp,x],dim=0)
        
            
        self.x = temp

        # sinuisoidal signal      
        self.fx = data
        
        self.masks = self._generate_square_subsequent_mask(t0,preN)
                
        
        # print out shapes to confirm desired output
        print("x: {}*{}".format(*list(self.x.shape)),
              "fx: {}*{}".format(*list(self.fx.shape)))
        
    def __len__(self):
        return len(self.fx)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        
        sample = (self.x[idx,:],
                  self.fx[idx,:],
                  self.masks)
        
        if self.transform:
            sample=self.transform(sample)
            
        return sample
    
    def _generate_square_subsequent_mask(self,t0,preN):
        mask = torch.zeros(t0+preN,t0+preN)
        for i in range(0,t0):
            mask[i,t0:] = 1 
        for i in range(t0,t0+preN):
            mask[i,i+1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))#.masked_fill(mask == 1, float(0.0))
        return mask