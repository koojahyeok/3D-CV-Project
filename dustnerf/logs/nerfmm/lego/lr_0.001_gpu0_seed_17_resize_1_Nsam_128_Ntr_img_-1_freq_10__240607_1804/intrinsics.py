import torch
import torch.nn as nn
import numpy as np


class LearnFocal(nn.Module):
    def __init__(self, H, W, n_images, order=1, init_focal=None, init_weight=2, init_global_bias=-300, init_local_bias=None, req_grad=True, fx_only=True):
        super(LearnFocal, self).__init__()
        self.H = H
        self.W = W
        self.n_images = n_images
        self.fx_only = fx_only  # If True, output [fx, fx]. If False, output [fx, fy]
        self.order = order  # check our supplementary section.
       
        self.init_weight = init_weight
        self.init_global_bias = init_global_bias
        
        if init_local_bias is None:
            self.init_local_bias = [0.0 for _ in range(self.n_images)]
        else:
            assert(len(init_local_bias) == self.n_images)
            self.init_local_bias = init_local_bias
        
        self.weight_global = nn.Parameter(torch.tensor(self.init_weight, dtype=torch.float32), requires_grad=req_grad)
        self.bias_global = nn.Parameter(torch.tensor(self.init_global_bias, dtype=torch.float32), requires_grad=req_grad)
        self.bias_local = nn.Parameter(torch.tensor(self.init_local_bias, dtype=torch.float32), requires_grad=req_grad)

        if init_focal is None:
            # self.fx = torch.FloatTensor(2.0, dtype=torch.float32), requires_grad=req_grad)
            raise ValueError("init point must be given!")
        else:
            self.fx = torch.FloatTensor(init_focal)
        self.fx /= self.W

    def forward(self, idx):  # the i=None is just to enable multi-gpu training
        if self.fx_only:
            if self.order == 2:
                # fxfy = torch.stack([self.fx ** 2 * self.W, self.fx ** 2 * self.W])
                fxfy = 1
            else:
                fxfy = torch.stack([self.fx[idx] * self.W, self.fx[idx] * self.W])
        else:
            raise ValueError("Not supported!!!!")
            if self.order == 2:
                fxfy = torch.stack([self.fx**2 * self.W, self.fy**2 * self.H])
            else:
                fxfy = torch.stack([self.fx * self.W, self.fy * self.H])
        print(fxfy.device, self.weight_global.device, self.bias_global.device, self.bias_local[idx].device)

        if idx == -1:
            # eval mode
            out = torch.mean(fxfy * self.weight_global + self.bias_global + self.bias_local)
        else:
            out = fxfy * self.weight_global + self.bias_global + self.bias_local[idx]
        
        return torch.abs(out)  # focal length should be positive


if __name__ == "__main__":
    model = LearnFocal(100, 200, 5, init_focal=torch.FloatTensor([200, 400, 300, 400, 500]).unsqueeze(1))
    
    print("test start")
    
    for i in range(5):
        print(f"{i}-th focal length:", model(i))
    
    print("success.")
    
    