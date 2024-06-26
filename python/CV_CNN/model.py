#Complex Conv ref:
#https://github.com/Medabid1/ComplexValuedCNN/tree/master/src
#http://t.csdn.cn/hX7nA
#http://t.csdn.cn/IqfOu
#http://t.csdn.cn/xoyxO
#https://github.com/wavefrontshaping/complexPyTorch
#https://github.com/williamFalcon/pytorch-complex-tensor
#https://github.com/ChihebTrabelsi/deep_complex_networks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from complexLayers import ComplexBatchNorm2d,ComplexConv2d,ComplexReLU,ComplexLinear,ComplexDropout,C_CSELU,ZReLU
from complexFunctions import complex_relu, complex_max_pool2d
class CNN_Attention(nn.Module):
    def __init__(self):
        super(CNN_Attention, self).__init__()
        self.a = nn.Sequential(
            ComplexConv2d(1, 1, kernel_size=(1, 1), stride=(1, 1)),
            # nn.GELU(),
            ComplexConv2d(1, 1, kernel_size=(3, 3), stride=(1, 1),padding=(1,1))
        )

        self.v = ComplexConv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
        self.proj = ComplexConv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x1 = self.a(x)*self.v(x)
        x1 = self.proj(x1)
        # x = F.normalize((x+x1), p=2, dim=0, eps=1e-12, out=None)
        # x = F.normalize((x1), p=2, dim=0, eps=1e-12, out=None)
        # x=(x + x1)
        return x-x1
class CV_CNN_Net(nn.Module):
    def __init__(self,device):
        super(CV_CNN_Net, self).__init__()
        self.device = device
        self.nn = nn.Sequential(
                                ComplexConv2d(1, 128, kernel_size = (3, 3),stride = (2, 2),padding=(1,1)),
                                ComplexBatchNorm2d(128),
                                # ComplexReLU(),
                                C_CSELU(),
                                # ZReLU(),

                                ComplexConv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                ComplexBatchNorm2d(128),
                                # ComplexReLU(),
                                C_CSELU(),
                                # ZReLU(),

                                ComplexConv2d(128, 128, kernel_size=(2, 2), stride=(1, 1)),
                                ComplexBatchNorm2d(128),
                                # ComplexReLU(),
                                C_CSELU(),
                                # ZReLU(),

                                ComplexConv2d(128, 128, kernel_size=(2, 2), stride=(1, 1)),
                                ComplexBatchNorm2d(128),
                                # ComplexReLU(),
                                C_CSELU(),
                                # ZReLU(),

                                nn.Flatten(),
                                ComplexLinear(4608,2048),
                                # ComplexReLU(),
                                C_CSELU(),
                                # ZReLU(),

                                ComplexDropout(p=0.3,device = self.device),
                                ComplexLinear(2048, 1024),
                                # ComplexReLU(),
                                C_CSELU(),
                                # ZReLU(),

                                ComplexDropout(p=0.3,device = self.device),
                                ComplexLinear(1024, 121),
                                # ComplexReLU(),
                                C_CSELU(),
                                # ZReLU(),
                                ComplexDropout(p=0.3, device=self.device),
                                )
        self.FNN = nn.Sequential(
            # nn.Linear(242,121),
            nn.Linear(242, 121, bias=False),
            nn.Sigmoid()
        )
    def forward(self, data_train):
        Outputs = self.nn(data_train)
        Outputs = torch.cat([Outputs.real,Outputs.imag],dim=1)
        Outputs = self.FNN(Outputs)
        return Outputs
def model_init(model):
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name[5:] or model_param_name[6:] in ['conv_r.weight','conv_i.weight']:
            nn.init.orthogonal_(model_param_value.unsqueeze(0))
            # nn.init.xavier_normal_(model_param_value.data.imag)
        # if model_param_name[5:] or model_param_name[6:]in ['bias','conv_r.bias','conv_i.bias','fc_i.bias','fc_r.bias']:
        #     nn.init.xavier_normal_(model_param_value.unsqueeze(0))
        # #     # nn.init.zeros_(model_param_value.data.imag)
        # if model_param_name[5:] or model_param_name[6:] in ['fc_r.weight', 'fc_i.weight']:
        #     nn.init.xavier_normal_(model_param_value.unsqueeze(0))
        # if model_param_name[5:] or model_param_name[6:] in ['weight']:
        #     nn.init.sparse_(model_param_value.data.unsqueeze(0), sparsity=0.1)

def get_layer_output(model, x, target_layer_type, target_layer_num):
    outputs = []
    layer_count = 0

    def save_intermediate_output(module, input, output):
        if isinstance(module, target_layer_type):
            nonlocal layer_count
            if layer_count == target_layer_num:
                outputs.append(output)

            layer_count += 1

    handles = []

    for module in model.modules():
        if isinstance(module, target_layer_type):
            handle = module.register_forward_hook(save_intermediate_output)
            handles.append(handle)

    with torch.no_grad():
        model(x)

    for handle in handles:
        handle.remove()

    return outputs[0] if outputs else None