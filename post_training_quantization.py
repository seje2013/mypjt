import torch
import torch.nn as nn
import torch.nn.functional as F

class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, q_lv):
        return input.mul(q_lv-1).round().div_(q_lv-1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output 

def quantize(input, qbit):
    max_value = input.abs().max()
    output = input/max_value
    output = output*(2**(qbit-1)-1)
    output = output.round()
    output = output/(2**(qbit-1)-1)
    output = output*max_value
    return output

def quantize_asy(input, qbit):
    max_value = input.max()
    output = input/max_value
    output = output*(2**qbit-1)
    output = output.round()
    output = output/(2**qbit-1)
    output = output*max_value
    return output

class QReLU(nn.Module):
    def __init__(self, qbit, threshold):
        super(QReLU, self).__init__()
        self.threshold = threshold
        self.qbit = qbit
    
    def forward(self, x):
        x = torch.clamp(x, min=0, max=self.threshold)
        x = quantize_asy(x, self.qbit)
        return x

def weight_ptq(model, qbit):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.weight.data.copy_(quantize(module.weight.data, qbit))
    return model

def activation_ptq(model, qbit, threshold, act_num = 0):
    for module_name in model._modules:
        if isinstance(model._modules[module_name], nn.ReLU):
            model._modules[module_name] = QReLU(qbit, threshold[act_num])
            act_num += 1
        if len(model._modules[module_name]._modules) > 0:
            activation_ptq(model._modules[module_name], qbit, threshold, act_num)
    return model

def fuse_bn_recursively(model):
    previous_name = None
    
    for module_name in model._modules:
        previous_name = module_name if previous_name is None else previous_name # Initialization
        print(previous_name, module_name)
        conv_fused = fuse_single_conv_bn_pair(model._modules[module_name], model._modules[previous_name])
        if conv_fused:
            model._modules[previous_name] = conv_fused
            model._modules[module_name] = nn.Identity()
            
        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])
            
        previous_name = module_name

    return model

def calibration(activation, act_num, qbit):
    threshold = {}
    for i in range(act_num):
        input, _ = activation[i].flatten().sort(descending=True)
        layer_divergence, layer_threshold = None, 0
        calib_idx = 0
        for j in range(1000):
            temp = input.clone()
            temp_threshold = temp[calib_idx]
            temp[0:calib_idx] = temp_threshold
            temp = temp.div(temp_threshold).mul(2**qbit-1)
            temp = temp.round()
            temp = temp.div(2**qbit-1).mul(temp_threshold)
            temp_divergence = nn.KLDivLoss()(input.log(), temp)
            if layer_divergence is None: 
                layer_divergence = temp_divergence
            else:
                if layer_divergence > temp_divergence:
                    layer_divergence = temp_divergence
                    layer_threshold = temp_threshold
            calib_idx += 1
        threshold[i] = layer_threshold
        print(f'==> calibrating relu{i}')
    return threshold