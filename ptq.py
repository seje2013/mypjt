import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

def quantize_symmetric(input, qbit):
    max_value = input.abs().max()
    output = input/max_value
    output = output*(2**(qbit-1)-1)
    output = output.round()
    output = output/(2**(qbit-1)-1)
    output = output*max_value
    return output

def quantize_asymmetric(input, qbit):
    max_value = input.max()
    output = input/max_value
    output = output*(2**qbit-1)
    output = output.round()
    output = output/(2**qbit-1)
    output = output*max_value
    return output

def weight_ptq(model, qbit):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.weight.data.copy_(quantize_symmetric(module.weight.data, qbit))
    return model

class QReLU(nn.Module):
    def __init__(self, qbit, threshold):
        super(QReLU, self).__init__()
        self.threshold = threshold
        self.qbit = qbit
    
    def forward(self, x):
        x = torch.clamp(x, min=0, max=self.threshold)
        #x = torch.clamp(x, min=0, max=x.max().data)
        x = quantize_asymmetric(x, self.qbit)
        return x

def set_qrelu(model, qbit, threshold, act_num = 0):
    for module_name in model._modules:
        if isinstance(model._modules[module_name], nn.ReLU):
            model._modules[module_name] = QReLU(qbit, threshold[act_num])
            act_num += 1
        if len(model._modules[module_name]._modules) > 0:
            set_qrelu(model._modules[module_name], qbit, threshold, act_num)
    return model

def calibration(data, layer_num, qbit):
    threshold = {}
    for i in range(layer_num):
        input, _ = data[i].flatten().sort(descending=True)
        layer_divergence, layer_threshold = None, 0
        calib_idx = 0
        idx = 0
        #for j in range(int(len(input)/500)):
        for j in range(1):
            temp = input.clone()
            temp_threshold = temp[calib_idx]
            temp[0:calib_idx] = temp_threshold
            quantize_asymmetric(temp,qbit)
            '''
            temp = temp.div(temp_threshold).mul(2**qbit-1)
            temp = temp.round()
            temp = temp.div(2**qbit-1).mul(temp_threshold)
            '''
            #temp_divergence = nn.KLDivLoss(reduction='batchmean')(input.log(), temp)
            temp_divergence = F.kl_div(temp, input, reduction='batchmean')
            if layer_divergence is None: 
                layer_divergence = temp_divergence
                layer_threshold = temp_threshold
                idx = calib_idx
            else:
                if layer_divergence > temp_divergence:
                    layer_divergence = temp_divergence
                    layer_threshold = temp_threshold
                    idx = calib_idx
            calib_idx += 1
        threshold[i] = layer_threshold
        print(f'==> calibrating relu{i} | calib_idx=[{idx}/{len(temp)}]')
    return threshold

def activation_ptq(model, qbit, calib_data):
    activation = {}
    layer_num = 0
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            handle = module.register_forward_hook(utils.get_activation(activation, layer_num))
            layer_num += 1
    _ = model(calib_data)

    threshold = calibration(activation, layer_num, qbit)
    model = set_qrelu(model, qbit, threshold)
    handle.remove()
    return model