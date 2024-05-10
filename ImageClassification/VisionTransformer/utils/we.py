import numpy as np
import torch
import torch.nn as nn

# sigmoid function
def sigmoid(x):
    # TODO: Implement sigmoid function
    return 1/(1 + np.exp(-x))
   

def get_valid_num(model, epoch, args):
    linear_weight = torch.cat([torch.norm(m1.weight.data, p=1, dim=1) / m1.weight.data.shape[1] for m1 in model.modules() if isinstance(m1, nn.Linear)])
    
    one_stage = args.epochs * 0.3
    two_stage = args.epochs * 0.6
        
    if epoch < one_stage:
        ratio_linear = args.ratio_linear * sigmoid(epoch/15)
    elif epoch < two_stage:
        ratio_linear = (args.ratio_linear / 2.5) * sigmoid((epoch-one_stage)/15)
    else:
        ratio_linear = (args.ratio_linear / 5) * sigmoid((epoch-two_stage)/15)

    print('epoch', epoch, ' ratio : ', ratio_linear)


    div_linear = len(linear_weight)*ratio_linear
    th_linear = sorted(linear_weight)[int(div_linear)]

    return th_linear
    
def evolution(model, epoch, args):
    th_linear = get_valid_num(model, epoch, args)
    for m1 in model.modules():
        if isinstance(m1, nn.Linear):
            with torch.no_grad():
                num_filters = len(m1.weight.data)
                m1_norm = torch.norm(m1.weight.data, p=1, dim=1) / m1.weight.data.shape[1]
                weight_sort, weight_sort_index = m1_norm.sort()

                invalid_num = 0
                for weight in weight_sort:
                    if weight <= th_linear:
                        invalid_num += 1

                # invalid_num = torch.min(invalid_num, int(float(num_filters) * args.max_threshold))

                valid_num = num_filters - invalid_num + 1

                invalid_index = weight_sort_index[:invalid_num]  
                valid_index = weight_sort_index[valid_num:]  
                weight_sort_index_zip = [list(t) for t in zip(valid_index, invalid_index)]

                # Crossover strategy.
                for [i,j] in weight_sort_index_zip:
                    a_value1 = torch.norm(m1.weight.data[j], p=1)
                    a_value2 = torch.norm(m1.weight.data[i], p=1)
                    alp = a_value1.abs() / (a_value1.abs() + a_value2.abs())
                    m1.weight.data[j] = m1.weight.data[j]*alp + m1.weight.data[i]*(1-alp)
                    if m1.bias is not None:
                        m1.bias.data[j] = m1.bias.data[j]*alp + m1.bias.data[i]*(1-alp)
  
    return model
