import numpy as np
import os
import random
import torch
from einops import repeat

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def load_checkpoint(checkpoint_path, model):
    weights = torch.load(checkpoint_path)
    model.load_state_dict(weights)

    print(f'load checkpoint from {checkpoint_path}')

    return model

def convert_device(inputs, device):
    if inputs is not None:
        for k in inputs.keys():
            inputs[k] = inputs[k].to(device)

    return inputs
    
def accuracy(outputs, targets, ignore=-100):
    _, pred = outputs.topk(5, -1, True, True)
    targets_len = (targets!=ignore).sum(-1)
    ignore_len = (targets==ignore).sum(-1)

    targets = repeat(targets, 'b l -> b l p', p=5)
    pred[targets==-100] = -100

    res = []
    for k in [1,5]:
        correct = (pred.eq(targets)[...,:k].sum(-1) >= 1).sum(-1)

        acc = ((correct - ignore_len) / targets_len).mean()
        res.append(acc)
        
    return res[0], res[1]
