import numpy as np
import os
import random

import torch

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
    for k in inputs.keys():
        inputs[k] = inputs[k].to(device)

    return inputs


def agg_inputs_to_batch(inputs, test_mode=False):
    boundary_ids = inputs[0]

    if not test_mode:
        captions = {
            'input_ids':torch.stack(inputs[1]),
            'attention_mask':torch.stack(inputs[2])
        }

        frames = {
            'boundary':torch.stack(inputs[3]),
            'before':torch.stack(list(inputs[4]), dim=0),
            'after':torch.stack(list(inputs[5]), dim=0)
        }

        labels = torch.stack(inputs[6])

        return boundary_ids, captions, frames, labels
    else:
        frames = {
            'boundary':torch.stack(inputs[1]),
            'before':torch.stack(list(inputs[2]), dim=0),
            'after':torch.stack(list(inputs[3]), dim=0)
        }

        return boundary_ids, frames