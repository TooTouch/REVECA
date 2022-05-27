import torch
import numpy as np
import os 
import json
from metrics.evaluation_utils import split_gt_parts, split_pred_parts, evaluate_on_caption
from utils import convert_device
from tqdm.auto import tqdm

import logging
_logger = logging.getLogger('train')

def infer(args, model, tokenizer, dataloader):
    pred_dict = {}

    model.eval()
    with torch.no_grad():
        for idx, inputs in enumerate(tqdm(dataloader)):
            boundary_ids, frames, seg_features, tsn_features = inputs
            frames, seg_features, tsn_features = convert_device(frames, args.device), convert_device(seg_features, args.device), convert_device(tsn_features, args.device)
        
            inputs = {
                'frames'       : frames,
                'seg_features' : seg_features,
                'tsn_features' : tsn_features
            }
            # predict
            output = model.generate(
                inputs, 
                max_length             = args.gen_max_length, 
                decoder_start_token_id = tokenizer.encode('Subject')[0], 
                num_beams              = args.num_beams, 
                top_k                  = args.top_k,
                top_p                  = args.top_p
            )   
    
            pred_caps = tokenizer.batch_decode(output)
            pred_dict.update(dict(zip(boundary_ids, pred_caps)))
                
    return pred_dict


def evaluate(pred_dict, gt_dict, savepath):
    gt = split_gt_parts(gt_dict)
    pred = split_pred_parts(pred_dict)

    res_pred_sub = evaluate_on_caption(pred['subject'], gt['subject'])
    res_pred_bef = evaluate_on_caption(pred['before'], gt['before'])
    res_pred_aft = evaluate_on_caption(pred['after'], gt['after'])

    res = {}
    for metric in res_pred_sub.keys():
        res[f'subject_{metric}'] = res_pred_sub[metric]
        res[f'before_{metric}'] = res_pred_bef[metric]
        res[f'after_{metric}'] = res_pred_aft[metric]
        res[metric] = np.mean([res_pred_sub[metric], res_pred_bef[metric], res_pred_aft[metric]])

    _logger.info('\n ****************** Evaluation Results ******************')
    _logger.info(f'Subject: {res_pred_sub}')
    _logger.info(f'Status_Before: {res_pred_bef}')
    _logger.info(f'Status_After: {res_pred_aft}')

    with open(savepath + '_results.json', 'w') as fp:
        json.dump(res, fp, indent=4)

    