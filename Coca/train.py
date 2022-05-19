import time
import os
import logging
import json
import wandb

from collections import OrderedDict

import torch

from utils import convert_device, agg_inputs_to_batch

_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def training(
    args, model, trainloader, testloader, optimizer, scheduler, device='cpu'
):   
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for step in range(args.num_training_steps):    
        inputs = next(iter(trainloader))
        _, captions, frames, labels = agg_inputs_to_batch(inputs)
        # optimizer condition
        opt_cond = (step + 1) % args.accumulation_steps == 0

        if opt_cond or step == 0:
            data_time_m.update(time.time() - end)
        
        captions, frames, labels = convert_device(captions, device), convert_device(frames, device), labels.to(device)

        # predict
        loss = model(captions=captions, frames=frames, labels=labels, return_loss=True)
        # loss for accumulation steps
        loss /= args.accumulation_steps        
        loss.backward()

        if opt_cond:
            # loss update
            optimizer.step()
            optimizer.zero_grad()
            
            scheduler.step()

            losses_m.update(loss.item()*args.accumulation_steps)
            
            step_time_m.update(time.time() - end)
        
            if ((step + 1) // args.accumulation_steps) % args.log_interval == 0 or step == 0: 
                _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'LR: {lr:.3e} '
                        'Time: {step_time.val:.3f}s ({step_time.avg:.3f}s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        (step+1)//args.accumulation_steps, args.num_training_steps//args.accumulation_steps, 
                        loss       = losses_m, 
                        lr         = optimizer.param_groups[0]['lr'],
                        step_time  = step_time_m,
                        data_time  = data_time_m))

        end = time.time()

        # checkpoint
        if ((step + 1) // args.accumulation_steps) % args.save_interval == 0:
            eval_metrics = validation(model, testloader, args.log_interval, device)
            save_results_and_model(args.exp_name, step, model, losses_m, eval_metrics, args.savedir)

            if args.wandb:
                # wandb
                metrics = OrderedDict(steps=step)
                metrics.update([('train_loss', losses_m.avg)])
                metrics.update([('val_loss', eval_metrics['loss'])])
                wandb.log(metrics)



def validation(model, dataloader, log_interval, device='cpu'):
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            if idx == 30:
                break
            _, captions, frames, labels = agg_inputs_to_batch(inputs)
            captions, frames, labels = convert_device(captions, device), convert_device(frames, device), labels.to(device)
            
            # predict
            loss = model(captions=captions, frames=frames, labels=labels, return_loss=True)
        
            # total loss and acc
            total_loss += loss.item()
            
            if (idx + 1) % log_interval == 0 and idx != 0: 
                _logger.info('TEST [%d/%d]: Loss: %.3f' % 
                            (idx+1, len(dataloader), total_loss/(idx+1)))
                
    return OrderedDict([('loss',total_loss/len(dataloader))])




def save_results_and_model(exp_name, step, model, train_metrics, eval_metrics, savedir):
    state = {'step':step, 'train_loss':train_metrics.avg, 'val_loss':eval_metrics['loss']}
    json.dump(state, open(os.path.join(savedir, f'{exp_name}_step{step}.json'),'w'), indent=4)

    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(savedir, f'{exp_name}_step{step}.pt'))
    
    _logger.info('Save a model')

