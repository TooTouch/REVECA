import time
import os
import logging
import json
import wandb

from collections import OrderedDict

import torch

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



def training(model, dataloader, optimizer, log_interval, accumulation_steps=1, device='cpu'):   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for idx, (_, captions, frames, labels) in enumerate(dataloader):
        # optimizer condition
        opt_cond = (idx + 1) % accumulation_steps == 0

        if opt_cond or idx == 0:
            data_time_m.update(time.time() - end)
        
        captions, frames, labels = captions.to(device), frames.to(device), labels.to(device)

        # predict
        loss = model(captions=captions, frames=frames, labels=labels, return_loss=True)
        # loss for accumulation steps
        loss /= accumulation_steps        
        loss.backward()

        if opt_cond:
            # loss update
            optimizer.step()
            optimizer.zero_grad()

            losses_m.update(loss.item()*accumulation_steps)
            
            batch_time_m.update(time.time() - end)
        
            if (idx // accumulation_steps) % log_interval == 0 and idx != 0: 
                _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        (idx+1)//accumulation_steps, len(dataloader)//accumulation_steps, 
                        loss       = losses_m, 
                        lr         = optimizer.param_groups[0]['lr'],
                        batch_time = batch_time_m,
                        data_time  = data_time_m))
   
        end = time.time()
    
    return OrderedDict([('loss',losses_m.avg)])


def evaluate(model, dataloader, log_interval, device='cpu'):
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        for idx, (_, captions, frames, labels) in enumerate(dataloader):
            captions, frames, labels = captions.to(device), frames.to(device), labels.to(device)
            
            # predict
            loss = model(captions=captions, frames=frames, labels=labels, return_loss=True)
        
            # total loss and acc
            total_loss += loss.item()
            
            if idx % log_interval == 0 and idx != 0: 
                _logger.info('TEST [%d/%d]: Loss: %.3f' % 
                            (idx+1, len(dataloader), total_loss/(idx+1)))
                
    return OrderedDict([('loss',total_loss/len(dataloader))])




def save_results_and_model(exp_name, epoch, model, train_metrics, eval_metrics, savedir):
    state = {'epoch':epoch, 'train_loss':train_metrics['loss'], 'val_loss':eval_metrics['loss']}
    json.dump(state, open(os.path.join(savedir, f'{exp_name}.json'),'w'), indent=4)

    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(savedir, f'{exp_name}.pt'))
    
    _logger.info('Save a model')


def fit(args, model, trainloader, testloader, optimizer, scheduler, device='cpu'):
    savedir = os.path.join(args.savedir,args.exp_name)
    os.makedirs(savedir, exist_ok=True)

    for epoch in range(args.epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{args.epochs}')
        train_metrics = training(model, trainloader, optimizer, args.log_interval, args.accumulation_steps, device)
        eval_metrics = evaluate(model, testloader, args.log_interval, device)

        scheduler.step()

        # wandb
        metrics = OrderedDict(epoch=epoch)
        metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
        metrics.update([('val_' + k, v) for k, v in eval_metrics.items()])
        wandb.log(metrics)
    
        # checkpoint
        if args.save_interval % epoch == 0:
            save_results_and_model(args.exp_name, epoch, model, train_metrics, eval_metrics, savedir)