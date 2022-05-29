import time
import os
import logging
import json
import wandb

from collections import OrderedDict

import torch

from utils import convert_device

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
    acc1_m = AverageMeter()
    acc5_m = AverageMeter()
    losses_m = AverageMeter()
    caption_losses_m = AverageMeter()
    contrastive_losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()

    for step in range(args.num_training_steps):    
        _, captions, frames, seg_features, tsn_features, labels = next(iter(trainloader))
        captions, labels = convert_device(captions, device), labels.to(device)
        frames, seg_features, tsn_features = convert_device(frames, device), convert_device(seg_features, device), convert_device(tsn_features, device)

        # optimizer condition
        opt_cond = (step + 1) % args.accumulation_steps == 0

        if opt_cond or step == 0:
            data_time_m.update(time.time() - end)

        # predict
        acc1, acc5, caption_loss, contrastive_loss = model(
            captions     = captions, 
            frames       = frames, 
            seg_features = seg_features, 
            tsn_features = tsn_features,
            labels       = labels, 
            return_loss   = True
        )

        loss = (caption_loss + contrastive_loss).mean()

        # loss for accumulation steps
        loss /= args.accumulation_steps        
        loss.backward()

        if opt_cond:
            # loss update
            optimizer.step()
            optimizer.zero_grad()
            
            scheduler.step()

            acc1_m.update(acc1.mean().item())
            acc5_m.update(acc5.mean().item())
            losses_m.update(loss.item()*args.accumulation_steps)
            caption_losses_m.update(caption_loss.mean().item()*args.accumulation_steps)
            contrastive_losses_m.update(contrastive_loss.mean().item()*args.accumulation_steps)
            
            step_time_m.update(time.time() - end)
        
            if ((step + 1) // args.accumulation_steps) % args.log_interval == 0 or step == 0: 
                _logger.info('TRAIN [{:>4d}/{}] '
                        'Acc@1: {acc1.val:>6.2%} ({acc1.avg:>6.2%}) '
                        'Acc@5: {acc5.val:>6.2%} ({acc5.avg:>6.2%}) '
                        'Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'Caption Loss: {caption_loss.val:>6.4f} ({caption_loss.avg:>6.4f}) '
                        'Contrastive Loss: {contrastive_loss.val:>6.4f} ({contrastive_loss.avg:>6.4f}) '
                        'LR: {lr:.3e} '
                        'Time: {step_time.val:.3f}s ({step_time.avg:.3f}s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        (step+1)//args.accumulation_steps, args.num_training_steps//args.accumulation_steps, 
                        acc1             = acc1_m,
                        acc5             = acc5_m,
                        loss             = losses_m, 
                        caption_loss     = caption_losses_m,
                        contrastive_loss = contrastive_losses_m,
                        lr               = optimizer.param_groups[0]['lr'],
                        step_time        = step_time_m,
                        data_time        = data_time_m))

                if args.wandb:
                    # wandb
                    metrics = OrderedDict(steps=step)
                    metrics.update([
                        ('train_acc1', acc1_m.val),
                        ('train_acc5', acc5_m.val),
                        ('train_loss', losses_m.val),
                        ('train_caption_loss', caption_losses_m.val),
                        ('train_constrastive_loss', contrastive_losses_m.val),
                        ('lr',optimizer.param_groups[0]['lr'])
                    ])
                    wandb.log(metrics)

        end = time.time()

        # checkpoint
        if ((step + 1) // args.accumulation_steps) % args.save_interval == 0:
            eval_metrics = validation(model, testloader, args.log_interval, device)
            save_results_and_model(args.exp_name, step, model, losses_m, eval_metrics, args.savedir)

            if args.wandb:
                # wandb
                metrics = OrderedDict(steps=step)
                metrics.update([
                    ('val_acc1', eval_metrics['acc1']),
                    ('val_acc5', eval_metrics['acc5']),
                    ('val_loss', eval_metrics['loss']),
                    ('val_caption_loss', eval_metrics['caption_loss']),
                    ('val_contrastive_loss', eval_metrics['contrastive_loss'])
                ])
                wandb.log(metrics)



def validation(model, dataloader, log_interval, device='cpu'):
    total_acc1 = 0
    total_acc5 = 0
    total_loss = 0
    total_caption_loss = 0
    total_contrastive_loss = 0

    model.eval()
    with torch.no_grad():
        for idx, (_, captions, frames, seg_features, tsn_features, labels) in enumerate(dataloader):
            captions, labels = convert_device(captions, device), labels.to(device)
            frames, seg_features, tsn_features = convert_device(frames, device), convert_device(seg_features, device), convert_device(tsn_features, device)
            
            # predict
            acc1, acc5, caption_loss, contrastive_loss = model(
                captions     = captions, 
                frames       = frames, 
                seg_features = seg_features, 
                tsn_features = tsn_features,
                labels       = labels, 
                return_loss   = True
            )
            
            loss = (caption_loss + contrastive_loss)

            # total loss and acc
            total_acc1 += acc1.mean().item()
            total_acc5 += acc5.mean().item()
            total_loss += loss.mean().item()
            total_caption_loss += caption_loss.mean().item()
            total_contrastive_loss += contrastive_loss.mean().item()
            
            if (idx + 1) % log_interval == 0 or idx == 0: 
                _logger.info('TEST [{idx:d}/{total:d}]: '
                             'Acc@1: {acc1:.2%} '
                             'Acc@5: {acc5:.2%} '
                             'Loss: {loss:.3f} '
                             'Contrastive Loss: {contrastive_loss:.3f} ' 
                             'Caption Loss: {caption_loss:.3f} '.format(
                                idx              = idx+1,
                                total            = len(dataloader),
                                acc1             = total_acc1/(idx+1),
                                acc5             = total_acc5/(idx+1),
                                loss             = total_loss/(idx+1),
                                contrastive_loss = total_contrastive_loss/(idx+1),
                                caption_loss     = total_caption_loss/(idx+1)
                             ))
                            
                
    return OrderedDict([
            ('acc1',total_acc1/len(dataloader)),
            ('acc5',total_acc5/len(dataloader)),
            ('loss',total_loss/len(dataloader)),
            ('caption_loss',total_caption_loss/len(dataloader)),
            ('contrastive_loss',total_contrastive_loss/len(dataloader))
        ])




def save_results_and_model(exp_name, step, model, train_metrics, eval_metrics, savedir):
    state = {'step':step, 'train_loss':train_metrics.avg, 'val_loss':eval_metrics['loss']}
    json.dump(state, open(os.path.join(savedir, f'{exp_name}_step{step}.json'),'w'), indent=4)

    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(savedir, f'{exp_name}_step{step}.pt'))
    
    _logger.info('Save a model')

