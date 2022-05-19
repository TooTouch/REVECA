from build_dataset import create_dataloader
from transformers import GPT2Tokenizer, get_cosine_schedule_with_warmup, Adafactor
from models import create_model
from train import fit
from inference import infer
from utils import torch_seed


import torch

import argparse
import wandb
import logging

from log import setup_default_logging

_logger = logging.getLogger('train')



def get_args(notebook=False):
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_infer', action='store_true', default=False)

    # dataset
    parser.add_argument('--datadir', type=str, default='/datasets/GEBC', help='dataset directory')
    parser.add_argument('--savedir', type=str, default='./output', help='save directory')
    parser.add_argument('--yaml_file', type=str, default='./config/captioning_config.yaml', help='config file path')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workeres in dataloader')
    parser.add_argument('--max_token_length', type=int, default=100, help='maximum token length')
    parser.add_argument('--max_sample_num', type=int, default=10, help='maximum frames of before or after')

    # model
    parser.add_argument('--image_modelname', type=str, default='vit_base_patch16_224', choices=['vit_base_patch16_224'], help='image model name')
    parser.add_argument('--unimodal_modelname', type=str, default='gpt2', choices=['gpt2'], help='unimodal model name')
    parser.add_argument('--multimodal_modelname', type=str, default='gpt2', choices=['gpt2'], help='multimodal model name')
    parser.add_argument('--img_size', type=int, default=224, help='image size')
    parser.add_argument('--caption_loss_weight', type=float, default=1.,help='caption loss weight')
    parser.add_argument('--contrastive_loss_weight', type=float, default=1.,help='contrastive loss weight')
    parser.add_argument('--num_img_queries', type=int, default=256 ,help='number of image queries')
    parser.add_argument('--num_heads', type=int, default=8, help='number of attentional pooling heads')

    # training
    parser.add_argument('--seed', type=int, default=223, help='my birthday')
    parser.add_argument('--exp_name', type=str, default='VideoBoundaryCoCa', help='experiment name')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--log_interval', type=int, default=50, help='log interval')
    parser.add_argument('--save_interval', type=int, default=10, help='save interval')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='accumulation steps')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')

    # learning rate
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--num_warmup_steps', type=int, default=5, help='warmup rate')


    # generation



    if notebook:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    return args



if __name__ == '__main__':
    setup_default_logging()
    
    args = get_args()
    torch_seed(args.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # wandb
    wandb.init(name=args.exp_name, project='GEBC VideoCoCa', config=args)

    # tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'cls_token':'[CLS]'})


    # models
    model = create_model(args, tokenizer)
    model.to(args.device)
    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

    if args.do_train:
        # dataloader
        trainloader = create_dataloader(args, 'train', tokenizer)
        testloader = create_dataloader(args, 'val', tokenizer)
        
        # compile
        optimizer = Adafactor(model.parameters(), lr=args.lr, beta1=0.9, weight_decay=0.01, warmup_init=True)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = args.num_warmup_steps, 
            num_training_steps = args.epochs
        )
        # training
        fit(args, model, trainloader, testloader, optimizer, scheduler, device)

    elif args.do_infer:
        testloader = create_dataloader(args, 'test', tokenizer)
        infer(args, model, testloader, device)