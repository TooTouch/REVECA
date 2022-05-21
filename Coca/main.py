from build_dataset import create_dataloader
from transformers import GPT2Tokenizer, get_cosine_schedule_with_warmup, Adafactor
from models import create_model
from train import training
from inference import infer, evaluate
from utils import torch_seed, load_checkpoint

import torch

import json
import os
import argparse
import wandb
import logging

from log import setup_default_logging

_logger = logging.getLogger('train')



def get_args(notebook=False):
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_val', action='store_true', default=False)
    parser.add_argument('--do_test', action='store_true', default=False)

    # dataset
    parser.add_argument('--datadir', type=str, default='/datasets/GEBC', help='dataset directory')
    parser.add_argument('--savedir', type=str, default='./output', help='save directory')
    parser.add_argument('--yaml_file', type=str, default='../config/captioning_config.yaml', help='config file path')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workeres in dataloader')
    parser.add_argument('--max_token_length', type=int, default=128, help='maximum token length')
    parser.add_argument('--max_sample_num', type=int, default=10, help='maximum frames of before or after')
    parser.add_argument('--use_saved_frame', action='store_true', default=False, help='use saved frames')

    # model
    parser.add_argument(
        '--image_modelname',
        type=str, 
        default='vit_huge_patch14_224_in21k', 
        choices=['vit_base_patch16_224', 'vit_huge_patch14_224_in21k'], 
        help='image model name'
    )
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
    parser.add_argument('--num_training_steps', type=int, default=200000, help='number of steps')
    parser.add_argument('--log_interval', type=int, default=50, help='log interval')
    parser.add_argument('--save_interval', type=int, default=10000, help='save interval')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='accumulation steps')
    parser.add_argument('--gpu_ids', type=str, default='0', help='number of gpus')
    parser.add_argument('--wandb', action='store_true', help='use wandb')

    # learning rate
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--num_warmup_steps', type=int, default=200, help='warmup rate')

    # generation
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint path')
    parser.add_argument('--gen_max_length', type=int, default=128, help='max length for generation')
    parser.add_argument('--num_beams', type=int, default=5, help='number for beam search')
    parser.add_argument('--top_k', type=int, default=None, help='top k generation')
    parser.add_argument('--top_p', type=int, default=None, help='top p generation')


    if notebook:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    # savedir
    if args.do_train:
        args.savedir = os.path.join(args.savedir,args.exp_name)
        os.makedirs(args.savedir, exist_ok=True)

    # gpus
    args.gpu_ids = list(map(int, args.gpu_ids.split(' ')))
    args.device = torch.device("cuda:" + str(args.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids if torch.cuda.is_available() else "-1"
    args.num_gpus = len(args.gpu_ids)
    assert args.num_gpus <= torch.cuda.device_count(), "Some of GPUs in args are unavailable, check your parameter."

    return args



if __name__ == '__main__':
    setup_default_logging()
    
    args = get_args()
    torch_seed(args.seed)

    # gpu device
    _logger.info('Device: {}'.format(args.device))

    if args.wandb and args.do_train:
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
        optimizer = Adafactor(
            model.parameters(), 
            lr=args.lr, 
            beta1=0.9, 
            weight_decay=0.01, 
            relative_step=False
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = args.num_warmup_steps, 
            num_training_steps = args.num_training_steps
        )

        # training
        training(args, model, trainloader, testloader, optimizer, scheduler, args.device)  

    elif args.do_val or args.do_test:
        model = load_checkpoint(args.checkpoint_path, model)

        testloader = create_dataloader(args, 'val' if args.do_val else 'test', tokenizer, test_mode=True)
        pred_dict = infer(args, model, tokenizer, testloader)
        
        filename = f'pred_beam{args.num_beams}_'
        filename += 'val' if args.do_val else 'test'
        savepath = os.path.join(args.savedir, args.exp_name, filename)

        # save predict
        with open(f'{savepath}.json','w') as fp:
            json.dump(pred_dict, fp, indent=4)

        if args.do_val:
            evaluate(
                pred_dict = pred_dict,
                gt_dict   = testloader.dataset.get_caption(),
                savepath  = savepath
            )

        
        