from build_dataset import create_dataloader
from transformers import GPT2Tokenizer, GPT2Model
from models import create_model

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--datadir', type=str, default='/datasets/GEBC', help='dataset directory')
    parser.add_argument('--yaml_file', type=str, default='./config/captioning_config.yaml', help='config file path')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workeres in dataloader')
    parser.add_argument('--max_token_length', type=int, default=100, help='maximum token length')
    parser.add_argument('--max_sample_num', type=int, default=10, help='maximum frames of before or after')

    # model
    parser.add_argument('--caption_loss_weight', type=float, default=1.,help='caption loss weight')
    parser.add_argument('--contrastive_loss_weight', type=float, default=1.,help='contrastive loss weight')
    parser.add_argument('--num_img_queries', type=int, default=256 ,help='number of image queries')
    parser.add_argument('--num_heads', type=int, default=8, help='number of attentional pooling heads')

    args = parser.parse_arges()

    return args


if __name__ == '__main__':
    args = get_args()

    # dataloader
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    trainloader = create_dataloader(args, 'train', tokenizer)
    testloader = create_dataloader(args, 'val', tokenizer)

    # models
    model = create_model(args, tokenizer)
    