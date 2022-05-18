from build_dataset import create_dataloader
from transformers import GPT2Tokenizer, GPT2Model


import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='/datasets/GEBC', help='dataset directory')
    parser.add_argument('--yaml_file', type=str, default='./config/captioning_config.yaml', help='config file path')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workeres in dataloader')
    parser.add_argument('--max_token_length', type=int, default=100, help='maximum token length')
    parser.add_argument('--max_sample_num', type=int, default=10, help='maximum frames of before or after')

    args = parser.parse_arges()

    return args




if __name__ == '__main__':
    args = get_args()

    # dataloader
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    trainloader = create_dataloader(args, 'train', tokenizer)
    testloader = create_dataloader(args, 'val', tokenizer)