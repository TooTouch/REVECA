import os
import yaml
import json
from glob import glob 
import pandas as pd
import cv2
import h5py
import numpy as np
from einops import repeat
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import logging

_logger = logging.getLogger('train')

def load_from_yaml_file(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def load_json(filename):
    with open(filename, "r") as f:
        return json.loads(f.readlines()[0].strip("\n"))


class VideoExtraction:
    def __init__(self, args, split, transform=None):
        # define details of videos
        self.max_sample_num = args.max_sample_num
        self.video_info = pd.DataFrame({'video_path':glob(os.path.join(args.datadir,f'{split}/*'))})
        self.video_info['video_id'] = self.video_info['video_path'].apply(lambda x: x.split('/')[-1][:11])
        self.transform = transform
        
    def _get_frames(self, video_id, timestamps):
        video_path = self.video_info[self.video_info['video_id']==video_id]['video_path'].values[0]
        # read video
        cap = cv2.VideoCapture(video_path)
        
        # extract time list of video
        time_list = self.extract_time_list(cap)
        
        # sampling index
        sampled_times = self.time_sampling(time_list, timestamps)
        
        # sampling frames
        frames = self.extract_frames(cap, sampled_times)

        # same padding frames
        frames = self.same_padding(frames)

        return frames
        
    def extract_time_list(self, cap):
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps

        timestamp = np.linspace(0.,duration,frame_count)

        return timestamp

    def extract_frames(self, cap, sampled_times):
        frame_dict = {'boundary':None,'before':[],'after':[]}

        idx = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if idx in sum(sampled_times.values(), []):
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform != None:
                    img = self.transform(img)
                    
                if idx in sampled_times['boundary']:
                    frame_dict['boundary'] = img

                if idx in sampled_times['before']:
                    frame_dict['before'].append(img)

                if idx in sampled_times['after']:
                    frame_dict['after'].append(img)

            idx += 1

        if len(frame_dict['before']) > 0:
            frame_dict['before'] = torch.stack(frame_dict['before'])
        if len(frame_dict['after']) > 0:
            frame_dict['after'] = torch.stack(frame_dict['after'])

        return frame_dict

    def same_padding(self, frames):
        if len(frames['before']) < self.max_sample_num:
            empty_num = self.max_sample_num - len(frames['before'])
            
            if empty_num == self.max_sample_num:
                frames['before'] = repeat(frames['boundary'], 'c h w -> e c h w', e=empty_num)
            else:
                frames['before'] = torch.cat([
                    repeat(frames['before'][0], 'c h w -> e c h w', e=empty_num),
                    frames['before']
                ], dim=0)
        
        if len(frames['after']) < self.max_sample_num:
            empty_num = self.max_sample_num - len(frames['after'])

            if empty_num == self.max_sample_num:
                frames['after'] = repeat(frames['boundary'], 'c h w -> e c h w', e=empty_num)
            else:
                frames['after'] = torch.cat([
                    frames['after'],
                    repeat(frames['after'][-1], 'c h w -> e c h w', e=empty_num),
                ], dim=0)

        return frames

    def time_sampling(self, time_list, timestamps):
        sampled_time = dict()
        # find nearest matching
        nearest_timestamps = [10000.00 for i in range(len(timestamps))]
        nearest_times_idx = [0 for i in range(len(timestamps))]
        for idx, time in enumerate(time_list):
            time = float(time)
            for t_idx in range(len(timestamps)):
                ts = timestamps[t_idx]
                if abs(time - ts) < abs(nearest_timestamps[t_idx] - ts):
                    nearest_timestamps[t_idx] = time
                    nearest_times_idx[t_idx] = idx

        # get candidates
        sampled_time['boundary'] = [nearest_times_idx[1]]
        sampled_time['before'] = []
        sampled_time['after'] = []
        for idx, time in enumerate(time_list):
            time = float(time)
            if nearest_timestamps[0] <= time < nearest_timestamps[1]:
                sampled_time['before'].append(idx)
            elif nearest_timestamps[1] < time <= nearest_timestamps[2]:
                sampled_time['after'].append(idx)

        # start sampling
        if len(sampled_time['before']) > self.max_sample_num:
            sampling_step = len(sampled_time['before']) / self.max_sample_num
            sampled_idx = [int((idx + 1) * sampling_step) - 1 for idx in range(self.max_sample_num)]
            sampled_time['before'] = [sampled_time['before'][idx] for idx in sampled_idx]

        if len(sampled_time['after']) > self.max_sample_num:
            sampling_step = len(sampled_time['after']) / self.max_sample_num
            sampled_idx = [int(idx * sampling_step) for idx in range(self.max_sample_num)]
            sampled_time['after'] = [sampled_time['after'][idx] for idx in sampled_idx]

        return sampled_time    


class CaptionExtraction:
    def __init__(self, args, tokenizer):
        # define details of captions
        self.max_token_length = args.max_token_length
        self.tokenizer = tokenizer
        
    def get_tokens(self, caption, splitter='//'):
        
        # check caption
        splitted_caption = caption.split(splitter)
        assert len(splitted_caption) == 3, "Invalid: Caption has more than 3 parts: " + caption
        
        # add eos token
        caption = caption + self.tokenizer.eos_token
    
        # input_ids
        input_ids = self.tokenizer.encode(caption)

        # extract label_ids
        input_ids, label_ids = self.extract_label_ids(input_ids)

        # attention_mask
        attention_mask = [1] * len(input_ids)

        # padding
        input_ids, label_ids, attention_mask = self.padding_tokens(
            input_ids      = input_ids,
            label_ids      = label_ids,
            attention_mask = attention_mask,
            eos_id         = self.tokenizer.encode(self.tokenizer.eos_token)
        )

        input_ids, attention_mask = self.add_cls_token(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            cls_id         = self.tokenizer.encode(self.tokenizer.cls_token)
        )

        return torch.tensor(input_ids), torch.tensor(label_ids), torch.LongTensor(attention_mask)

    def padding_tokens(self, input_ids, label_ids, attention_mask, eos_id):
        """
        Add [CLS] token for CoCa
        """
        if len(input_ids) < self.max_token_length:
            pad_length = self.max_token_length - len(input_ids)
            input_ids += (eos_id * pad_length) 
            attention_mask += ([0] * pad_length)

            label_ids += ([-100] * pad_length)

        elif len(input_ids) > self.max_token_length:
            input_ids = input_ids[:self.max_token_length-1] + eos_id
            attention_mask = attention_mask[:self.max_token_length] 

            label_ids = input_ids[:self.max_token_length-1] + eos_id

        return input_ids, label_ids, attention_mask
    
    def add_cls_token(self, input_ids, attention_mask, cls_id):
        input_ids += cls_id
        attention_mask += [1]
        return input_ids, attention_mask

    def extract_label_ids(self, input_ids):
        return input_ids[:-1], input_ids[1:]


class BoundaryCaptioningDataset(Dataset, VideoExtraction, CaptionExtraction):
    def __init__(self, args, split, tokenizer, transform=None, test_mode=False):
        super(BoundaryCaptioningDataset).__init__()
        VideoExtraction.__init__(self, args, split, transform)
        CaptionExtraction.__init__(self, args, tokenizer)
        
        # test mode
        self.test_mode = test_mode

        # read yaml file
        self.yaml_file = args.yaml_file
        self.cfg = load_from_yaml_file(self.yaml_file)

        # read annotation
        self.split = split
        assert self.split in ['train', 'test', 'val'], "Invalid split: split must in 'train', 'test' and 'val'."
        self.annotation = load_json(self.cfg[f'{split}_annotation'])

        # read saved frames
        self.datadir = args.datadir
        self.use_saved_frame = args.use_saved_frame

        # read seg features
        self.use_seg_features = args.use_seg_features
        self.max_sample_num = args.max_sample_num

        # read tsn features
        self.use_tsn_features = args.use_tsn_features

        # make boundary list
        self.boundary_list = self.build_boundary_list()

    def __len__(self):
        return len(self.boundary_list)

    def __getitem__(self, idx):
        # boundary info
        boundary = self.boundary_list[idx]
        boundary_id = boundary['boundary_id']
        boundary_idx = int(boundary_id.split('_')[-1])
        video_id = boundary_id[:11]

        # time stamps
        timestamps = [boundary['prev_timestamp'], boundary['timestamp'], boundary['next_timestamp']]

        # frames info
        frames = self.get_frames(video_id, boundary_id, timestamps)
        seg_features = self.get_seg_features(boundary_id)
        tsn_features = self.get_tsn_features(video_id, boundary_idx)

        # train or test mode
        if not self.test_mode:
            caption = boundary['caption']        
            input_ids, label_ids, attention_mask = self.get_tokens(caption)

            return boundary_id, {'input_ids':input_ids, 'attention_mask':attention_mask}, frames, seg_features, tsn_features, label_ids
            
        else:
            return boundary_id, frames, seg_features, tsn_features
            

    def build_boundary_list(self):
        boundary_list = []
        for vid, boundaries in self.annotation.items():
            for b in boundaries:
                boundary_list.append(b)
        return boundary_list

    def get_caption(self):
        caption_dict = dict()
        for boundary in self.boundary_list:
            caption_dict[boundary['boundary_id']] = [boundary['caption']]
        return caption_dict

    def get_frames(self, video_id, boundary_id, timestamps):
        if self.use_saved_frame:
            frames = torch.load(
                os.path.join(self.datadir, 'frames', f'{self.split}/{boundary_id}.pt')
            )[boundary_id]
        else:
            frames = self._get_frames(video_id, timestamps)

        return frames

    def get_seg_features(self, boundary_id):
        seg_features = None

        if self.use_seg_features:
            seg_features = {}
            saved_seg_features = torch.load(os.path.join(self.datadir, 'seg_features', f'{self.split}/{boundary_id}.pt'))
            seg_features['before'] = saved_seg_features[:self.max_sample_num].to(torch.float).unsqueeze(1)
            seg_features['boundary'] = saved_seg_features[self.max_sample_num].to(torch.float).unsqueeze(0)
            seg_features['after'] = saved_seg_features[-self.max_sample_num:].to(torch.float).unsqueeze(1)    
    
        return seg_features

    def get_tsn_features(self, video_id, boundary_idx):
        tsn_features = None

        if self.use_tsn_features:
            filename = os.path.join(self.datadir, 'tsn_captioning_feature', f'{video_id}.hdf5')
            with h5py.File(filename, 'r') as h5:
                tsn_features = dict(
                    before=h5[str(boundary_idx)][:],
                    after=h5[str(boundary_idx + 1)][:]
                )

        return tsn_features


def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def create_dataloader(args, split, tokenizer, test_mode=False):
    # Load Data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_size,args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = BoundaryCaptioningDataset(
        args, split, tokenizer, transform, test_mode
    )

    samples_per_gpu = args.batch_size
    samples_per_batch = samples_per_gpu * args.num_gpus
    iters_per_batch = len(dataset) // samples_per_batch
    num_iters = iters_per_batch * args.num_training_steps

    if split == 'train':
        _logger.info("Train with {} samples per GPU.".format(samples_per_gpu))
        _logger.info("Total batch size {}".format(samples_per_batch))
        _logger.info("Total training steps {}".format(num_iters))


    dataloader = DataLoader(
        dataset, 
        batch_size  = samples_per_batch, 
        sampler     = make_data_sampler(dataset, split=='train'),
        num_workers = args.num_workers,
        pin_memory  = True
    )

    return dataloader
