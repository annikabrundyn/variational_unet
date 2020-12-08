import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split

import pytorch_lightning as pl
from PIL import Image


class NYUDepth(Dataset):

    def __init__(self,
                 root_dir,
                 image_set='train',
                 frames_per_sample=1,
                 resize=1,
                 img_transform=None,
                 target_transform=None
                 ):
        self.root_dir = root_dir
        self.image_set = image_set

        new_height = round(480*resize)
        new_width = round(640*resize)

        if not img_transform:
            self.img_transform = transforms.Compose([transforms.Resize((new_height, new_width)),
                                                     transforms.ToTensor()])
        else:
            self.img_transform = img_transform

        if not target_transform:
            self.target_transform = transforms.Compose([transforms.Resize((new_height, new_width)),
                                                        transforms.ToTensor()])
        else:
            self.target_transform = target_transform

        # create dict with each video name (of diff. scenes) as a key and a list of corresponding frames for that video
        self.videos = {}
        self.frames_per_sample = frames_per_sample
        img_list = self.read_image_list(os.path.join(root_dir, '{:s}.csv'.format(image_set)))

        for (img_filename, target_filename) in img_list:
            key, jpg = img_filename.split('/')[2:]
            frame_num = jpg.split('.')[0]
            if key in self.videos:
                self.videos[key].append(int(frame_num))
            else:
                self.videos[key] = [int(frame_num)]
        #{'v1':[1,2,3,4,5,6,7,8], 'v2':[1,2,3,4,5,6]}

        # sort the frames and create samples containing k frames per sample
        self.all_samples = []
        for key, value in self.videos.items():
            value.sort()

            if self.frames_per_sample > 1:
                step_size = 1 # sample overlap size
                for i in range(0, len(value)-self.frames_per_sample, step_size):
                    frames = value[i:i+self.frames_per_sample]

                    # Randomly drop one frame to reduce correlation between samples
                    if self.frames_per_sample > 2:
                        rand_idx = random.randint(0, self.frames_per_sample - 2)
                        _ = frames.pop(rand_idx)

                    self.all_samples += [(key, frames)]

            # only using single frame
            else:
                self.all_samples += ([(key, [i]) for i in self.videos[key]])
        print("len of all samples:", len(self.all_samples))

        # shuffle
        random.shuffle(self.all_samples)


    def read_image_list(self, filename):
        """
        Read one of the image index lists
        Parameters:
            filename (string):  path to the image list file
        Returns:
            list (int):  list of strings that correspond to image names
        """
        list_file = open(filename, 'r')
        img_list = []
        while True:
            next_line = list_file.readline()
            if not next_line:
                break
            jpg, png = next_line.rstrip().split(',')

            img_list.append((jpg, png))
        return img_list

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        sample = self.all_samples[index]
        video_name = sample[0]
        frames = sample[1]

        images = []
        for frame in frames:
            img_path = os.path.join(self.root_dir, 'nyu2_{}'.format(self.image_set), video_name, '{}.jpg'.format(frame))
            image = Image.open(img_path)
            image = self.img_transform(image)   #TODO: applying multiple times
            images.append(image)

        image_tensor = torch.stack(images)
        image_tensor = torch.squeeze(image_tensor, 1)

        target_path = os.path.join(self.root_dir, 'nyu2_{}'.format(self.image_set), video_name, '{}.png'.format(frames[-1]))
        target = Image.open(target_path)
        target = self.target_transform(target)

        return image_tensor, target


class NYUDepthDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            frames_per_sample: int = 1,
            resize: float = 0.5,
            val_split: float = 0.2,
            num_workers: int = 8,
            batch_size: int = 32,
            seed: int = 42,
            shuffle_val: bool = False,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.frames_per_sample = frames_per_sample
        self.resize = resize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle_val = shuffle_val

        self.dataset = NYUDepth(self.data_dir, frames_per_sample=self.frames_per_sample, resize=self.resize)

        val_len = int(val_split * len(self.dataset))
        train_len = len(self.dataset) - val_len

        self.trainset, self.valset = random_split(self.dataset, lengths=[train_len, val_len])

    def train_dataloader(self):
        loader = DataLoader(self.trainset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.valset,
                            batch_size=self.batch_size,
                            shuffle=self.shuffle_val,
                            num_workers=self.num_workers)
        return loader

