from __future__ import print_function, division
import os
import cv2
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class SastaDataset(Dataset):
    """Obstacle avoidance dataset"""

    def __init__(self, txt_file, img_dir, transform=None):
        self.ground_truth_values = self.preprocess(txt_file)
        self.img_dir = img_dir
        self.transform = transform

    def preprocess(self, txt_file):
        data = open(txt_file, encoding='utf8').read().split('\n')
        data = [line for line in data][1:-1]                            # Drop first and last rows
        data = [item.split('\t') for item in data]
        return data

    def __len__(self):
        return len(self.ground_truth_values)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir, self.ground_truth_values[idx][-1])
        # Not using io.imread coz image is by default in RGBA space - so don't know the function in skimage to convert it in RGB
        # image = io.imread(img_path)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGBA2RGB)
        image = np.clip(np.asarray(image, dtype=float)/255, 0, 1)
        waypoint = np.array(self.ground_truth_values[idx][1:8])
        waypoint = waypoint.astype('float')
        sample = {'image': image, 'waypoint': waypoint}

        if self.transform:
            sample = self.transform(sample)

        return sample  
      
class Rescale(object):
    """Rescale the image to a desired input value"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, waypoint = sample['image'], sample['waypoint']
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {'image': img, 'waypoint': waypoint}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, waypoint = sample['image'], sample['waypoint']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'waypoint': torch.from_numpy(waypoint)}


if __name__ == "__main__":

    dataset = SastaDataset(txt_file="../Dataset/manual_1/manual_1/airsim_rec.txt", img_dir="../Dataset/manual_1/manual_1/images", transform=transforms.Compose([Rescale((432, 768)),ToTensor()]))
    
    # Test Cases
    
    #____________________________Checking Dataset class___________________________________
    # fig = plt.figure()
    # for i in range(len(sd)):
    #     sample = sd[i]
    #     print(i, sample['image'].shape, sample['waypoint'].shape)
    #     ax = plt.subplot(1,4,i+1)
    #     plt.tight_layout()
    #     ax.set_title('Sample #{}'.format(i))
    #     ax.axis('off')
    #     plt.imshow(sample['image'])
    #     # cv2.imshow(f'image_{i}', sample['image'])
    #     print(sample['waypoint'], f'corresponding to image {sd.ground_truth_values[i][-1]}')

    #     if i == 3:
    #         plt.show()
    #         # cv2.waitKey(0)
    #         break

    #____________________________Checking Rescale class__________________________________
    # scale = Rescale((400, 400))
    # composed = transforms.Compose([scale])
    # fig = plt.figure()
    # sample = sd[3]
    # for i, tsfrm in enumerate([scale, composed]):
    #     transformed_sample = tsfrm(sample)

    #     ax = plt.subplot(1, 3, i + 1)
    #     plt.tight_layout()
    #     ax.set_title(type(tsfrm).__name__)
    #     plt.imshow(transformed_sample['image'])

    # plt.show()

    #___________________________Checking the iteration on dataset_________________________
    # for i in range(len(dataset)):
    #     sample = dataset[i]
    #     print(i, sample['image'].size(), sample['waypoint'].size())
    #     if i == 3:
    #         break

    #___________________________Batch Wise images_______________________________
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # Frame stacking Wrapper to combine images in a dataset before shuffling ??
    
    def show_waypoint_batch(sample_batched):
        """Show image with waypoint for a batch of samples."""
        images_batch, waypoint_batch = sample_batched['image'], sample_batched['waypoint']
        batch_size = len(images_batch)
        im_size = images_batch.size(2)
        grid_border_size = 2

        grid = utils.make_grid(images_batch)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.title('Batch from dataloader')

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['waypoint'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_waypoint_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break