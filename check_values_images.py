import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch 
import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import json
import re
from PIL import Image


label_map = {'bishop': 0, 
             'king': 1,
             'knight': 2,
             'pawn': 3,
             'queen': 4,
             'rook': 5,
             }


class GelsightDepth(Dataset):
    def __init__(self, root_dir, transform=None):
        self.output_path = os.path.join(root_dir, 'out.json')
        self.root_dir = root_dir
        self.transform = transform
        # self.image_names = glob.glob(os.path.join(root_dir, '*.tif'))

    def __len__(self):
         with open(self.output_path) as f:
            data = json.load(f)
            return len(data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # img_name_list = os.listdir(image_path)
        with open(self.output_path) as f:
            data = json.load(f)
            depth_name = data[idx]['Depth_image']
            blur_name = data[idx]['Deapth_image_blur']
            mask_name = data[idx]['Depth_image_masked']
            i = data[idx]['i']
            j = data[idx]['j']
            k = data[idx]['k']
            # print(img_name)
            label_name = re.split(r'[_\d]'  ,depth_name)[1]
            img_path = os.path.join(self.root_dir, blur_name)
        # print(img_path)
        # print(img_path)
        # image = Image.open(img_path)
        # image = np.array(image)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # print(image.max())
        # print(image.dtype)
        # print(image.shape)
        label = label_map[label_name]
        # image = np.expand_dims(image [:, :, 0], 2)
        # print(image.shape)
        # image = np.clip(image, 0, 1)
        # image = np.transpose(image, (2, 0, 1))
        # image = image.permute(2, 0, 1)
        img_name = blur_name
        if self.transform:
            image = self.transform(image)
        # print(image.shape)
        return image, label, img_name


def check_values(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    print(img.shape)
    print(np.max(img))
    print(np.min(img))
    plt.imshow(img[:, :, 0])
    plt.show()

def normalize_values(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, k, _ in tqdm.tqdm(loader):
        # images = images.permute(0, 3, 1, 2)
        # print(images.shape)
        b, c, h, w = images.shape
        # print(images.shape)
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                      cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                            cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
      snd_moment - fst_moment ** 2)        
    return mean,std

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize([0.1050, 0.1050, 0.1050], [0.2312, 0.2312, 0.2312]),
    ])

    path = '/home/rpmdt05/Code/Tacto_good/data_aug/data_mod'
    # check_values(path)
    dataset = GelsightDepth(path, transform=train_transform)
    image, label, img_name = dataset.__getitem__(0)
    # print(image.shape)
    plt.title(img_name)
    plt.imshow(image.numpy().transpose(1, 2, 0))
    plt.show()
    gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=gen)
    train_dataloaders = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    normal_values = normalize_values(train_dataloaders)
    print(normal_values)

if __name__ == '__main__':
    main()