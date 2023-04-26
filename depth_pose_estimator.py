import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os
import cv2
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights, resnet
import pandas as pd
import time
import copy
import glob
import json
import re
from sklearn.metrics import confusion_matrix, classification_report
import tqdm
import seaborn as sns
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

label_map = {'bishop': 0, 
             'king': 1,
             'knight': 2,
             'pawn': 3,
             'queen': 4,
             'rook': 5,
             }


class GelsightResNet(resnet.ResNet):
    def __init__(self, **kwargs):
        super(GelsightResNet, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        # self.avgpool = nn.Identity()
        
class GelsightDepthPose(Dataset):
    def __init__(self, root_dir, transform=None):
        self.output_path = os.path.join(root_dir, 'king.json')
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
            img_name = data[idx]['Depth_image'].split('.')[0]+'.tiff'
            # print(img_name)
            # label_name = re.split(r'[_\d]'  ,img_name)[1]
            pose = data[idx]['j']
            img_path = os.path.join(self.root_dir, img_name)
        # print(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = image / 65535
        masked_image = np.where(image > 0, 1, 0)
        # print(image.shape)
        # label = label_map[label_name]
        masked_image = np.expand_dims(image [:, :, 0], 2)
        # print(image.shape)
        # image = np.clip(image, 0, 1)
        # image = np.transpose(image, (2, 0, 1))
        # image = image.permute(2, 0, 1)
        # print(pose)
        if self.transform:
            masked_image = self.transform(masked_image)
        # print(image.shape)
        return masked_image, pose, img_name

class GelsightRealDepth(Dataset):
    def __init__(self, root_dir, transform=None):
        self.output_path = os.path.join(root_dir, 'king.json')
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
            img_name = data[idx]['Depth_image'].split('.')[0]+'.png'
            # print(img_name)
            label_name = re.split(r'[_\d]'  ,img_name)[1]
            img_path = os.path.join(self.root_dir, img_name)
        # print(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # print(image.shape)
        label = label_map[label_name]
        # print(label)
        image = image / 65535 * 255 
        image = np.expand_dims(image , 2)
        # print(image.shape)
        # image = np.clip(image, 0, 1)
        # image = np.transpose(image, (2, 0, 1))
        # image = image.permute(2, 0, 1)
        if self.transform:
            image = self.transform(image)
        # print(image.shape)
        return image, label, img_name

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, device='cpu'):

    writer = SummaryWriter('Tacto_Resnet50')
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            loss_list = []
            acc_list = []
            tqdm_data = tqdm.tqdm(dataloaders[phase], total=len(dataloaders[phase]))
            for inputs, labels, img_name in tqdm_data:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                tqdm_data.set_postfix_str(f'{phase} Loss: {loss.item():.4f} Acc: {torch.sum(preds == labels.data).double() / inputs.size(0):.4f}')
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            writer.add_scalar(f'{phase}_loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}_acc', epoch_acc, epoch)
            loss_list.append(epoch_loss)
            acc_list.append(epoch_acc)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    writer.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, dataloaders, dataset_sizes, device='cpu'):
    model.eval()
    running_corrects = 0
    y_pred = []
    y_true = []
    incorrect_names = []
    for inputs, labels, img_name in tqdm.tqdm(dataloaders):
        # print(inputs.shape)
        inputs = inputs.to(device)
        labels = labels.to(device)
        y_true.extend(labels.data.cpu().numpy())
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.data.cpu().numpy())
        running_corrects += torch.sum(preds == labels.data)
        incorrect = torch.argwhere(preds != labels.data)
        for i in incorrect:
            incorrect_names.append(img_name[i])
    test_acc = running_corrects.double() / dataset_sizes['test']
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(f'Incorrect: {len(incorrect_names)}')
    print('Incorrect names: ', incorrect_names)
    print(f'Test Acc: {test_acc:.4f}')
    print(conf_matrix)
    print(classification_report(y_true, y_pred, digits=4))
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.title('Confusion matrix')  
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

def normalize_values(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, k, _ in tqdm.tqdm(loader):
        images = images.permute(0, 3, 1, 2)
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

def view_images(inp, mean, std, title=None,):
    inp = inp.numpy().transpose((1, 2, 0))
    print(inp.shape)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    print(inp.max())
    plt.imshow(inp/ inp.max())
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

def main():
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    train_transform = transforms.Compose([transforms.ToTensor(),
                                        #   transforms.CenterCrop((270, 362)), 
                                          transforms.Resize((224, 224)),  
                                          transforms.RandomHorizontalFlip(), 
                                          transforms.RandomVerticalFlip(),
                                          transforms.Normalize((3.6842,), (11.6374,)),
                                          transforms.Grayscale(1),
                                          transforms.ConvertImageDtype(torch.float32)
                                        ])
    
    real_transform = transforms.Compose([transforms.ToTensor(), 
                                         transforms.Resize((224, 224)), 
                                         transforms.Normalize((3.6842,), (11.6374,)),
                                         transforms.Grayscale(1),
                                            transforms.ConvertImageDtype(torch.float32)])
    
    # simulated dataset
    dataset = GelsightDepthPose('data', transform=train_transform)
    image, pose, img_name = dataset.__getitem__(0)
    # print(image.shape)
    # print(pose)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    # normal_values = normalize_values(dataloader)
    # print(normal_values)

    gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=gen)

    train_dataloaders = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

    mean, std = torch.tensor([3.6842,]), torch.tensor([11.6374,])
    inputs, classes, _ = next(iter(train_dataloaders))
    print(inputs.shape)
    
    grid = torchvision.utils.make_grid(inputs)
    view_images(grid, mean, std, title=None)
    plt.show()

    val_dataloaders = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=8)
    test_dataloaders = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)
    dataloader = {'train': train_dataloaders, 'val': val_dataloaders}
    dataset_sizes = {'train': train_dataset.__len__(), 'val': val_dataset.__len__(), 'test': test_dataset.__len__()}

    model = GelsightResNet(block=resnet.Bottleneck, layers=[3, 4, 6, 3])
    model_ftrs = model.fc.in_features
    # print(model_ftrs)
    model.fc = nn.Linear(model_ftrs, 1)
    model = model.to(device)

    # real dataset
    # dataset = GelsightRealDepth('real_depth/gs_data', transform=real_transform)
    # image, label, img_name = dataset.__getitem__(0)
    # print(image.shape)
    # print(label)
    # plt.imshow(image.numpy().transpose((1, 2, 0)))
    # # plt.show()
    # dataset_sizes = {'test': dataset.__len__()}
    # test_dataloaders = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

    num_epochs = 25
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.MSELoss()
    
    # training
    model = train_model(model, criterion, optimizer, scheduler, dataloader, dataset_sizes, num_epochs, device)
    torch.save(model.state_dict(), 'depth_pose_king.pth')

    # testing
    model.load_state_dict(torch.load('depth_pose_king.pth'))
    test_model(model, test_dataloaders, dataset_sizes, device)

    #testing on real data
     
    

if __name__ == '__main__':
    main()
