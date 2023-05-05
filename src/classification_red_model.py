import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
import os
import cv2
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights, resnet
import time
from datetime import datetime
import copy
import glob
import json
from sklearn.metrics import confusion_matrix, classification_report
import tqdm
# import seaborn as sns
import wandb
from torchvision.models import ResNet50_Weights
import numpy as np


class GelsightResNet(resnet.ResNet):
    def __init__(self, **kwargs):
        super(GelsightResNet, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.maxpool = nn.Identity()
        
class Pen(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        # self.image_names = os.listdir(root_dir)
        self.output_path = os.path.join(root_dir, 'pen.json')
        self.transform = transform

    def __len__(self):
        with open(self.output_path) as f:
            data = json.load(f)
            return len(data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with open(self.output_path) as f:
            data = json.load(f)
            image_name = data[idx]['RGB_image']
            img_path = os.path.join(self.root, image_name)
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            label = data[idx]['j']
        if self.transform:
            image = self.transform(image)
        return image, label, image_name


class GelsightDepth(Dataset):
    def __init__(self, root_dir, transform=None):
        self.output_path = os.path.join(root_dir, 'pen.json')
        with open(self.output_path) as f:
            self.data = json.load(f)
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
        data = self.data
        depth_name = data[idx]['Depth_image']
        rgb_name = data[idx]['RGB_image']
        #blur_name = data[idx]['Deapth_image_blur']
        #mask_name = data[idx]['Depth_image_masked']
        i = data[idx]['i']
        j = data[idx]['j']
        k = data[idx]['k']
        # print(img_name)
        #label_name = re.split(r'[_\d]'  ,depth_name)[1]
        img_path = os.path.join(self.root_dir, depth_name)
     
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) / 65535.0
     
        label = F.one_hot(((torch.tensor(data[idx]['j']*180/np.pi) % 360)).long(), num_classes=360)


        img_name = depth_name
        if self.transform:
            image = self.transform(image)

        return image, label, img_name

class GelsightRealDepth(Dataset):
    def __init__(self, root_dir, transform=None, fine_tune=False):
        # self.output_path = os.path.join(root_dir, 'out.json')
        self.root_dir = root_dir
        self.transform = transform
        if fine_tune:
            self.image_names = glob.glob(os.path.join(root_dir, '*.tiff'))
        else:
            self.image_names = glob.glob(os.path.join(root_dir, '*.tiff'))
    def __len__(self):
        #  with open(self.output_path) as f:
            # data = json.load(f)
            return len(self.image_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # print(self.image_names)
        img_path = self.image_names[idx]
        img_name = img_path.split('/')[-1]
        # print(img_name)
        label_name = img_name.split('_')[1]
   
        depth_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) / 65535.0
       
        depth_image = depth_image / (depth_image.max() - depth_image.min())
        
        image = depth_image
      
        label = None # TODO FIX: data[idx]['j']
        exit()
      
        if self.transform:
            image = self.transform(image)
        # print(image.shape)
        return image, label, img_name

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=50, device='cpu'):

    # writer = SummaryWriter('Tacto_Mask_Resnet50')
    wandb.init(
    # set the wandb project where this run will be logged
    project="pen-rot-sim",
    id="pen_rot_class_8",
    # track hyperparameters and run metadata
    config={
    "epochs": 9,
    })
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
                labels = labels.to(torch.float32)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # print(outputs.size())
                    # print(labels.size())
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == torch.argmax(labels, dim=1))
                tqdm_data.set_postfix_str(f'{phase} Loss: {loss.item():.4f} Acc: {torch.sum(preds == torch.argmax(labels, dim=1)).double() / inputs.size(0):.4f}')
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            wandb.log({f'{phase}_loss': epoch_loss}, step=epoch)
            wandb.log({f'{phase}_acc': epoch_acc}, step=epoch)
            loss_list.append(epoch_loss)
            acc_list.append(epoch_acc)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            torch.save(model.state_dict(), 'pen_rot_class.pth')


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    wandb.finish()

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
        # print(inputs.shape)
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



def view_images(inp, mean, std, title=None,):
    # print(inp.shape)
    inp = inp.numpy().transpose((1, 2, 0))
    # print(inp.shape)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    # print(inp.max())
    plt.imshow(inp/ inp.max())
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

def main():

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    train_transform = transforms.Compose([transforms.ToTensor(),
                                        #   transforms.CenterCrop((270, 362)), 
                                          transforms.Resize((224, 224)),  
                                        #   transforms.RandomHorizontalFlip(), 
                                        #   transforms.RandomVerticalFlip(),
                                        #   transforms.Normalize((0.1050,), (0.2312,)),
                                        # transforms.ConvertImageDtype(torch.float32),
                                        ])
    
    real_transform = transforms.Compose([transforms.ToTensor(), 
                                         transforms.Resize((224, 224), antialias=True), 
                                        #  transforms.Normalize((0.1050,), (0.2312,)),
                                        #  transforms.Grayscale(1),
                                         transforms.ConvertImageDtype(torch.float32),
                                         ])
    
    
    # simulated dataset
    # dataset = GelsightDepth('/home/rpmdt05/Code/gelsight/data_mod/', transform=train_transform)
    # dataset_sim = dataset
    # image1, label, img_name = dataset.__getitem__(0)
    # print(image.shape)
    # plt.title(img_name)
    # plt.imshow(image.numpy().transpose((1, 2, 0)))
    # plt.show()

    # gen = torch.Generator().manual_seed(42)
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=gen)

    # train_dataloaders = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    # val_dataloaders = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=8)
    # test_dataloaders = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)

    # mean, std = 0, 1
    # inputs, classes, _ = next(iter(train_dataloaders))
    # # print(inputs.shape)
    
    # grid = torchvision.utils.make_grid(inputs)
    # view_images(grid, mean, std, title=None)

  
    # dataloader = {'train': train_dataloaders, 'val': val_dataloaders}
    # dataset_sizes_train = {'train': train_dataset.__len__(), 'val': val_dataset.__len__(), 'test': test_dataset.__len__()}

    model = GelsightResNet(block=resnet.Bottleneck, layers=[3, 4, 6, 3])
    model_ftrs = model.fc.in_features
    # print(model_ftrs)
    # add dropout
    model.fc = nn.Sequential(
        nn.Dropout(0.8),
        nn.Linear(model_ftrs, 360)
    )
    # model.fc = nn.Linear(model_ftrs, 6)
    model = model.to(device)
    
    
    model = model.to(device)

    # # real dataset
    dataset = GelsightDepth('/home/rpmdt05/Code/Tacto_good/data/', transform=real_transform)

    gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=gen)

    train_dataloaders = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_dataloaders = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=8)
    test_dataloaders = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)

    dataloader = {'train': train_dataloaders, 'val': val_dataloaders}
    dataset_sizes_train = {'train': train_dataset.__len__(), 'val': val_dataset.__len__(), 'test': test_dataset.__len__()}

    num_epochs = 10
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()  
      
    # # training
    model = train_model(model, criterion, optimizer, scheduler, dataloader, dataset_sizes_train, num_epochs, device)
    

    # # testing
    model.load_state_dict(torch.load('/home/rpmdt05/Code/gelsight/src/pen_rot_class.pth'))
    # dataset_sizes_test = {'test': dataset.__len__()}
    # test_model(model, test_dataloaders, dataset_sizes_test, device)
    # dataset = GelsightDepth('/home/rpmdt05/Code/Tacto_good/data/', transform=real_transform)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    # image, label, img_name = dataset.__getitem__(0)
    # print(image.shape)
    # print(label)
    # plt.imshow(image.numpy().transpose((1, 2, 0)))
    # plt.title(img_name)
    # plt.show()
    # print(depth_image.dtype)
 
    # print(depth_image.min())
    # depth_image = depth_image / (depth_image.max() - depth_image.min())
    # print(depth_image.max())
    # cv2.imshow('depth', depth_image)
    # inputs = torch.tensor(depth_image, dtype=torch.float).to(device)
    # # Convert from 224x224 to 1x224x224
    # inputs = inputs.unsqueeze(0)

    # # Convert to a batch of 1
    # inputs = inputs.unsqueeze(0)
    
    # print(inputs.shape)
    # inputs = inputs.unsqueeze(-1)
    # inputs = inputs.permute((2, 0, 1))
    # inputs = inputs.unsqueeze(0)
    # print(inputs.shape)
    model.eval()
    # outputs = model(inputs.view((1, 1, 224, 224)))
    image, label, img_name = next(iter(test_dataloaders))
    for i in range(image.shape[0]):
        image = image.to(device)
        outputs = model(image)
        plt.imshow(image[i].cpu().numpy().transpose((1, 2, 0)))
        plt.title('true = {} pred = {}'.format(torch.argmax(label[i]), torch.argmax(outputs[i])))
        plt.show()
        # print('label = ', label)
        # image = image.to(device)
        # outputs = model(image)
        # print('pred = ',outputs)
        
if __name__ == '__main__':
    main()
