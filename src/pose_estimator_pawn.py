import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os
import io
import cv2
import matplotlib.pyplot as plt
from torchvision.models import resnet50
#import pandas as pd
import time
import copy
import json
import numpy as np
import re
#from torchmetrics.classification import MulticlassAccuracy
#from sklearn.metrics import confusion_matrix
import tqdm
import transforms3d as t3d
#import seaborn as sns


class GelsightDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.output_path = os.path.join(root_dir, 'king.json')
        self.root_dir = root_dir
        self.transform = transform

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
            img_name = data[idx]['RGB_image']
            label = [data[idx]['x'], data[idx]['y'], data[idx]['z'], data[idx]['qx'], data[idx]['qy'], data[idx]['qz'], data[idx]['qw']]
            label = torch.from_numpy(np.asarray(label).astype(np.float32))
            img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label
    
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, device='cpu'):
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
            for inputs, labels in tqdm_data:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print(outputs.shape)
                    preds = outputs
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                tqdm_data.set_postfix_str(f'{phase} Loss: {loss.item():.4f} Acc: {torch.sum((preds - labels.data) < 5.0).double() / inputs.size(0):.4f}')
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
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
    #conf_matrix = confusion_matrix(y_true, y_pred)
    print(f'Incorrect: {len(incorrect_names)}')
    print('Incorrect names: ', incorrect_names)
    print(f'Test Acc: {test_acc:.4f}')
    #print(conf_matrix)
    plt.figure(figsize=(10, 10))
    #sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.title('Confusion matrix')  
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

def main():
    torch.cuda.empty_cache()
    train_transform = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Resize((224, 224)), 
                                          transforms.RandomRotation(45), 
                                          transforms.RandomHorizontalFlip(), 
                                          transforms.RandomVerticalFlip(),])
    
    dataset = GelsightDataset('tacto_dataset/temp', transform=train_transform)
    dataset_size = dataset.__len__()
    gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=gen)
    print('full dataset = ', dataset.__len__())
    print('train_dataset = ', train_dataset.__len__())
    print('valid_dataset = ', val_dataset.__len__())
    print('test_dataset = ', test_dataset.__len__())
    # image, label = train_dataset.__getitem__(10)
    # print(label)
    # print(image.shape)
    # print(dataset.__len__())
    # plt.imshow(image.permute(1, 2, 0))
    # plt.show()
    train_dataloaders = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_dataloaders = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=8)
    test_dataloaders = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)
    dataloader = {'train': train_dataloaders, 'val': val_dataloaders}
    dataset_sizes = {'train': train_dataset.__len__(), 'val': val_dataset.__len__(), 'test': test_dataset.__len__()}
    model = torchvision.models.resnet50(pretrained=True)
    print(model)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    num_epochs = 25
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model_ftrs = model.fc.in_features
    model.fc = nn.Linear(model_ftrs, 7)
    criterion = nn.MSELoss()
    model = model.to(device)
    model = train_model(model, criterion, optimizer, scheduler, dataloader, dataset_sizes, num_epochs, device)
    # torch.save(model.state_dict(), 'classifier.pth')
    model.load_state_dict(torch.load('pawn_pose_estimator.pth'))
    #test_model(model, test_dataloaders, dataset_sizes, device)


if __name__ == "__main__":
    main()